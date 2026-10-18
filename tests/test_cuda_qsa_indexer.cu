#include "kernels/cuda/qsa.h"
#include "kernels/cuda/matmul.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
void require(bool x,const char* what){if(!x){std::fprintf(stderr,"QSA indexer: %s\n",what);std::exit(2);}}
struct Buffer {
    void* data;
    explicit Buffer(size_t bytes){check(cudaMalloc(&data,std::max(size_t(4),bytes)));}
    ~Buffer(){cudaFree(data);}
    float* f(size_t offset=0){return static_cast<float*>(data)+offset;}
};

// Replay a bounded real snapshot without loading the full model. Rebuild all
// pooled history from saved raw K, so changing normalization does not reuse
// stale pooled values. Prefix Q is irrelevant to the retained current rows.
void replay_snapshot(const char* input,const char* output) {
    FILE* source=std::fopen(input,"rb");require(source,"open snapshot");
    char magic[8];uint32_t h[10],flags;float eps;
    require(std::fread(magic,1,8,source)==8&&!std::memcmp(magic,"QSAIDX2\0",8)&&
        std::fread(h,4,10,source)==10&&std::fread(&eps,4,1,source)==1&&
        std::fread(&flags,4,1,source)==1&&flags==1,"BF16 snapshot header");
    require(h[0]>0&&h[0]<=2048&&h[1]<=8192&&h[0]+h[1]<=8192,"bounded query/past header");
    const int q=h[0],past=h[1],heads=h[2],dim=h[3],rope=h[4],ratio=h[5],budget=h[6];
    require(q>0&&q<=2048&&past>=0&&past+q<=8192&&heads>0&&heads<=16&&dim>1&&dim<=256&&
        rope>=2&&rope<=dim&&rope%2==0&&ratio>0&&ratio<=16&&budget>0&&budget<=4096-ratio+1&&
        budget%ratio==0&&h[7]<=262144&&std::isfinite(eps)&&eps>0,"bounded snapshot geometry");
    const int keys=past+q,groups=keys/ratio,width=(heads+1)*dim,slots=budget+ratio-1;
    require(keys>budget+ratio,"sparse replay history");
    require(h[8]>=uint32_t(std::max(1,groups))&&h[9]==uint32_t(slots),"snapshot strides/slots");
    const size_t elements=size_t(q)*(width+heads*dim+groups+slots)+size_t(keys+groups)*dim+2*dim+rope/2;
    require(elements<=size_t(64)*1024*1024/4,"bounded payload");
    std::vector<float> payload(elements);
    require(std::fread(payload.data(),4,elements,source)==elements&&std::fgetc(source)==EOF,"complete snapshot payload");
    std::fclose(source);
    size_t cursor=0;
    auto part=[&](size_t count){float* p=payload.data()+cursor;cursor+=count;return p;};
    float* projection=part(size_t(q)*width),*qw=part(dim),*kw=part(dim),*freq=part(rope/2);
    float* raw=part(size_t(keys)*dim),*query=part(size_t(q)*heads*dim),*pooled=part(size_t(groups)*dim);
    float* scores=part(size_t(q)*groups);
    auto* ids=reinterpret_cast<int32_t*>(payload.data()+cursor);
    std::vector<float> full_projection(size_t(keys)*width,0.f);
    for(int row=0;row<keys;++row)std::copy_n(raw+size_t(row)*dim,dim,full_projection.data()+size_t(row)*width+heads*dim);
    std::copy_n(projection,size_t(q)*width,full_projection.data()+size_t(past)*width);
    Buffer dp(full_projection.size()*4),dwq(dim*4),dwk(dim*4),df(rope/2*4),dr(size_t(keys)*dim*4);
    Buffer dq(size_t(keys)*heads*dim*4),dk(size_t(groups)*dim*4),ds(size_t(keys)*groups*4),di(size_t(keys)*slots*4);
    check(cudaMemcpy(dp.data,full_projection.data(),full_projection.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dwq.data,qw,dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dwk.data,kw,dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(df.data,freq,rope/2*4,cudaMemcpyHostToDevice));
    mollm_cuda::QsaIndexerParams p{};p.heads=heads;p.dimension=dim;p.rope_dim=rope;p.ratio=ratio;
    p.queries=p.capacity=keys;p.position_offset=h[7];p.eps=eps;p.projection_stride=width;p.score_stride=groups;p.diagnostic_bf16=true;
    check(mollm_cuda::launch_qsa_indexer_update(dp.f(),dwq.f(),dwk.f(),df.f(),dr.f(),dk.f(),dq.f(),ds.f(),p));
    check(mollm_cuda::launch_qsa_causal_block_topk(ds.f(),static_cast<int32_t*>(di.data),keys,keys,0,budget,ratio,groups,slots));
    check(cudaMemcpy(raw,dr.data,size_t(keys)*dim*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(query,dq.f(size_t(past)*heads*dim),size_t(q)*heads*dim*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(pooled,dk.data,size_t(groups)*dim*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(scores,ds.f(size_t(past)*groups),size_t(q)*groups*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ids,static_cast<int32_t*>(di.data)+size_t(past)*slots,size_t(q)*slots*4,cudaMemcpyDeviceToHost));
    FILE* destination=std::fopen(output,"wbx");require(destination,"exclusively create replay output");
    require(std::fwrite(magic,1,8,destination)==8&&std::fwrite(h,4,10,destination)==10&&
        std::fwrite(&eps,4,1,destination)==1&&std::fwrite(&flags,4,1,destination)==1&&
        std::fwrite(payload.data(),4,payload.size(),destination)==payload.size(),"write replay output");
    require(std::fclose(destination)==0,"close replay output");
    std::puts("QSA snapshot replay: rebuilt all pooled history and current rows");
}

std::vector<float> normalized_rotated(const float* input,const std::vector<float>& weight,
    const std::vector<float>& freq,int rope,int position) {
    const int dim=weight.size();
    double mean=0;
    for(int d=0;d<dim;++d)mean+=double(input[d])*input[d];
    const double inv=1/std::sqrt(mean/dim+1e-6);
    std::vector<float> result(dim),normalized(dim);
    for(int d=0;d<dim;++d)result[d]=normalized[d]=float(input[d]*inv*(1.+weight[d]));
    for(int d=0;d<rope;++d) {
        const int half=rope/2;
        // float position/frequency multiplication matches the supplied RoPE
        // representation; normalization and dot accumulation are independent.
        const float angle=float(position)*freq[d%half];
        const double other=d<half?-normalized[d+half]:normalized[d-half];
        result[d]=float(normalized[d]*std::cos(double(angle))+other*std::sin(double(angle)));
    }
    return result;
}

void fixture(int n,int dim,int heads,int rope,int budget) {
    const int ratio=4,offset=37,capacity=n+7,width=(heads+1)*dim;
    const size_t ps=width+3,ss=capacity/ratio+3,slots=budget+ratio-1,is=slots+3;
    std::vector<float> projection(n*ps),qw(dim),kw(dim),freq(rope/2);
    unsigned random=189271;
    for(float& x:projection){random^=random<<13;random^=random>>17;random^=random<<5;x=(int(random%20001)-10000)*.00013f;}
    for(int d=0;d<dim;++d){qw[d]=.07f*std::sin(float(d));kw[d]=.09f*std::cos(float(d));}
    for(int d=0;d<rope/2;++d)freq[d]=1/std::pow(10000000.f,2.f*d/rope);
    Buffer dp(projection.size()*4),dq((size_t(n)*heads*dim+16)*4),dr((size_t(capacity)*dim+16)*4);
    Buffer db((size_t(capacity/ratio)*dim+16)*4),ds((n*ss+16)*4),di((n*is+16)*4);
    Buffer dwq(dim*4),dwk(dim*4),df(freq.size()*4);
    check(cudaMemcpy(dwq.data,qw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dwk.data,kw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(df.data,freq.data(),freq.size()*4,cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    std::vector<float> whole_q,whole_scores,whole_raw,whole_pooled;
    std::vector<int32_t> whole_ids;
    for(int variant=0;variant<2;++variant) {
        if(variant)for(float& x:projection)x=x*-.7f+.13f;
        check(cudaMemcpy(dp.data,projection.data(),projection.size()*4,cudaMemcpyHostToDevice));
        // Independent scalar reference recomputes complete groups, never
        // treats partial groups as permanent pooled keys.
        std::vector<float> expected_q(size_t(n)*heads*dim),expected_pooled(size_t(n/ratio)*dim);
        for(int r=0;r<n;++r)for(int h=0;h<heads;++h) {
            auto q=normalized_rotated(projection.data()+r*ps+h*dim,qw,freq,rope,offset+r);
            std::copy(q.begin(),q.end(),expected_q.begin()+(size_t(r)*heads+h)*dim);
        }
        for(int b=0;b<n/ratio;++b) {
            std::vector<float> pooled(dim);
            for(int d=0;d<dim;++d) {
                double sum=0;
                for(int i=0;i<ratio;++i)sum+=projection[(b*ratio+i)*ps+heads*dim+d];
                pooled[d]=float(sum/ratio);
            }
            auto k=normalized_rotated(pooled.data(),kw,freq,rope,offset+b*ratio);
            std::copy(k.begin(),k.end(),expected_pooled.begin()+size_t(b)*dim);
        }
        const std::vector<std::vector<int>> schedules=n<100?
            std::vector<std::vector<int>>{{n},{3,2,1,5,7},{1}}:
            std::vector<std::vector<int>>{{n},{3,511,1,2,1024},{31,32,33}};
        for(size_t schedule=0;schedule<schedules.size();++schedule) {
            const float guard=12345.f;
            auto poison=[&](Buffer& b,size_t count){std::vector<float> v(count+16,guard);check(cudaMemcpy(b.data,v.data(),v.size()*4,cudaMemcpyHostToDevice));};
            poison(dq,size_t(n)*heads*dim);poison(dr,size_t(capacity)*dim);
            poison(db,size_t(capacity/ratio)*dim);poison(ds,n*ss);
            std::vector<int32_t> initial(n*is+16,12345);
            check(cudaMemcpy(di.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));
            int past=0,step=0;
            std::vector<int> score_lengths(n);
            while(past<n) {
                const auto& sequence=schedules[schedule];
                const int rows=std::min(n-past,sequence[std::min(size_t(step),sequence.size()-1)]);
                mollm_cuda::QsaIndexerParams p{};
                p.heads=heads;p.dimension=dim;p.rope_dim=rope;p.ratio=ratio;
                p.queries=rows;p.past=past;p.capacity=capacity;p.position_offset=offset;
                p.projection_stride=ps;p.score_stride=ss;
                for(int r=past;r<past+rows;++r)score_lengths[r]=(past+rows)/ratio;
                const auto launch=[&] {
                    check(mollm_cuda::launch_qsa_indexer_update(dp.f(past*ps),dwq.f(),dwk.f(),df.f(),
                        dr.f(8),db.f(8),dq.f(8+size_t(past)*heads*dim),ds.f(8+past*ss),p,stream));
                    check(mollm_cuda::launch_qsa_causal_block_topk(ds.f(8+past*ss),
                        static_cast<int32_t*>(di.data)+8+past*is,rows,past+rows,past,budget,ratio,ss,is,stream));
                };
                if(past==0) {
                    auto bad=p;bad.queries=capacity+1;
                    require(mollm_cuda::launch_qsa_indexer_update(dp.f(),dwq.f(),dwk.f(),df.f(),
                        dr.f(8),db.f(8),dq.f(8),ds.f(8),bad,stream)==cudaErrorInvalidValue,"reject cache overflow before mutation");
                    bad=p;bad.rope_dim=dim+2;
                    require(mollm_cuda::launch_qsa_indexer_update(dp.f(),dwq.f(),dwk.f(),df.f(),
                        dr.f(8),db.f(8),dq.f(8),ds.f(8),bad,stream)==cudaErrorInvalidValue,"reject wrong main RoPE width");
                }
                if(schedule==1) {
                    cudaGraph_t graph;cudaGraphExec_t exec;
                    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
                    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                    check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
                    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
                } else launch();
                past+=rows;++step;
            }
            check(cudaStreamSynchronize(stream));
            auto read=[&](Buffer& b,size_t count){std::vector<float> v(count+16);check(cudaMemcpy(v.data(),b.data,v.size()*4,cudaMemcpyDeviceToHost));
                for(int i=0;i<8;++i)require(v[i]==guard&&v[count+8+i]==guard,"output guards");
                return std::vector<float>(v.begin()+8,v.end()-8);};
            auto query=read(dq,size_t(n)*heads*dim),raw=read(dr,size_t(capacity)*dim);
            auto pooled=read(db,size_t(capacity/ratio)*dim),scores=read(ds,n*ss);
            std::vector<int32_t> indices(n*is+16);check(cudaMemcpy(indices.data(),di.data,indices.size()*4,cudaMemcpyDeviceToHost));
            for(int i=0;i<8;++i)require(indices[i]==12345&&indices[n*is+8+i]==12345,"index guards");
            for(size_t i=0;i<query.size();++i)require(std::fabs(query[i]-expected_q[i])<4e-6,"query norm/RoPE");
            for(size_t i=0;i<expected_pooled.size();++i)require(std::fabs(pooled[i]-expected_pooled[i])<4e-6,"pooled norm/RoPE");
            for(int r=0;r<n;++r)for(int d=0;d<dim;++d)require(raw[size_t(r)*dim+d]==projection[r*ps+heads*dim+d],"cache raw K exact");
            for(size_t i=size_t(n)*dim;i<raw.size();++i)require(raw[i]==guard,"future raw cache untouched");
            for(size_t i=expected_pooled.size();i<pooled.size();++i)require(pooled[i]==guard,"partial/future pooled groups untouched");
            for(int r=0;r<n;++r) {
                const int complete=(r+1)/ratio,top=std::min(budget/ratio,complete);
                std::vector<double> reference(complete);
                for(int b=complete;b<score_lengths[r];++b)
                    require(scores[r*ss+b]==-std::numeric_limits<float>::infinity(),"future group scores inactive");
                for(size_t b=score_lengths[r];b<ss;++b)
                    require(scores[r*ss+b]==guard,"unallocated score columns untouched");
                for(int b=0;b<complete;++b) {
                    double score=0;
                    for(int h=0;h<heads;++h) {
                        double dot=0;
                        for(int d=0;d<dim;++d)dot+=double(expected_q[(size_t(r)*heads+h)*dim+d])*expected_pooled[size_t(b)*dim+d];
                        score+=std::max(0.,dot);
                    }
                    reference[b]=score/std::sqrt(double(dim));
                    require(std::fabs(scores[r*ss+b]-reference[b])<4e-5,"block scores versus scalar reference");
                }
                std::vector<int> chosen;
                const int count=top*ratio+(r+1)%ratio;
                for(int s=0;s<int(slots);++s) {
                    int id=indices[8+r*is+s];
                    if(s>=count){require(id==-1,"unused selection padding");continue;}
                    require(id>=0&&id<=r,"selection visible");
                    if(s)require(id>indices[8+r*is+s-1],"selection unique chronological");
                    if(id<complete*ratio&&id%ratio==0)chosen.push_back(id/ratio);
                }
                require(int(chosen.size())==top,"whole block count");
                for(int b:chosen)for(int d=0;d<ratio;++d)
                    require(std::find(indices.begin()+8+r*is,indices.begin()+8+r*is+count,b*ratio+d)!=indices.begin()+8+r*is+count,"whole groups retained");
                for(int id=complete*ratio;id<=r;++id)
                    require(indices[8+r*is+count-(r+1)%ratio+id-complete*ratio]==id,"tail retained");
                if(top&&complete>top) {
                    double lowest=std::numeric_limits<double>::infinity();
                    for(int b:chosen)lowest=std::min(lowest,reference[b]);
                    for(int b=0;b<complete;++b)if(std::find(chosen.begin(),chosen.end(),b)==chosen.end())
                        require(reference[b]<=lowest+8e-5,"chosen scores satisfy independent cutoff (ties allowed)");
                }
                for(size_t s=slots;s<is;++s)require(indices[8+r*is+s]==12345,"index row padding");
            }
            if(schedule==0){whole_q=query;whole_scores=scores;whole_raw=raw;whole_pooled=pooled;whole_ids=indices;}
            else {
                require(query==whole_q&&raw==whole_raw&&pooled==whole_pooled&&indices==whole_ids,"whole/chunk/decode/reset exact");
                for(int r=0;r<n;++r)for(int b=0;b<(r+1)/ratio;++b)
                    require(scores[r*ss+b]==whole_scores[r*ss+b],"chunk scores exact");
            }
        }
    }
    check(cudaStreamDestroy(stream));
}

void long_history(int keys=32768) {
    constexpr int rows=3,dim=128,heads=4,rope=64,slots=2051;
    const int past=keys-rows,groups=keys/4;
    std::vector<float> raw(size_t(keys)*dim),pooled(size_t(groups)*dim,12345.f);
    std::vector<float> projection(rows*(heads+1)*dim),qw(dim),kw(dim),freq(rope/2);
    unsigned random=71937;
    const auto next=[&]() {random^=random<<13;random^=random>>17;random^=random<<5;return (int(random%20001)-10000)*.00013f;};
    for(float& x:raw)x=next();for(float& x:projection)x=next();
    for(int d=0;d<dim;++d){qw[d]=.07f*std::sin(float(d));kw[d]=.09f*std::cos(float(d));}
    for(int d=0;d<rope/2;++d)freq[d]=1/std::pow(10000000.f,2.f*d/rope);
    const auto pool=[&](int b) {
        std::vector<float> mean(dim);
        for(int d=0;d<dim;++d) {
            double sum=0;for(int i=0;i<4;++i)sum+=raw[size_t(b*4+i)*dim+d];
            mean[d]=float(sum/4);
        }
        return normalized_rotated(mean.data(),kw,freq,rope,37+b*4);
    };
    // A completed-prefix cache is supplied independently; only its last
    // partial group may be finalized by this three-token continuation.
    for(int b=0;b<past/4;++b) {
        auto k=pool(b);std::copy(k.begin(),k.end(),pooled.begin()+size_t(b)*dim);
    }
    Buffer dr(raw.size()*4),db(pooled.size()*4),dp(projection.size()*4),dq(rows*heads*dim*4);
    Buffer ds(rows*(groups+3)*4),di(rows*(slots+3)*4),dwq(dim*4),dwk(dim*4),df(freq.size()*4);
    Buffer candidates(rows*1024*4),candidate_ids(rows*1024*4);
    check(cudaMemcpy(dr.data,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,pooled.data(),pooled.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dp.data,projection.data(),projection.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dwq.data,qw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dwk.data,kw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(df.data,freq.data(),freq.size()*4,cudaMemcpyHostToDevice));
    mollm_cuda::QsaIndexerParams p{};
    p.queries=rows;p.past=past;p.capacity=keys;p.position_offset=37;
    p.projection_stride=(heads+1)*dim;p.score_stride=groups+3;
    check(mollm_cuda::launch_qsa_indexer_update(dp.f(),dwq.f(),dwk.f(),df.f(),dr.f(),db.f(),dq.f(),ds.f(),p));
    check(mollm_cuda::launch_qsa_causal_block_topk(ds.f(),static_cast<int32_t*>(di.data),
        rows,keys,past,2048,4,groups+3,slots+3,nullptr,candidates.f(),static_cast<int32_t*>(candidate_ids.data)));
    check(cudaDeviceSynchronize());
    std::vector<float> scores(rows*(groups+3)),actual_pooled(pooled.size());
    std::vector<int32_t> indices(rows*(slots+3));
    check(cudaMemcpy(scores.data(),ds.data,scores.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(indices.data(),di.data,indices.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual_pooled.data(),db.data,actual_pooled.size()*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<size_t(past/4)*dim;++i)require(actual_pooled[i]==pooled[i],"32K old pooled cache unchanged");
    for(int r=0;r<rows;++r)for(int d=0;d<dim;++d)
        raw[size_t(past+r)*dim+d]=projection[(r*(heads+1)+heads)*dim+d];
    auto last=pool(groups-1);std::copy(last.begin(),last.end(),pooled.end()-dim);
    for(int d=0;d<dim;++d)require(std::fabs(last[d]-actual_pooled[(groups-1)*dim+d])<4e-6,"32K cross-chunk pooled key");
    for(int r=0;r<rows;++r) {
        std::vector<float> query(heads*dim);
        for(int h=0;h<heads;++h) {
            auto q=normalized_rotated(projection.data()+(r*(heads+1)+h)*dim,qw,freq,rope,37+past+r);
            std::copy(q.begin(),q.end(),query.begin()+h*dim);
        }
        const int complete=(past+r+1)/4,tail=(past+r+1)%4;
        std::vector<double> reference(complete);
        for(int b=0;b<complete;++b) {
            double score=0;
            for(int h=0;h<heads;++h) {
                double dot=0;for(int d=0;d<dim;++d)dot+=double(query[h*dim+d])*pooled[size_t(b)*dim+d];
                score+=std::max(0.,dot);
            }
            reference[b]=score/std::sqrt(double(dim));
            require(std::fabs(scores[r*(groups+3)+b]-reference[b])<4e-5,"32K scalar scores");
        }
        std::vector<int> chosen;
        for(int s=0;s<2048;++s) {
            const int id=indices[r*(slots+3)+s];
            require(id>=0&&id<complete*4,"32K selected full-group bounds");
            if(s)require(id>indices[r*(slots+3)+s-1],"32K unique IDs");
            if(s%4==0){require(id%4==0,"32K group begins at first token");chosen.push_back(id/4);}
            else require(id==indices[r*(slots+3)+s-1]+1,"32K all group tokens");
        }
        double cutoff=std::numeric_limits<double>::infinity();
        for(int b:chosen)cutoff=std::min(cutoff,reference[b]);
        for(int b=0;b<complete;++b)if(std::find(chosen.begin(),chosen.end(),b)==chosen.end())
            require(reference[b]<=cutoff+8e-5,"32K independent cutoff");
        for(int i=0;i<tail;++i)require(indices[r*(slots+3)+2048+i]==complete*4+i,"32K tail retained");
        for(int i=2048+tail;i<slots;++i)require(indices[r*(slots+3)+i]==-1,"32K padding");
    }
}

void projected_pipeline() {
    constexpr int hidden=2560,n=9,heads=4,dim=128,width=(heads+1)*dim,rope=64,slots=7;
    std::vector<__half> weights(size_t(width)*hidden,__float2half(0.f)),input(n*hidden);
    std::vector<float> projected(n*width),qw(dim,0.f),kw(dim,0.f),frequency(rope/2);
    unsigned random=91735;
    for(auto& x:input){random^=random<<13;random^=random>>17;random^=random<<5;x=__float2half((int(random%20001)-10000)*.0001f);}
    // Full production projection geometry, sparse dyadic fixture weights.
    // Two nonzero terms are exactly representable in FP32, so projection
    // layout/precision can be checked independently without a GEMM tolerance.
    for(int d=0;d<width;++d) {
        const int a=(d*3)%hidden,b=(d*7+13)%hidden;
        weights[size_t(d)*hidden+a]=__float2half(.75f);
        weights[size_t(d)*hidden+b]=__float2half(.125f);
        for(int r=0;r<n;++r)projected[r*width+d]=__half2float(input[r*hidden+a])*.75f+__half2float(input[r*hidden+b])*.125f;
    }
    for(int i=0;i<rope/2;++i)frequency[i]=1/std::pow(10000000.f,2.f*i/rope);
    Buffer dw(weights.size()*2),dx(input.size()*2),dp(projected.size()*4),dr(n*dim*4),db((n/4)*dim*4);
    Buffer dq(n*heads*dim*4),ds(n*3*4),di(n*slots*4),out(n*heads*dim*4),wq(dim*4),wk(dim*4),df(frequency.size()*4);
    check(cudaMemcpy(dw.data,weights.data(),weights.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx.data,input.data(),input.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(wq.data,qw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(wk.data,kw.data(),dim*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(df.data,frequency.data(),frequency.size()*4,cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    cublasHandle_t handle;require(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS,"create dense projection handle");
    require(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"projection stream");
    require(mollm_cuda::run_dense_matmul(handle,dw.data,CUDA_R_16F,dx.data,CUDA_R_16F,dp.f(),n,width,hidden,hidden,true),"native index QK projection");
    mollm_cuda::QsaIndexerParams ip{};
    ip.queries=n;ip.capacity=n;ip.projection_stride=width;ip.score_stride=3;
    check(mollm_cuda::launch_qsa_indexer_update(dp.f(),wq.f(),wk.f(),df.f(),dr.f(),db.f(),dq.f(),ds.f(),ip,stream));
    check(mollm_cuda::launch_qsa_causal_block_topk(ds.f(),static_cast<int32_t*>(di.data),n,n,0,4,4,3,slots,stream));
    mollm_cuda::QsaSelectedAttentionParams ap{};
    ap.heads=heads;ap.kv_heads=1;ap.queries=n;ap.keys=n;ap.key_dim=dim;ap.value_dim=dim;ap.slots=slots;
    ap.scale=1/std::sqrt(float(dim));ap.query_position_stride=heads*dim;ap.query_head_stride=dim;
    ap.key_position_stride=dim;ap.key_head_stride=n*dim;ap.value_position_stride=dim;ap.value_head_stride=n*dim;
    ap.output_position_stride=heads*dim;ap.output_head_stride=dim;ap.index_position_stride=slots;
    // Independent synthetic MAIN attention uses raw index K for K and V;
    // tying these fixture inputs does not model actual checkpoint projections.
    check(mollm_cuda::launch_qsa_selected_attention(dq.f(),dr.f(),dr.f(),false,
        static_cast<int32_t*>(di.data),out.f(),ap,stream));
    check(cudaStreamSynchronize(stream));
    std::vector<float> actual_projected(projected.size()),actual(n*heads*dim);
    std::vector<int32_t> ids(n*slots);
    check(cudaMemcpy(actual_projected.data(),dp.data,projected.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(),out.data,actual.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ids.data(),di.data,ids.size()*4,cudaMemcpyDeviceToHost));
    require(actual_projected==projected,"full geometry projection exact scalar fixture");
    std::vector<float> raw(n*dim),queries(n*heads*dim),pooled((n/4)*dim);
    for(int r=0;r<n;++r) {
        std::copy_n(projected.begin()+r*width+heads*dim,dim,raw.begin()+r*dim);
        for(int h=0;h<heads;++h) {
            auto q=normalized_rotated(projected.data()+r*width+h*dim,qw,frequency,rope,r);
            std::copy(q.begin(),q.end(),queries.begin()+(r*heads+h)*dim);
        }
    }
    for(int b=0;b<n/4;++b) {
        std::vector<float> mean(dim);
        for(int d=0;d<dim;++d){double sum=0;for(int t=0;t<4;++t)sum+=raw[(b*4+t)*dim+d];mean[d]=float(sum/4);}
        auto k=normalized_rotated(mean.data(),kw,frequency,rope,b*4);
        std::copy(k.begin(),k.end(),pooled.begin()+b*dim);
    }
    for(int r=0;r<n;++r) {
        int best=-1;double best_score=-1;
        for(int b=0;b<(r+1)/4;++b) {
            double score=0;
            for(int h=0;h<heads;++h){double dot=0;for(int d=0;d<dim;++d)dot+=double(queries[(r*heads+h)*dim+d])*pooled[b*dim+d];score+=std::max(0.,dot);}
            if(score>best_score){best=b;best_score=score;}
        }
        std::vector<int> selected;
        if(best>=0)for(int t=0;t<4;++t)selected.push_back(best*4+t);
        for(int t=(r+1)/4*4;t<=r;++t)selected.push_back(t);
        for(int s=0;s<slots;++s)require(ids[r*slots+s]==(s<int(selected.size())?selected[s]:-1),"pipeline independent selected IDs");
        for(int h=0;h<heads;++h) {
            std::vector<double> logits;double maximum=-std::numeric_limits<double>::infinity(),total=0;
            for(int t:selected){double dot=0;for(int d=0;d<dim;++d)dot+=double(queries[(r*heads+h)*dim+d])*raw[t*dim+d];logits.push_back(dot*ap.scale);maximum=std::max(maximum,logits.back());}
            for(auto& x:logits){x=std::exp(x-maximum);total+=x;}
            for(int d=0;d<dim;++d){double expected=0;for(size_t s=0;s<selected.size();++s)expected+=logits[s]/total*raw[selected[s]*dim+d];
                require(std::fabs(actual[(r*heads+h)*dim+d]-expected)<4e-6,"native projection-indexer-topk-attention versus scalar");}
        }
    }
    require(cublasDestroy(handle)==CUBLAS_STATUS_SUCCESS,"destroy projection handle");check(cudaStreamDestroy(stream));
}
}
int main(int argc,char** argv) {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    if(argc==4&&!std::strcmp(argv[1],"--snapshot-replay")) {
        replay_snapshot(argv[2],argv[3]);return 0;
    }
    if(argc==2&&std::strcmp(argv[1],"--long-only")==0){long_history();return 0;}
    if(argc==2&&std::strcmp(argv[1],"--extended-history-only")==0) {
        for(int keys:{65536,131072,262144})long_history(keys);
        std::puts("QSA extended indexer:64K/128K/262K continuation, prefix reuse, pooling, scores and cutoff pass");return 0;
    }
    fixture(73,8,2,4,8);
    fixture(129,128,4,64,2048);
    if(argc==1){fixture(2055,128,4,64,2048);long_history();projected_pipeline();}
    else require(argc==2&&std::strcmp(argv[1],"--small-only")==0,"unknown arguments");
    std::puts("QSA FP32 indexer update/pool/score/topk: scalar, partial groups, chunk/decode/reset and capture pass");
}
