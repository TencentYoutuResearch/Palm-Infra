#include "backends/cuda/backend.h"
#include "graph/execute.h"
#include "core/cache_layout.h"
#include "core/fp16.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>

namespace {
void require(bool x,const char* s){if(!x){std::fprintf(stderr,"QSA graph: %s\n",s);std::exit(1);}}
int dim=8,vdim=5,heads=2,capacity=64,slots=11;
struct Frame {
    CudaBackend backend;
    Graph graph;
    BufferPool pool;
    ExecContext ctx;
    bool half;
    std::vector<float> keys,values;
    Frame(bool half_cache,int liveness):half(half_cache) {
        require(backend.available(),"backend available");
        backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
        graph.nodes.resize(14);graph.runtime.tensors.resize(14);
        for(int i=0;i<14;++i) {
            auto& node=graph.nodes[i];node.id=i;node.op_type=i<12?OpType::INPUT:OpType::QSA_INDEXER;
            node.out_prec=Precision::FP32;
            std::fill_n(node.out_shape,4,1);
            if(i<12)graph.graph_inputs.push_back(i);
        }
        graph.nodes[12].op_type=OpType::QSA_INDEXER;graph.nodes[12].out_prec=Precision::INT32;
        graph.nodes[12].inputs={0,1,2,3,4,5,6,7};graph.nodes[12].params.i32={heads,dim,4,4,slots-3};
        graph.nodes[12].params.f32={1e-6f};graph.nodes[12].out_shape[0]=slots;
        graph.nodes[13].op_type=OpType::QSA_SDPA;graph.nodes[13].inputs={8,9,10,7,6,11,12};
        graph.nodes[13].params.i32={2,1,heads,1,dim,vdim};graph.nodes[13].params.f32={1/std::sqrt(float(dim))};
        graph.nodes[13].out_shape[0]=vdim;graph.nodes[13].out_shape[2]=heads;
        for(int i:{12,13})graph.nodes[i].dim_expr[1].kind=DIM_SEQ;
        graph.graph_outputs={12,13};
        auto& ts=graph.runtime.tensors;
        ts[1]=allocate(dim,1);ts[2]=allocate(dim,1);ts[3]=allocate(2,1);
        write(ts[1],std::vector<float>(dim,0.f));write(ts[2],std::vector<float>(dim,0.f));
        write(ts[3],{1.f,1/std::sqrt(10000000.f)});
        ts[4]=allocate(dim,capacity);ts[5]=allocate(dim,capacity/4);
        require(backend.zero_tensor(ts[4],ts[4].nbytes())&&backend.zero_tensor(ts[5],ts[5].nbytes()),"zero raw/pooled cache");
        for(int i:{6,11}) {
            const int d=i==6?dim:vdim;
            ts[i]=Tensor::create(half?Precision::FP16:Precision::FP32,MemoryType::EXTERNAL,d,capacity);
            backend.alloc_persistent(ts[i],CacheMetadata::SIZE+size_t(d)*capacity*(half?2:4),
                PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,CacheMetadata::SIZE);
            require(ts[i].data!=nullptr,"KV persistent allocation");
            CacheMetadata meta;meta.max_seq_len=capacity;meta.num_kv_heads=1;
            meta.head_dim=dim;meta.v_head_dim=vdim;
            require(backend.copy_from_host(&meta,ts[i],sizeof(meta)),"initialize cache metadata");
        }
        ctx.graph=&graph;ctx.pool=&pool;ctx.backend=&backend;
        ctx.conservative_stateful_liveness=liveness>0;ctx.stateful_readonly_last_use=liveness==2;
        prepare_execution(ctx);
    }
    Tensor allocate(int d,int rows,int h=1) {
        auto t=Tensor::create(Precision::FP32,MemoryType::NONE,d,rows,h);
        require(backend.alloc_output(t,t.nbytes(),nullptr)!=nullptr,"allocate tensor");return t;
    }
    void write(Tensor& t,const std::vector<float>& x) {
        require(t.nbytes()==x.size()*4&&backend.copy_from_host(x.data(),t,t.nbytes()),"upload input");
    }
    std::vector<float> read(const Tensor& t) {
        std::vector<float> x(t.nelements());require(backend.copy_to_host(t,x.data(),x.size()*4),"read tensor");return x;
    }
    void bind(int id,int d,int rows,int h,const std::vector<float>& x) {
        auto& t=graph.runtime.tensors[id];if(t.data)backend.free_output(t,nullptr);
        t=allocate(d,rows,h);write(t,x);
    }
    void pass(int past,int rows,int seed,int bad_mask,std::vector<float>& output,std::vector<int32_t>& selections) {
        auto& ts=graph.runtime.tensors;
        for(int i:{6,11}) {
            uint64_t length=past;require(backend.copy_from_host(&length,ts[i],sizeof(length),offsetof(CacheMetadata,current_seq_len)),"set logical cache position");
        }
        std::vector<float> projection(rows*(heads+1)*dim),q(rows*heads*dim),k(rows*dim),v(rows*vdim);
        const int total=past+rows;
        keys.resize(total*dim);values.resize(total*vdim);
        for(int r=0;r<rows;++r) {
            for(int d=0;d<(heads+1)*dim;++d)projection[r*(heads+1)*dim+d]=std::sin(float((past+r)*31+d*13+seed)*.13f);
            for(int h=0;h<heads;++h)for(int d=0;d<dim;++d)q[(h*rows+r)*dim+d]=std::cos(float((past+r)*7+h*19+d+seed)*.09f)*.4f;
            for(int d=0;d<dim;++d){float x=std::sin(float((past+r)*11+d+seed)*.08f)*.3f;k[r*dim+d]=x;keys[(past+r)*dim+d]=half?float(mollm::fp16_t(x)):x;}
            for(int d=0;d<vdim;++d){float x=std::cos(float((past+r)*9+d+seed)*.07f)*.2f;v[r*vdim+d]=x;values[(past+r)*vdim+d]=half?float(mollm::fp16_t(x)):x;}
        }
        std::vector<float> mask(rows*total,-1e38f);
        for(int r=0;r<rows;++r)std::fill_n(mask.begin()+r*total,past+r+1,0.f);
        if(bad_mask==1)mask[0]=-1e38f;
        if(bad_mask==2)mask[total-1]=0.f;
        if(bad_mask==3)mask[total-1]=std::numeric_limits<float>::quiet_NaN();
        bind(0,(heads+1)*dim,rows,1,projection);bind(7,total,rows,1,mask);
        bind(8,dim,rows,heads,q);bind(9,dim,rows,1,k);bind(10,vdim,rows,1,v);
        ctx.runtime_seq_len=rows;inject_runtime_shapes(ctx);
        auto raw_before=read(ts[4]);auto pooled_before=read(ts[5]);
        backend.clear_dispatch_error();backend.begin_graph();execute_graph(ctx);backend.end_graph();
        if(bad_mask) {
            require(ctx.execution_failed&&backend.dispatch_failed(),"unsupported mask fails explicitly");
            require(read(ts[4])==raw_before&&read(ts[5])==pooled_before,"failed mask does not mutate QSA state");return;
        }
        require(!ctx.execution_failed&&!backend.dispatch_failed(),"native graph execution");
        auto actual=read(ts[13]);std::vector<int32_t> ids(ts[12].nelements());
        require(backend.copy_to_host(ts[12],ids.data(),ids.size()*4),"download indexer IDs");
        for(int r=0;r<rows;++r) {
            std::vector<int> selected;
            for(int s=0;s<slots;++s){int id=ids[r*slots+s];if(id>=0){require(id<=past+r,"IDs causal");selected.push_back(id);}}
            require(!selected.empty(),"nonempty selection");
            for(int h=0;h<heads;++h) {
                std::vector<double> logits;double maximum=-1e300,sum=0;
                for(int id:selected){double dot=0;for(int d=0;d<dim;++d)dot+=double(q[(h*rows+r)*dim+d])*keys[id*dim+d];logits.push_back(dot/std::sqrt(double(dim)));maximum=std::max(maximum,logits.back());}
                for(auto& x:logits){x=std::exp(x-maximum);sum+=x;}
                for(int d=0;d<vdim;++d){double expected=0;for(size_t s=0;s<selected.size();++s)expected+=logits[s]/sum*values[selected[s]*vdim+d];
                    const float got=actual[(h*rows+r)*vdim+d];require(std::isfinite(got)&&std::fabs(got-expected)<3e-6,"cached sparse output versus scalar");output.push_back(got);}
            }
            selections.insert(selections.end(),ids.begin()+r*slots,ids.begin()+(r+1)*slots);
        }
        auto raw=read(ts[4]);for(int r=0;r<rows;++r)for(int d=0;d<dim;++d)
            require(raw[(past+r)*dim+d]==projection[r*(heads+1)*dim+heads*dim+d],"raw K graph state");
    }
    ~Frame(){backend.release_graph_plan(graph);for(int i:{0,1,2,3,4,5,7,8,9,10,12,13})if(graph.runtime.tensors[i].data)backend.free_output(graph.runtime.tensors[i],nullptr);}
};
void run(bool half,int liveness) {
    Frame whole(half,liveness),chunk(half,liveness);
    std::vector<float> a,b;std::vector<int32_t> ia,ib;
    whole.pass(0,29,17,0,a,ia);
    int past=0;for(int rows:{3,2,1,5,7,11}){chunk.pass(past,rows,17,0,b,ib);past+=rows;}
    require(a==b&&ia==ib,"whole versus chunk graph exact");
    // Reset logical history AND auxiliary state, like the engine reset path.
    require(chunk.backend.zero_tensor(chunk.graph.runtime.tensors[4],capacity*dim*4)&&
            chunk.backend.zero_tensor(chunk.graph.runtime.tensors[5],capacity/4*dim*4),"reset auxiliary state");
    std::vector<float> reset,reference;std::vector<int32_t> ir,ix;
    chunk.pass(0,13,71,0,reset,ir);
    Frame fresh(half,liveness);fresh.pass(0,13,71,0,reference,ix);
    require(reset==reference&&ir==ix,"reset graph fresh exact");
}
}
int main() {
    setenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS","1",1);
    setenv("MOLLM_CUDA_QSA_GQA_REUSE","1",1);
    setenv("MOLLM_CUDA_QSA_DECODE_SPLIT","1",1);
    CudaBackend probe;if(!probe.available())return 77;
    for(bool half:{false,true})for(int mode:{0,1,2})run(half,mode);
    for(int bad:{1,2,3}){Frame f(true,2);std::vector<float> out;std::vector<int32_t> ids;f.pass(0,13,7,bad,out,ids);}
    // Exercise both backend selectors: grouped whole prefill, then grouped
    // chunk, legacy small tail and split QK decode. Optimized paths preserve
    // the legacy normalization/PV order, so outputs must be exact.
    dim=vdim=256;heads=12;capacity=128;slots=131;
    for(bool half:{false,true}) {
        Frame whole(half,2),chunk(half,2);
        std::vector<float> a,b;std::vector<int32_t> ia,ib;
        whole.pass(0,64,17,0,a,ia);
        chunk.pass(0,32,17,0,b,ib);chunk.pass(32,31,17,0,b,ib);chunk.pass(63,1,17,0,b,ib);
        require(ia==ib&&a==b,"optimized backend exact selection/cache/output equivalence");
    }
    std::puts("QSA graph: native FP16/FP32 KV, liveness, chunk/reset and mask rejection pass");
}
