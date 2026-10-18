#include "backends/cpu/backend.h"
#include "graph/execute.h"
#include <cstdio>
#include <cstring>

// Test only the diagnostic's special ShortConv storage indexing, not its math.
class DimMajorProbe : public CPUBackend {
public:
    void dispatch(const GraphNode& node, const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* pool) override {
        if(node.op_type!=OpType::SHORTCONV){CPUBackend::dispatch(node,inputs,output,pool);return;}
        for(int group=0;group<3;++group)for(int row=0;row<5;++row)
            output->ptr<float>()[group*5+row]=100.f*group+row;
    }
};
int main(int argc,char** argv) {
    const bool dim_major=argc==2&&!std::strcmp(argv[1],"--dim-major");
    if(argc>1&&!dim_major)return 64;
    Graph graph;
    graph.nodes.resize(2); graph.runtime.tensors.resize(2);
    for (int i=0;i<2;++i) {
        graph.nodes[i].id=i; graph.nodes[i].out_shape[0]=3;
        graph.nodes[i].out_shape[1]=5;
        graph.runtime.tensors[i]=Tensor::create(Precision::FP32,MemoryType::NONE,3,5,1,1,nullptr);
    }
    graph.nodes[0].op_type=OpType::INPUT;
    graph.nodes[1].op_type=OpType::ADD; graph.nodes[1].inputs={0,0};
    if(dim_major){graph.nodes[1].op_type=OpType::SHORTCONV;graph.nodes[1].inputs={0};}
    graph.graph_inputs={0}; graph.graph_outputs={1};
    float input[15]; for(int i=0;i<15;++i)input[i]=float(i);
    graph.runtime.tensors[0]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,3,5,1,1,input);
    DimMajorProbe backend; BufferPool pool; ExecContext ctx;
    ctx.graph=&graph;ctx.backend=&backend;ctx.pool=&pool;ctx.runtime_seq_len=5;
    prepare_execution(ctx);execute_graph(ctx);
    if(ctx.execution_failed)return 1;
    const auto& out=graph.runtime.tensors[1];
    if(!out.data)return 2;
    for(int i=0;i<15;++i)
        if(out.ptr<float>()[i]!=(dim_major?100.f*(i/5)+i%5:2.f*i))return 3;
    std::puts("dump does not modify output");
}
