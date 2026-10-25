// Compiled, never launched. Check the feature image used by native NVFP4 MMA.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
#if !defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(__CUDA_ARCH_FEAT_SM121_ALL)
#error Native NVFP4 requires the architecture-specific CUDA target
#endif
#endif
__global__ void mollm_cuda_architecture_probe() {}
int main() { return 0; }
