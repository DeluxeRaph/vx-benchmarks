// Compiled by NVRTC for compute_87; one output element per thread.
extern "C" __global__ void add(float* result, const float* a, const float* b) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = a[i] + b[i];
}
