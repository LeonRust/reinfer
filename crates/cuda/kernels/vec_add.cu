// vec_add：012 C1 链路最小内核。
// 导出必须 extern "C" __global__（符号否则为 C++ mangled，加载 500 失败——工具链实测）。
extern "C" __global__ void vec_add(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
