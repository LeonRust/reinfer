// diff 内核：rms_norm / rope / masked_softmax（expose C 导出，012 D2）。
// 算法与 crates/kernels::refs 一一对应（f32 累积；禁用 -use_fast_math 由
// 编译流程保证——kernel 仅用标准数学函数：fmaxf/expf/cosf/sinf/powf/rsqrtf）。
#include <math.h>

// 单行 RMSNorm：x / sqrt(mean(x^2) + eps) * w；n <= 8192（循环 256-lane）。
extern "C" __global__ void rms_norm_row(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ out,
                                        int n, float eps) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < n; i += 256) { float v = x[i]; acc += v * v; }
    s[tid] = acc;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { s[tid] += s[tid + off]; }
        __syncthreads();
    }
    float rstd = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += 256) { out[i] = x[i] * rstd * w[i]; }
}

// Neox 半旋转（p 与 p+half 对）；threadIdx.x = p；half <= 1024。
extern "C" __global__ void rope_row(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int half, int pos, float eta) {
    int p = threadIdx.x;
    if (p < half) {
        float theta = (float)pos * powf(eta, -2.f * (float)p / (2.f * (float)half));
        float c = cosf(theta), s = sinf(theta);
        float a = x[p], b = x[p + half];
        out[p] = a * c - b * s;
        out[p + half] = a * s + b * c;
    }
}

// s = s + mask（mask 含 -inf 位；causal 掩码注入——014 T7）。
extern "C" __global__ void add_f32_f32_inplace(float* __restrict__ s,
                                               const float* __restrict__ mask,
                                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = s[i] + mask[i];
    }
}

// f32 转置 [rows×cols] → [cols×rows]（行主序；014 T7：S col-major → 行主序）。
extern "C" __global__ void transpose_f32(const float* __restrict__ x,
                                          float* __restrict__ out,
                                          int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        out[c * rows + r] = x[r * cols + c];
    }
}

// f16 转置 [rows×cols] → [cols×rows]（行主序；014 T7：K → K^T 行序）。
extern "C" __global__ void transpose_f16(const unsigned short* __restrict__ x,
                                         unsigned short* __restrict__ out,
                                         int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        out[c * rows + r] = x[r * cols + c];
    }
}

// f32 → f16 元素转换（RNE；014 T7：softmax 输出进 PV 前的 dtype 一致化）。
extern "C" __global__ void cast_f32_to_f16(const float* __restrict__ x,
                                           unsigned short* __restrict__ out,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float f = x[i];
    unsigned int bits = __float_as_uint(f);
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xff);
    unsigned int man = bits & 0x7fffffu;
    if (exp == 0xff) {
        out[i] = (unsigned short)(sign | 0x7c00u | ((man >> 13) & 0x3ffu));
        return;
    }
    int half_exp = exp - 127 + 15;
    if (half_exp <= 0) {
        if (half_exp < -10) {
            out[i] = (unsigned short)sign;
            return;
        }
        unsigned int subm = (man | 0x800000u) >> (1 - half_exp + 13);
        out[i] = (unsigned short)(sign | subm);
        return;
    }
    if (half_exp >= 31) {
        out[i] = (unsigned short)(sign | 0x7c00u);
        return;
    }
    out[i] = (unsigned short)(sign | ((unsigned int)half_exp << 10) | (man >> 13));
}

// 批量行式 masked softmax（014 T7）：grid = rows（每行 1 block）；行长 ≤ 4096。
extern "C" __global__ void masked_softmax_matrix(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int rows,
                                                 int rowlen) {
    int r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row = x + (size_t)r * rowlen;
    float* orow = out + (size_t)r * rowlen;
    __shared__ float s[256];
    int tid = threadIdx.x;
    float m = -INFINITY;
    for (int i = tid; i < rowlen; i += 256) { m = fmaxf(m, row[i]); }
    s[tid] = m;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { s[tid] = fmaxf(s[tid], s[tid + off]); }
        __syncthreads();
    }
    float maxv = s[0];
    float sum = 0.f;
    for (int i = tid; i < rowlen; i += 256) { sum += expf(row[i] - maxv); }
    s[tid] = sum;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { s[tid] += s[tid + off]; }
        __syncthreads();
    }
    // 全无效行（max=-inf）：输出全 0（与 refs 一致）。
    if (!isfinite(maxv)) {
        for (int i = tid; i < rowlen; i += 256) { orow[i] = 0.f; }
        return;
    }
    float inv = (s[0] != 0.f) ? 1.f / s[0] : 0.f;
    for (int i = tid; i < rowlen; i += 256) {
        float e = expf(row[i] - maxv);
        orow[i] = e * inv;
    }
}

// 单行 softmax（输入已含 -inf 掩码位；无效位输出 0——exp(-inf)=0 数学结果，
// 与 refs::masked_softmax_ref 一致；全无效行 → 全 0）。
extern "C" __global__ void masked_softmax_row(const float* __restrict__ x,
                                              float* __restrict__ out,
                                              int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float m = -INFINITY;
    for (int i = tid; i < n; i += 256) { m = fmaxf(m, x[i]); }
    s[tid] = m;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { s[tid] = fmaxf(s[tid], s[tid + off]); }
        __syncthreads();
    }
    float maxv = s[0];
    float sum = 0.f;
    for (int i = tid; i < n; i += 256) { sum += expf(x[i] - maxv); }
    s[tid] = sum;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { s[tid] += s[tid + off]; }
        __syncthreads();
    }
    if (!(s[0] > 0.0f)) {
        // 全无效行（sum==0）：输出全 0（与 ref 一致，避免 0*inf 的 NaN）
        for (int i = tid; i < n; i += 256) { out[i] = 0.0f; }
        return;
    }
    float inv = 1.f / s[0];
    for (int i = tid; i < n; i += 256) {
        float e = expf(x[i] - maxv);
        out[i] = e * inv; // 无效位 exp(-inf)=0 → 0（-inf 输入不产生 NaN）
    }
}
