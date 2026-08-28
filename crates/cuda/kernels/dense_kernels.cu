// dense 解码前向小内核（014 L3 单请求；f16 位存储层 + f32 数学累积）。
//
// 数值语义对齐 CPU 参考（crates/cpu/src/ops.rs）：
// - RMSNorm：`x / sqrt(mean(x²) + eps) * w`；
// - 半旋转 RoPE（ggml NEOX 布局 [ccccssss]——对偶 (p, p+half)，θ =
//   pos·eta^(-2p/half)）；
// - SiLU-GLU：`silu(gate) * up`；
// - f16 转换：与 diff_kernels.cu `cast_f32_to_f16`（RNE 位构造）一致。
//
// 无模型名/架构名特判：形状由调用方传入。

__device__ __forceinline__ float hbits_to_f32(unsigned short h) {
    // 软件位构造（dequant_kernels.cu 已验证版的一致实现——T5 0-ulp 判据过）。
    unsigned int sign = (unsigned int)(h >> 15) << 31;
    unsigned int exp  = (h >> 10) & 0x1f;
    unsigned int man  = h & 0x03ff;
    if (exp == 0) {
        if (man == 0) {
            return __uint_as_float(sign);
        }
        unsigned int m = man;
        unsigned int s = 0;
        while ((m & 0x0400) == 0) {
            m <<= 1;
            s += 1;
        }
        return __uint_as_float(sign | ((113 - s) << 23) | ((m & 0x03ff) << 13));
    }
    if (exp == 0x1f) {
        return __uint_as_float(sign | 0x7f800000u | (man << 13));
    }
    return __uint_as_float(sign | ((exp + 112) << 23) | (man << 13));
}

__device__ __forceinline__ unsigned short f32_to_hbits(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xff);
    unsigned int man = bits & 0x7fffffu;
    if (exp == 0xff) {
        return (unsigned short)(sign | 0x7c00u | ((man >> 13) & 0x3ffu));
    }
    int half_exp = exp - 127 + 15;
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return (unsigned short)sign;
        }
        unsigned int subm = (man | 0x800000u) >> (1 - half_exp + 13);
        return (unsigned short)(sign | subm);
    }
    if (half_exp >= 31) {
        return (unsigned short)(sign | 0x7c00u);
    }
    return (unsigned short)(sign | ((unsigned int)half_exp << 10) | (man >> 13));
}

// RMSNorm 行：out = x / sqrt(mean(x²)+eps) * w；单 block（n ≤ 1024）。
// NORMAL：f16 bits 进出；w f16 bits；均值平方与 1/sqrt 在 f32 进行。
extern "C" __global__ void rms_norm_row_f16(const unsigned short* __restrict__ x,
                                            unsigned short* __restrict__ out,
                                            const unsigned short* __restrict__ w,
                                            int n, float eps) {
    int tid = threadIdx.x;
    __shared__ float s_sh[256];
    float s = 0.f;
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(x[i]);
        s += v * v;
    }
    s_sh[tid] = s;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            s_sh[tid] += s_sh[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mean_sq = s_sh[0] / (float)n;
        s_sh[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rstd = s_sh[0];
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(x[i]) * rstd * hbits_to_f32(w[i]);
        out[i] = f32_to_hbits(v);
    }
}

// 每行作 RMSNorm（多行共用一个 权重向量；grid = rows，block = 256）。
// 用途：Qwen3 q_norm/k_norm——每头行 [hdim]；rows = q_heads（或 kv_heads）。
extern "C" __global__ void rms_norm_heads_f16(const unsigned short* __restrict__ x,
                                              unsigned short* __restrict__ out,
                                              const unsigned short* __restrict__ w,
                                              int rows, int n, float eps) {
    int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * n;
    unsigned short* orr = out + (size_t)row * n;
    int tid = threadIdx.x;
    __shared__ float s_sh[256];
    float s = 0.f;
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(xr[i]);
        s += v * v;
    }
    s_sh[tid] = s;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            s_sh[tid] += s_sh[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_sh[0] = rsqrtf(s_sh[0] / (float)n + eps);
    }
    __syncthreads();
    float rstd = s_sh[0];
    for (int i = tid; i < n; i += 256) {
        float v = hbits_to_f32(xr[i]) * rstd * hbits_to_f32(w[i]);
        orr[i] = f32_to_hbits(v);
    }
}

// 半旋转 RoPE（NEOX [ccccssss]）就地：对偶 (p, p+half)；θ_p = pos·eta^(-2p/half)。
// threadIdx.x = p < half；half ≤ 1024。
extern "C" __global__ void rope_neox_f16(unsigned short* __restrict__ x,
                                         int half, int pos, float eta) {
    int p = threadIdx.x;
    if (p < half) {
        // θ_p = pos·eta^(-2p/(2·half))——ggml NEOX 频率分母为**全维**（n_rot=2·half）
        float theta = (float)pos * powf(eta, -2.f * (float)p / (2.f * (float)half));
        float c = cosf(theta), s = sinf(theta);
        float a = hbits_to_f32(x[p]);
        float b = hbits_to_f32(x[p + half]);
        x[p] = f32_to_hbits(a * c - b * s);
        x[p + half] = f32_to_hbits(a * s + b * c);
    }
}

// out = out + x（f16 bits；残差加——x 为 attn/ffn 输出，out 为层跳线）。
extern "C" __global__ void add_f16_to_f16(unsigned short* __restrict__ out,
                                          const unsigned short* __restrict__ x,
                                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = hbits_to_f32(out[i]) + hbits_to_f32(x[i]);
    out[i] = f32_to_hbits(v);
}

// SiLU-GLU：out[i] = silu(gate[i]) * up[i]（f32 数学）。
extern "C" __global__ void swiglu_f16(const unsigned short* __restrict__ gate,
                                      const unsigned short* __restrict__ up,
                                      unsigned short* __restrict__ out,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float g = hbits_to_f32(gate[i]);
    float u = hbits_to_f32(up[i]);
    float silu = g / (1.f + expf(-g));
    out[i] = f32_to_hbits(silu * u);
}

// KV 行写：把当前 token 的 K 行（[kv_heads*d]）与 V 行写进页布局。
// 布局同 decode_step_gqa：k 区 [total_pages][block_len][kv_heads][d]；
// v 区紧随（k 区元素数 = total_pages*block_len*per_tok）。
// 无掩码（即写即用，页面由调用方保证已分配）。
extern "C" __global__ void kv_write_row(const unsigned short* __restrict__ k_row,
                                        const unsigned short* __restrict__ v_row,
                                        unsigned short* __restrict__ kv,
                                        int phys, int off, int block_len,
                                        int kv_heads, int d, int total_pages) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per_tok = kv_heads * d;
    if (i >= per_tok) {
        return;
    }
    int kh = i / d;
    int di = i % d;
    int tok_base = ((phys * block_len + off) * kv_heads + kh) * d + di;
    kv[tok_base] = k_row[i];
    size_t k_region = (size_t)total_pages * block_len * per_tok;
    kv[k_region + tok_base] = v_row[i];
}

// 元素缩放：x[i] *= scale（f32 数学——注意力 score 的 1/sqrt(d) 缩放点）。
extern "C" __global__ void scale_f16(unsigned short* __restrict__ x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = f32_to_hbits(hbits_to_f32(x[i]) * scale);
    }
}

// embed 行拷贝：src[row*n ..] → dst[..]（单行 n 个 f16；grid 1 block）。
extern "C" __global__ void gather_row(const unsigned short* __restrict__ src,
                                      unsigned short* __restrict__ dst,
                                      int row, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[(size_t)row * n + i];
    }
}
