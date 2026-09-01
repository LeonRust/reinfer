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

// Batched half-split RoPE (NEOX) with optional element scale — one block per
// head row. Replaces the q_heads+kv_heads per-head `rope_neox_f16` launches
// plus the separate `scale_f16` pass (S1-2 launch-count wave): per layer the
// decode step drops 32 rope launches + 1 scale launch to 2 launches.
//
// Bit-identical to the original two-pass sequence:
//   rope writes f32_to_hbits(v1); scale reads hbits_to_f32(that) * scale and
//   rounds again — the fused kernel applies exactly that rounding order
//   (round rope result to f16, widen, multiply, round to f16). For scale=1.0
//   (k heads are not scaled) the f16 rounding is idempotent, so the result
//   is bit-identical to plain rope.
extern "C" __global__ void rope_heads_f16(unsigned short* __restrict__ x,
                                          int heads, int half, int pos,
                                          float eta, float scale) {
    int p = threadIdx.x;
    if (p >= half) {
        return;
    }
    int row = blockIdx.x;
    unsigned short* xr = x + (size_t)row * 2 * half;
    // Same theta formula as rope_neox_f16 (full-dimension frequency denominator).
    float theta = (float)pos * powf(eta, -2.f * (float)p / (2.f * (float)half));
    float c = cosf(theta), s = sinf(theta);
    float a = hbits_to_f32(xr[p]);
    float b = hbits_to_f32(xr[p + half]);
    float v1 = a * c - b * s;
    float v2 = a * s + b * c;
    xr[p] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v1)) * scale);
    xr[p + half] = f32_to_hbits(hbits_to_f32(f32_to_hbits(v2)) * scale);
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

// Fused f32 -> f16 cast + residual add: out[i] = out[i] + f16(x[i]).
// Replaces the `cast_f32_to_f16` (diff_kernels.cu) + `add_f16_to_f16`
// two-launch sequence at the o-projection and ffn residuals (S1-2 wave).
//
// Bit-identical: the cast kernel writes f32_to_hbits(x[i]); the add kernel
// computes f32_to_hbits(hbits_to_f32(out[i]) + hbits_to_f32(that)) — the
// fused kernel evaluates the same expression with the same rounding order.
extern "C" __global__ void add_cast_f16(unsigned short* __restrict__ out,
                                        const float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = hbits_to_f32(out[i]) + hbits_to_f32(f32_to_hbits(x[i]));
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

// ---------------------------------------------------------------------------
// S1-4: fused FFN micro-kernels (006-2 T4 ②①) — bit-identical replacements
// for the split sequences they replace (the same f16 round/widen/round
// ordering order, by construction):
//   - fused_cast_swiglu_f16: cast_f32_to_f16(gate) + cast_f32_to_f16(up) +
//     swiglu_f16 -> one launch (3 -> 1); the gate/up GEMM f32 outputs come
//     in, the f16 SiLU-GLU product goes out (the two f16 intermediate
//     buffers are skipped).
//   - fused_add_rms_f16: add_cast_f16 + rms_norm_row_f16 -> one launch
//     (2 -> 1) for the o-projection residual followed by the FFN RMSNorm;
//     the residual stream x is still updated in place (the ffn down
//     residual add re-reads it later in the layer).
// ---------------------------------------------------------------------------

// Fused f32 -> f16 cast + SiLU-GLU: out[i] = f16(silu(g16) * u16), where
// g16/u16 are the RNE f16 roundings of the gate/up GEMM outputs. Replaces
// cast_f32_to_f16 (gate), cast_f32_to_f16 (up), swiglu_f16 (3 launches ->
// 1). Bit-identical: the casts write f32_to_hbits(x); swiglu reads those
// bits widened to f32, computes silu(g)*u in f32 and rounds once — the
// fused kernel evaluates the same expression with the same rounding order.
extern "C" __global__ void fused_cast_swiglu_f16(const float* __restrict__ gate,
                                                 const float* __restrict__ up,
                                                 unsigned short* __restrict__ out,
                                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float g = hbits_to_f32(f32_to_hbits(gate[i]));
    float u = hbits_to_f32(f32_to_hbits(up[i]));
    float silu = g / (1.f + expf(-g));
    out[i] = f32_to_hbits(silu * u);
}

// Fused residual add + RMSNorm row (single block 256; n <= 1024):
//   x[i] = f16(f32(x[i]) + f16(c[i]))   (in place — same bits as add_cast)
//   out[i] = f16(f32(x'[i]) * rsqrtf(mean(x'^2) + eps) * w[i])   (same as rms)
// Replaces add_cast_f16 + rms_norm_row_f16 (2 launches -> 1). Bit-identical:
// each element's f16-rounded sum is cached in registers, so the mean-square
// pass and the normalize pass read exactly the stored f16 bits, in the same
// per-thread iteration order as the split kernels.
extern "C" __global__ void fused_add_rms_f16(unsigned short* __restrict__ x,
                                             const float* __restrict__ c,
                                             unsigned short* __restrict__ out,
                                             const unsigned short* __restrict__ w,
                                             int n, float eps) {
    int tid = threadIdx.x;
    __shared__ float s_sh[256];
    // Fused sum with the f16 rounding of the add_cast write (round addend to
    // f16, add in f32, round the sum to f16); keep the widened f16 values for
    // the rms passes (identical to the rms kernel re-reading the stored bits).
    float v[4]; // n <= 1024, 256 threads -> at most 4 elements per thread
    int cnt = 0;
    for (int i = tid; i < n; i += 256) {
        float sum = hbits_to_f32(x[i]) + hbits_to_f32(f32_to_hbits(c[i]));
        unsigned short v16 = f32_to_hbits(sum);
        x[i] = v16;
        v[cnt++] = hbits_to_f32(v16);
    }
    float s = 0.f;
    for (int k = 0; k < cnt; ++k) {
        s += v[k] * v[k];
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
    for (int i = tid, k = 0; i < n; i += 256, ++k) {
        out[i] = f32_to_hbits(v[k] * rstd * hbits_to_f32(w[i]));
    }
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

// ---------------------------------------------------------------------------
// 014 S0-3b: parity-f32 criterion tier — f32 variants of the dense small
// kernels (activations in f32; weights are the f16-valued tensors expanded to
// f32 at load, so the values match the f16 channel bit for bit).
// ---------------------------------------------------------------------------

// embed 行拷贝（f32 目的；src 为 parity 档 f32 embed）。
extern "C" __global__ void gather_row_f32(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          int row, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[(size_t)row * n + i];
    }
}

// 元素缩放：x[i] *= scale（f32；注意力 score 的 1/sqrt(d) 缩放点）。
extern "C" __global__ void scale_f32(float* __restrict__ x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * scale;
    }
}

// SiLU-GLU：out[i] = silu(gate[i]) * up[i]（f32；与 swiglu_f16 同数学）。
extern "C" __global__ void swiglu_f32(const float* __restrict__ gate,
                                      const float* __restrict__ up,
                                      float* __restrict__ out,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float g = gate[i];
    float u = up[i];
    float silu = g / (1.f + expf(-g));
    out[i] = silu * u;
}
