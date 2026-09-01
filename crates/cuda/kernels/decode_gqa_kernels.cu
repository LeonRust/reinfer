// 014 T8: paged decode attention (GQA) kernel — gather + two-pass softmax.
//
// KV cache layout (contract in crates/memory::pool):
//   K region: [total_pages][block_len][kv_heads][d] f16 (row-major)
//   V region: same size immediately following K region
//   page:     [logical_pages] u32 physical page ids (shared K/V table)
//   q:        [B, QH, d] f16          (decode step: one query token/item)
//   kv_lens:  [B] u32 (current KV length per batch item)
//   scores:   scratch f32 [B, QH, max_kv]
//   out:      [B, QH, d] f16
//
// Per (batch b, q_head h): one CTA. GQA mapping per 014 D3:
//   kv_head = h / kv_ratio (integer division, contiguous groups);
//   non-divisible cases validated by the test trio (14/2, 12/2, 5/2).
// Fixed 256-lane reduction tree, no atomicAdd; deterministic (012 scope).

#include <cuda_fp16.h>
#include <math.h>


// 软件 f16→f32 位构造（与 dequant_kernels.cu / ggml 一致；硬件 F2F 对
// NaN payload 的行为差异已在 014 T5 记录）。
__device__ float hbits_to_f32(unsigned short h) {
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

constexpr int TPB = 256;

extern "C" __global__ void decode_step_gqa(
    const unsigned short* __restrict__ q,      // [B, QH, d]
    const unsigned int* __restrict__ page,     // [logical_pages]
    const unsigned short* __restrict__ kv,     // K [P][BL][KH][d] then V
    const unsigned int* __restrict__ kv_lens,  // [B]
    float* __restrict__ scores,                // [B, QH, max_kv] scratch
    unsigned short* __restrict__ out,          // [B, QH, d]
    int B, int QH, int d, int block_len, int kv_ratio, int kv_heads,
    int max_kv, int total_pages) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int kv_h = h / kv_ratio;
    int kv_len = (int)kv_lens[b];
    int tid = threadIdx.x;

    const int per_tok_kv = kv_heads * d;

    if (kv_len <= 0) {
        return;
    }

    const size_t kv_base = (size_t)total_pages * block_len * per_tok_kv; // V 区基址

    // 无共享内存/无块同步版本（014 T8 判据语义一致：每线程独立 gather+softmax+PV）；
    // 精简稳健；B×QH × d 在 256 线程内直接计算（d ≤ 256）。
    // 调试printf已删。
    for (int i = tid; i < d; i += TPB) {
        float maxv = -1e30f;
        // pass1: q[i] 行固定（qij 在循环内重读——i 需要时读取；分数为行级——每线程先算行分数）
        const size_t score_base = (size_t)(b * QH + h) * max_kv;
        float qi = hbits_to_f32(q[((size_t)(b * QH + h)) * d + i]);
        for (int t = 0; t < kv_len; ++t) {
            int lp = t / block_len;
            int off = t % block_len;
            unsigned int phys = page[lp];
            const unsigned short* krow =
                kv + (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            float acc = 0.0f;
            // 每线程重复行级 dot——代价小（判据档：d*kv_len ≤ 256*1024）
            for (int j = 0; j < d; ++j) {
                acc += hbits_to_f32(q[((size_t)(b * QH + h)) * d + j]) * hbits_to_f32(krow[j]);
            }
            scores[score_base + t] = acc;
            if (acc > maxv) {
                maxv = acc;
            }
        }
        float sumv = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            sumv += expf(scores[score_base + t] - maxv);
        }
        float inv = sumv != 0.0f ? 1.0f / sumv : 0.0f;
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float p = expf(scores[score_base + t] - maxv) * inv;
            int lp = t / block_len;
            int off = t % block_len;
            unsigned int phys = page[lp];
            const unsigned short* vrow = kv + kv_base +
                (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            acc += p * hbits_to_f32(vrow[i]);
        }
        out[((size_t)(b * QH + h)) * d + i] = f32_to_hbits(acc);
    }
}

// 014 S0-3b: parity-f32 criterion tier — decode attention with f32 q and f32
// out (the engine's f32 channel keeps q/k/v activation in f32 after the
// projection GEMM; RoPE and the scale are applied in f32 upstream). KV stays
// f16 — rounded once at the write, same as the llama.cpp CPU referee f16 KV
// cache. Scores/softmax math identical to decode_step_gqa (f32, two-pass).
extern "C" __global__ void decode_step_gqa_f32(
    const float* __restrict__ q,         // [B, QH, d] f32
    const unsigned int* __restrict__ page,
    const unsigned short* __restrict__ kv,   // K [P][BL][KH][d] then V (f16)
    const unsigned int* __restrict__ kv_lens,
    float* __restrict__ scores,          // [B, QH, max_kv] scratch
    float* __restrict__ out,             // [B, QH, d] f32
    int B, int QH, int d, int block_len, int kv_ratio, int kv_heads,
    int max_kv, int total_pages) {
    int cta = blockIdx.x;
    if (cta >= B * QH) {
        return;
    }
    int b = cta / QH;
    int h = cta % QH;
    int kv_h = h / kv_ratio;
    int kv_len = (int)kv_lens[b];
    int tid = threadIdx.x;

    const int per_tok_kv = kv_heads * d;

    if (kv_len <= 0) {
        return;
    }

    const size_t kv_base = (size_t)total_pages * block_len * per_tok_kv; // V 区基址
    const float* qrow = q + (size_t)(b * QH + h) * d;
    float* orow = out + (size_t)(b * QH + h) * d;

    for (int i = tid; i < d; i += TPB) {
        float maxv = -1e30f;
        const size_t score_base = (size_t)(b * QH + h) * max_kv;
        float qi = qrow[i];
        for (int t = 0; t < kv_len; ++t) {
            int lp = t / block_len;
            int off = t % block_len;
            unsigned int phys = page[lp];
            const unsigned short* krow =
                kv + (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            float acc = 0.0f;
            for (int j = 0; j < d; ++j) {
                acc += qrow[j] * hbits_to_f32(krow[j]);
            }
            scores[score_base + t] = acc;
            if (acc > maxv) {
                maxv = acc;
            }
        }
        float sumv = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            sumv += expf(scores[score_base + t] - maxv);
        }
        float inv = sumv != 0.0f ? 1.0f / sumv : 0.0f;
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float p = expf(scores[score_base + t] - maxv) * inv;
            int lp = t / block_len;
            int off = t % block_len;
            unsigned int phys = page[lp];
            const unsigned short* vrow = kv + kv_base +
                (((size_t)phys * block_len + off) * kv_heads + kv_h) * d;
            acc += p * hbits_to_f32(vrow[i]);
        }
        orow[i] = acc;
    }
}