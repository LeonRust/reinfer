/* Q8_0 dequantization reference dumper — 014 T2 golden truth generator.
 *
 * Links the referee llama.cpp build's libggml-cpu (f280b2698) and calls its
 * exported dequantize_row_q8_0, producing the bit-exact reference f32
 * values that crates/gguf::codes::dequantize_q8_0 must match (0 ulp).
 *
 * Input:  raw Q8_0 block bytes on stdin (34 B per block: fp16 scale LE
 *         + 32 int8), QK8_0 blocks total.
 * Output: each f32 as 8-hex-digit little-endian bit pattern, one per line.
 *
 * Build: gcc -O2 -o q8_0_refdump q8_0_refdump.c -I<repo>/llama.cpp/ggml/include \
 *          -L<repo>/llama.cpp/build/bin -lggml-cpu -lggml-base -Wl,-rpath,<...> -lm
 *
 * The tool is a verification aid only — it never participates in the
 * reinfer engine (runtime is 100% Rust; see 014 T0/T2).
 */
#include <stdint.h>
#include <stdio.h>
#ifndef RESTRICT
#define RESTRICT restrict
#endif

/* Mirrors ggml block_q8_0 layout: 2-byte fp16 scale + 32 int8 (34 B). */
typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_q8_0;

extern void dequantize_row_q8_0(const block_q8_0 * RESTRICT x, float * RESTRICT y, int64_t k);

int main(void) {
    block_q8_0 blk;
    float y[32];
    while (fread(&blk, sizeof(blk), 1, stdin) == 1) {
        dequantize_row_q8_0(&blk, y, 32);
        for (int i = 0; i < 32; ++i) {
            uint32_t bits;
            __builtin_memcpy(&bits, &y[i], 4);
            printf("%08x\n", bits);
        }
    }
    return 0;
}
