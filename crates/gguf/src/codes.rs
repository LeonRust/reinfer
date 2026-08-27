//! 张量数据解量化 codec（014 T2）：Q8_0 / F16 / FP32 → f32。
//!
//! 语义与 llama.cpp 对齐，判据为**位精确**（014 r1 差异注记：Q8_0 判据从 ≤1 ULP
//! 上调为 0 ulp——单乘语义可达成）：
//!
//! - **Q8_0**：block = 2 字节 fp16 scale（小端）+ 32 个 int8（`QK8_0 = 32`，34 B/block；
//!   256 是 K 系量化的块宽，与 Q8_0 无关）。解量化 `y = f32(q) * f32(f16(scale))`：
//!   scale 按存储的 fp16 位模式精确展开（[`f16_to_f32`]，位构造法，无查表/近似），
//!   随后一次 f32 乘法（禁止 FMA 化写法 `q * d + 0.0` 与双精度中间量）。
//!   对任意合法输入（普通/次正规 scale × 任意 i8），结果与 llama.cpp
//!   `dequantize_row_q8_0`（ggml-quants.c）逐位一致。
//! - **F16**：逐元素 fp16 → fp32（fp16 值域/精度是 fp32 的真子集，转换本身零舍入）。
//! - **FP32**：直接拷贝。
//!
//! 真值来源：llama.cpp `dequantize_row_q8_0` / `dequantize_row_f16`。
//! llama-quantize 产物金块对拍脚本见 `scripts/golden/gen_q8_0_golden.sh`
//! （随 013/001 真模型存档补水后纳入；本模块以字节级金块 + 全位模式交叉验证兜底）。

use crate::reader::GgufReader;
use crate::schema::{GgufDtype, GgufError, GgufTensor};

/// Q8_0 每块元素数（llama.cpp `QK8_0`；014 差异注记：与 K 系 256 无关）。
pub const QK8_0: usize = 32;

/// Q8_0 每块字节数：2 字节 fp16 scale + 32 字节 int8。
pub const Q8_0_BLOCK_BYTES: usize = QK8_0 + 2;

/// fp16（IEEE 754 half）位模式 → fp32 的**精确**转换（位构造法）。
///
/// 对任意位型均精确：普通数/次正规/±0/±inf/NaN 全覆盖（NaN 保持非数语义，
/// 尾数位整体左移 13 位嵌入 fp32）；不依赖查表或浮点算术，无舍入可能。
/// 与 llama.cpp `GGML_FP16_TO_FP32`（查表）数值等价——两侧都是 half 精确展开。
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x03ff);
    let out = match exp {
        // 0 指数：±0（尾数 0）或次正规（man × 2^-24；归一化到 11 位后嵌入 f32 指数）。
        0 if man == 0 => sign,
        0 => {
            let mut m = man;
            let mut s = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                s += 1;
            }
            // 次正规值 = (1 + m'/1024) × 2^(-14 - s) → f32 指数域 = 113 - s。
            sign | ((113 - s) << 23) | ((m & 0x03ff) << 13)
        }
        // 全 1 指数：±inf（尾数 0）或 NaN。
        0x1f => sign | 0x7f80_0000 | (man << 13),
        // 普通数：(1 + man/1024) × 2^(e-15) → f32 指数域 = e + 112。
        e => sign | ((e + 112) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// 解量化 Q8_0 数据（`blob` 长度必须是 [`Q8_0_BLOCK_BYTES`] 的整数倍；
/// `out` 至少能容纳 `blob 块数 × QK8_0` 个元素）。
///
/// 语义 `y[i] = f32(qs[i]) * f32(f16(scale))`：scale 直接按存储的 fp16 位模式精确
/// 展开，单个 f32 乘法（不引入 FMA/双精度中间值）。对合法输入（普通/次正规 scale、
/// 任意 i8）结果与 llama.cpp `dequantize_row_q8_0` 位精确一致；非法长度/缓冲返回
/// [`GgufError::BadData`]。
pub fn dequantize_q8_0(blob: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    if !blob.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(GgufError::BadData {
            what: "q8_0 blob length must be a multiple of 34",
            at: blob.len() as u64,
        });
    }
    let elems = (blob.len() / Q8_0_BLOCK_BYTES) * QK8_0;
    if out.len() < elems {
        return Err(GgufError::BadData {
            what: "output buffer too small for q8_0 dequantization",
            at: elems as u64,
        });
    }
    // 长度已校验为 Q8_0_BLOCK_BYTES 的整数倍 → as_chunks 无余数。
    let (blocks, rest) = blob.as_chunks::<Q8_0_BLOCK_BYTES>();
    debug_assert!(rest.is_empty());
    for (block_idx, block) in blocks.iter().enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = block_idx * QK8_0;
        for j in 0..QK8_0 {
            out[base + j] = scale * f32::from(block[2 + j] as i8);
        }
    }
    Ok(())
}

/// 解量化 FP16 数据（逐元素精确 fp16 → fp32；`blob` 长度必须是偶数；
/// `out` 至少能容纳 `blob.len() / 2` 个元素）。
pub fn dequantize_f16(blob: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    if !blob.len().is_multiple_of(2) {
        return Err(GgufError::BadData {
            what: "f16 blob length must be even",
            at: blob.len() as u64,
        });
    }
    let elems = blob.len() / 2;
    if out.len() < elems {
        return Err(GgufError::BadData {
            what: "output buffer too small for f16 dequantization",
            at: elems as u64,
        });
    }
    let (pairs, rest) = blob.as_chunks::<2>();
    debug_assert!(rest.is_empty());
    for (i, pair) in pairs.iter().enumerate() {
        out[i] = f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
    }
    Ok(())
}

/// 按张量 dtype 将整个权重张量解量化到 `Vec<f32>`。
///
/// - F32：直拷（小端）；F16：逐元素精确转换；Q8_0：按 [`dequantize_q8_0`]。
/// - 其余 dtype（K 系/IQ 系等未实现 codec）→ [`GgufError::UnsupportedDtype`]。
/// - 数据长度与 dtype×shape 推导不符 → [`GgufError::BadData`]（防御性校验；
///   正常文件由 reader 的 length 推导保证一致）。
pub fn dequantize_tensor(reader: &GgufReader, tensor: &GgufTensor) -> Result<Vec<f32>, GgufError> {
    if !matches!(tensor.dtype, GgufDtype::F32 | GgufDtype::F16 | GgufDtype::Q8_0) {
        return Err(GgufError::UnsupportedDtype(tensor.dtype));
    }
    let n = tensor.element_count().ok_or_else(|| GgufError::InvalidTensor {
        name: tensor.name.clone(),
        why: "shape product overflows u64",
    })?;
    let n_usize = usize::try_from(n).map_err(|_| GgufError::InvalidTensor {
        name: tensor.name.clone(),
        why: "shape product exceeds usize",
    })?;
    let expected =
        GgufDtype::size_bytes(tensor.dtype, n).ok_or_else(|| GgufError::InvalidTensor {
            name: tensor.name.clone(),
            why: "element count violates dtype block alignment",
        })?;
    let blob = reader.tensor_data(tensor)?;
    if blob.len() as u64 != expected {
        return Err(GgufError::BadData {
            what: "tensor data length does not match dtype x shape",
            at: blob.len() as u64,
        });
    }
    let mut out = vec![0.0f32; n_usize];
    match tensor.dtype {
        GgufDtype::F32 => {
            let (words, rest) = blob.as_chunks::<4>();
            debug_assert!(rest.is_empty());
            for (i, word) in words.iter().enumerate() {
                out[i] = f32::from_le_bytes(*word);
            }
        }
        GgufDtype::F16 => dequantize_f16(&blob, &mut out)?,
        GgufDtype::Q8_0 => dequantize_q8_0(&blob, &mut out)?,
        // 上面已 gate，此分支不可达；保留以保持 match 穷尽。
        _ => unreachable!("dtype gate above"),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试模块：金块断言直接 unwrap（仓库风格）
    #![allow(clippy::excessive_precision)] // 手工金块：十进制字面量是精确二进制分数，全精度是意图

    use super::*;
    use crate::fixture::{FixtureTensor, build_gguf};
    use proptest::prelude::*;

    /// 独立的按值参考路径（测试专用）：IEEE 754 half 的数学定义，不经位构造。
    fn half_value_f64(bits: u16) -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = (bits >> 10) & 0x1f;
        let man = (bits & 0x03ff) as u64;
        match exp {
            0 if man == 0 => 0.0,
            0 => sign * (man as f64) * 2f64.powi(-24),
            0x1f if man == 0 => sign * f64::INFINITY,
            0x1f => f64::NAN,
            e => sign * (1.0 + man as f64 / 1024.0) * 2f64.powi(i32::from(e) - 15),
        }
    }

    fn q8_0_block(scale_bits: u16, qs: &[i8]) -> Vec<u8> {
        assert_eq!(qs.len(), QK8_0);
        let mut b = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        b.extend_from_slice(&scale_bits.to_le_bytes());
        b.extend(qs.iter().map(|q| *q as u8));
        b
    }

    #[test]
    fn f16_to_f32_hand_computed_table() {
        // (half 位模式, 期望 f32) —— 手算对照（符号/阶/尾数按 IEEE 754 half 展开）。
        let cases: &[(u16, f32)] = &[
            (0x0000, 0.0),             // +0
            (0x8000, -0.0),            // -0
            (0x3c00, 1.0),             // 1.0
            (0xbc00, -1.0),            // -1.0
            (0x3e00, 1.5),             // (1 + 512/1024) × 2^0
            (0x3800, 0.5),             // (1 + 0) × 2^-1
            (0x4000, 2.0),             // 2.0
            (0xc000, -2.0),            // -2.0
            (0x3a00, 0.75),            // (1 + 512/1024) × 2^-1
            (0x7400, 16384.0),         // (1 + 0) × 2^14
            (0x3d00, 1.25),            // (1 + 256/1024) × 2^0
            (0x2e66, 0.0999755859375), // 0.1 的 half 表示（精确：1638 × 2^-14）
            (0x3e66, 1.599609375),     // (1638/1024) × 2^0
            (0x7a66, 52416.0),         // (1638/1024) × 2^15 = 1638 × 32
            (0x3555, 0.333251953125),  // (1365/1024) × 2^-2 = 1365/4096
            (0x7bff, 65504.0),         // 最大有限 half：(2047/1024) × 2^15
            (0xfbff, -65504.0),
            (0x0001, 2f32.powi(-24)),          // 最小次正规：2^-24
            (0x0002, 2f32.powi(-23)),          // 次正规：2 × 2^-24
            (0x03ff, 1023.0 * 2f32.powi(-24)), // 最大次正规：1023 × 2^-24
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ];
        for (bits, expected) in cases {
            assert_eq!(f16_to_f32(*bits).to_bits(), expected.to_bits(), "half bits {bits:#06x}");
        }
        // NaN（quiet 位 0x200 已置）。
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0xfe00).is_nan());
    }

    #[test]
    fn f16_to_f32_matches_value_reference_for_all_bit_patterns() {
        // 全 65536 个位模式：位构造实现 vs 按值参考路径（f64 精确算术）。
        // 所有非 NaN 模式的参考值在 f64 中精确（half → f64 值域/精度真子集），
        // f64::from(f32) 亦精确 → 相等即证明位精确。
        for bits in 0u16..=u16::MAX {
            let got = f16_to_f32(bits);
            let reference = half_value_f64(bits);
            if reference.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}");
            } else {
                assert_eq!(f64::from(got), reference, "bits {bits:#06x}");
            }
        }
    }

    // 手工金块（014 r1 差异注记的 QK8_0=32 判据）：
    // 块 A：scale 0x7A66 = 52416.0（half 指数域 30 → (1638/1024) × 2^15 = 1638×32），
    //       所有乘积 ≤ 6.7M < 2^24，f32 精确（无舍入）。
    const QS_A: [i8; QK8_0] = [
        0, 1, -1, 2, -2, 127, -128, 5, -5, 100, 3, -3, 64, -64, 32, -32, 16, -16, 8, -8, 4, -4, 7,
        -7, 50, -50, 126, -126, 101, -101, 99, -99,
    ];
    const EXPECTED_A: [f32; QK8_0] = [
        0.0, 52416.0, -52416.0, 104832.0, -104832.0, 6656832.0, -6709248.0, 262080.0, -262080.0,
        5241600.0, 157248.0, -157248.0, 3354624.0, -3354624.0, 1677312.0, -1677312.0, 838656.0,
        -838656.0, 419328.0, -419328.0, 209664.0, -209664.0, 366912.0, -366912.0, 2620800.0,
        -2620800.0, 6604416.0, -6604416.0, 5294016.0, -5294016.0, 5189184.0, -5189184.0,
    ];
    // 块 B：scale 0x3E00 = 1.5，半整数值（同样精确）。
    const QS_B: [i8; QK8_0] = [
        1, 2, 3, -1, -3, 64, -64, 32, -32, 10, -10, 100, -100, 127, -128, 5, -5, 4, -4, 2, -2, 8,
        -8, 16, -16, 33, -33, 6, -6, 7, -7, 1,
    ];
    const EXPECTED_B: [f32; QK8_0] = [
        1.5, 3.0, 4.5, -1.5, -4.5, 96.0, -96.0, 48.0, -48.0, 15.0, -15.0, 150.0, -150.0, 190.5,
        -192.0, 7.5, -7.5, 6.0, -6.0, 3.0, -3.0, 12.0, -12.0, 24.0, -24.0, 49.5, -49.5, 9.0, -9.0,
        10.5, -10.5, 1.5,
    ];

    #[test]
    fn dequantize_q8_0_hand_computed_bit_exact() {
        let mut blob = q8_0_block(0x7a66, &QS_A);
        blob.extend(q8_0_block(0x3e00, &QS_B));
        let mut out = [0.0f32; 2 * QK8_0];
        dequantize_q8_0(&blob, &mut out).unwrap();
        let mut expected = [0.0f32; 2 * QK8_0];
        expected[..QK8_0].copy_from_slice(&EXPECTED_A);
        expected[QK8_0..].copy_from_slice(&EXPECTED_B);
        for i in 0..2 * QK8_0 {
            assert_eq!(out[i].to_bits(), expected[i].to_bits(), "element {i}");
        }
    }

    #[test]
    fn dequantize_q8_0_length_and_buffer_errors() {
        let mut out = [0.0f32; 2 * QK8_0];
        // 非 34 倍数：33 / 69 → BadData。
        assert!(matches!(dequantize_q8_0(&[0u8; 33], &mut out), Err(GgufError::BadData { .. })));
        assert!(matches!(dequantize_q8_0(&[0u8; 69], &mut out), Err(GgufError::BadData { .. })));
        // 空 blob → Ok（0 元素）。
        assert!(dequantize_q8_0(&[], &mut []).is_ok());
        // 缓冲不足（1 块 → 需 32 元素，只给 31）→ BadData。
        let mut small = [0.0f32; QK8_0 - 1];
        assert!(matches!(
            dequantize_q8_0(&[0u8; Q8_0_BLOCK_BYTES], &mut small),
            Err(GgufError::BadData { .. })
        ));
    }

    #[test]
    fn dequantize_q8_0_special_scales_do_not_panic() {
        // 构造 32 元素块，仅前几个 q 有意义（其余 0）。
        let block = |scale_bits: u16, qs: &[i8]| {
            let mut full = [0i8; QK8_0];
            full[..qs.len()].copy_from_slice(qs);
            q8_0_block(scale_bits, &full)
        };
        // scale = +inf：q>0 → +inf，q<0 → -inf，q=0 → NaN（0 × inf）。
        let blob = block(0x7c00, &[1, -1, 0, 5]);
        let mut out = [0.0f32; QK8_0];
        dequantize_q8_0(&blob, &mut out).unwrap();
        assert_eq!(out[0], f32::INFINITY);
        assert_eq!(out[1], f32::NEG_INFINITY);
        assert!(out[2].is_nan());
        assert_eq!(out[3], f32::INFINITY);
        // scale = NaN：全部 NaN，不 panic。
        let blob = block(0x7e00, &[1, 2, 3, 4]);
        let mut out = [0.0f32; QK8_0];
        dequantize_q8_0(&blob, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn dequantize_f16_hand_bytes() {
        // 1.0 (0x3C00), -2.5 (0xC100), 3.75 (0x4380) —— 小端。
        let blob = [0x00, 0x3c, 0x00, 0xc1, 0x80, 0x43];
        let mut out = [0.0f32; 3];
        dequantize_f16(&blob, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(out[1].to_bits(), (-2.5_f32).to_bits());
        assert_eq!(out[2].to_bits(), 3.75_f32.to_bits());
        // 奇数长度 → BadData。
        let mut out = [0.0f32; 3];
        assert!(matches!(
            dequantize_f16(&[0x00, 0x3c, 0x00], &mut out),
            Err(GgufError::BadData { .. })
        ));
    }

    #[test]
    fn dequantize_tensor_fixture_three_dtypes() {
        let mut q8_data = q8_0_block(0x7a66, &QS_A);
        q8_data.extend(q8_0_block(0x3e00, &QS_B));
        let gguf = build_gguf(
            3,
            &[],
            &[
                FixtureTensor {
                    name: "f32_t".into(),
                    shape: vec![3],
                    dtype: GgufDtype::F32,
                    data: [1.0_f32, -2.5, 3.75].into_iter().flat_map(|x| x.to_le_bytes()).collect(),
                },
                FixtureTensor {
                    name: "f16_t".into(),
                    shape: vec![3],
                    dtype: GgufDtype::F16,
                    data: vec![0x00, 0x3c, 0x00, 0xc1, 0x80, 0x43],
                },
                FixtureTensor {
                    name: "q8_0_t".into(),
                    shape: vec![64],
                    dtype: GgufDtype::Q8_0,
                    data: q8_data,
                },
            ],
        );
        let reader = GgufReader::from_bytes(gguf).unwrap();

        let f32_v = dequantize_tensor(&reader, reader.tensor("f32_t").unwrap()).unwrap();
        assert_eq!(f32_v, vec![1.0_f32, -2.5, 3.75]);

        let f16_v = dequantize_tensor(&reader, reader.tensor("f16_t").unwrap()).unwrap();
        assert_eq!(f16_v, vec![1.0_f32, -2.5, 3.75]);

        let q8_v = dequantize_tensor(&reader, reader.tensor("q8_0_t").unwrap()).unwrap();
        let mut expected = [0.0f32; 2 * QK8_0];
        expected[..QK8_0].copy_from_slice(&EXPECTED_A);
        expected[QK8_0..].copy_from_slice(&EXPECTED_B);
        for (i, (got, exp)) in q8_v.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.to_bits(), exp.to_bits(), "element {i}");
        }
    }

    #[test]
    fn dequantize_tensor_rejects_unsupported_dtype() {
        // Q4_0 codec 未实现 → UnsupportedDtype。
        let gguf = build_gguf(
            3,
            &[],
            &[FixtureTensor {
                name: "q4_0_t".into(),
                shape: vec![32],
                dtype: GgufDtype::Q4_0,
                data: vec![0u8; 18],
            }],
        );
        let reader = GgufReader::from_bytes(gguf).unwrap();
        assert!(matches!(
            dequantize_tensor(&reader, reader.tensor("q4_0_t").unwrap()),
            Err(GgufError::UnsupportedDtype(GgufDtype::Q4_0))
        ));
        // 元素数不满足块对齐（63 非 32 倍数）：reader 解析期即拒绝 InvalidTensor
        // （dequantize_tensor 内的同类校验是防御性路径，正常文件不可达）。
        let bad = build_gguf(
            3,
            &[],
            &[FixtureTensor {
                name: "q8_bad_t".into(),
                shape: vec![63],
                dtype: GgufDtype::Q8_0,
                data: vec![],
            }],
        );
        assert!(matches!(GgufReader::from_bytes(bad), Err(GgufError::InvalidTensor { .. })));
    }

    proptest! {
        /// 随机 blob：非法长度必然 Err，合法长度绝不 panic；且与 f64 按值参考
        /// 逐元素一致（合法输入的乘积在 f32/f64 中均精确 → 位精确判据的机器证明）。
        #[test]
        fn q8_0_random_blobs_match_f64_reference(blob in prop::collection::vec(any::<u8>(), 0..2048)) {
            let n = blob.len();
            let mut out = vec![0.0f32; n];
            let r = dequantize_q8_0(&blob, &mut out);
            if !n.is_multiple_of(Q8_0_BLOCK_BYTES) {
                prop_assert!(r.is_err());
                return Ok(());
            }
            prop_assert!(r.is_ok());
            let (blocks, rest) = blob.as_chunks::<Q8_0_BLOCK_BYTES>();
            debug_assert!(rest.is_empty());
            for (i, block) in blocks.iter().enumerate() {
                let scale = half_value_f64(u16::from_le_bytes([block[0], block[1]]));
                if !scale.is_finite() {
                    continue; // inf/NaN scale：跳过（语义是传播，不参与数值对照）
                }
                let base = i * QK8_0;
                for j in 0..QK8_0 {
                    let expected = scale * f64::from(block[2 + j] as i8);
                    prop_assert_eq!(f64::from(out[base + j]), expected, "block {} elem {}", i, j);
                }
            }
        }
    }
}
