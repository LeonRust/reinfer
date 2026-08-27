//! diff 内核的 CPU 参考（纯函数；012 T5-D7/plan D6）。
//!
//! 所有算法语义**与 GPU 侧定义一致**（D2 差分目标）：
//! - `rms_norm`：均值平方（f32 累积）→ `x / sqrt(var+eps) * w`；
//! - `rope`：Neox 式半旋转（对偶 `(i, i+half)`），f32 累积，eta 由调用方给定；
//! - `masked_softmax`：online-max，无效位输出 `-inf`（掩码一致即匹配；
//!   全无效行 → 全 `-inf`，两侧语义一致即可）。

/// 单行 RMSNorm：`x / sqrt(mean(x²) + eps) * w`（len(x) == len(w)）。
pub fn rms_norm_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    if x.is_empty() {
        return vec![];
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rstd = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(&v, &wt)| v * rstd * wt).collect()
}

/// 单头单位置 RoPE（Neox 半旋转）：
/// `p = i ∈ [0, half)`，`θ_p = pos * eta^(-2p/half)`；
/// `x'[p] = x[p]·cosθ - x[p+half]·sinθ`；`x'[p+half] = x[p]·sinθ + x[p+half]·cosθ`。
/// 要求 `x.len() == 2*half`。
pub fn rope_ref(x: &[f32], half: usize, pos: u32, eta: f32) -> Vec<f32> {
    assert_eq!(x.len(), 2 * half);
    let mut out = x.to_vec();
    for p in 0..half {
        let exp = -(2.0 * p as f32) / half as f32;
        let theta = pos as f32 * eta.powf(exp);
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = (x[p], x[p + half]);
        out[p] = a * c - b * s;
        out[p + half] = a * s + b * c;
    }
    out
}

/// 单行 masked softmax：`mask[i] == false` 的位置输出 **0**（数学结果
/// 等价于输入取 `-inf` 后 `exp→0`——GPU 侧同语义）；全无效行 → 全 0。
/// 掩码位置 D6 规则：掩码一致即匹配（无效位都输出 0，不参与容差）。
pub fn masked_softmax_ref(x: &[f32], mask: &[bool]) -> Vec<f32> {
    assert_eq!(x.len(), mask.len());
    let mut max_v = f32::NEG_INFINITY;
    for (&v, &m) in x.iter().zip(mask) {
        if m && v > max_v {
            max_v = v;
        }
    }
    let mut out = vec![0.0f32; x.len()];
    if max_v.is_finite() {
        let mut sum = 0.0f32;
        for (i, (&v, &m)) in x.iter().zip(mask).enumerate() {
            if m {
                let e = (v - max_v).exp();
                sum += e;
                out[i] = e;
            }
        }
        if sum > 0.0 {
            for (v, m) in out.iter_mut().zip(mask) {
                if *m {
                    *v /= sum;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn rms_norm_zero_row_and_unit() {
        // 全零行（eps 语义：无 NaN，输出 0）
        let out = rms_norm_ref(&[0.0; 4], &[1.0; 4], 1e-5);
        assert!(out.iter().all(|v| *v == 0.0));
        // w=1 时 = x / (||x||/√n)（均值平方根）
        let out = rms_norm_ref(&[3.0, 4.0], &[1.0, 2.0], 0.0);
        let mean_sq: f32 = (9.0 + 16.0) / 2.0;
        let rstd = 1.0 / mean_sq.sqrt();
        assert!((out[0] - 3.0 * rstd).abs() < 1e-6);
        assert!((out[1] - 4.0 * 2.0 * rstd).abs() < 1e-6);
    }

    #[test]
    fn rope_rotation_preserves_pair_norm() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = rope_ref(&x, 3, 7, 5000.0);
        for p in 0..3 {
            let before = x[p] * x[p] + x[p + 3] * x[p + 3];
            let after = out[p] * out[p] + out[p + 3] * out[p + 3];
            assert!((before - after).abs() < 1e-4, "rotation must preserve norm");
        }
        // pos=0 → 恒等
        let id = rope_ref(&x, 3, 0, 5000.0);
        assert_eq!(id.to_vec(), x.to_vec());
    }

    #[test]
    fn masked_softmax_invalid_is_zero() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [true, true, false, false];
        let out = masked_softmax_ref(&x, &mask);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        assert!(out[0] > 0.0 && out[1] > 0.0);
        assert!((out[0] + out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn masked_softmax_all_masked_row() {
        let out = masked_softmax_ref(&[1.0, 2.0], &[false, false]);
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
