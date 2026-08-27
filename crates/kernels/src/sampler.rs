//! sample host 管线（012 D3；005 的 RNG 数学锚点）。
//!
//! `SplitMix64` 为纯函数确定性 RNG（同 seed → 同流；无设备随机性）。
//! 采样：`sample_from_probs`（累计分布 + 均匀噪声）；温度语义归上层
//! （temp=0 → `sample_argmax` 决定论；temp>0 由调用方先做
//! `logits/temp → softmax`——组合差分链路直接使用 GPU softmax 输出）。

use crate::error::LaunchError;

/// SplitMix64 确定性 RNG。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// 新建（种子任意；同种子同流）。
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// 下一个 64 位值（标准 SplitMix64 混频）。
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// [0,1) 均匀分布（53 位精度取法，避免 1.0）。
    pub fn next_f32_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}

/// 全 mask → Err 的哨兵消息（fail-closed：拒绝静默错采样）。
pub fn all_masked_message() -> &'static str {
    "sampler: all tokens masked"
}

/// argmax 决定论采样（temp=0；mask 内取最大值；全 masked → 错误）。
pub fn sample_argmax(logits: &[f32], mask: &[bool]) -> Result<usize, LaunchError> {
    assert_eq!(logits.len(), mask.len());
    let mut best: Option<(usize, f32)> = None;
    for (i, (&l, &m)) in logits.iter().zip(mask).enumerate() {
        if !m {
            continue;
        }
        if best.map_or(true, |(_, bl)| l > bl) {
            best = Some((i, l));
        }
    }
    best.map(|(i, _)| i).ok_or(LaunchError::Fatal)
}

/// 按概率累计分布采样（GPU softmax 输出与 CPU 参考共用——组合差分锚点）。
/// 概率和为 ≤1（浮点）；全 0/全 masked → 错误。
pub fn sample_from_probs(probs: &[f32], rng: &mut SplitMix64) -> Result<usize, LaunchError> {
    let total: f32 = probs.iter().copied().sum();
    if !(total > 0.0) {
        return Err(LaunchError::Fatal);
    }
    let u = rng.next_f32_unit().min(0.999_999) * total;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            acc += p;
            if u < acc {
                return Ok(i);
            }
        }
    }
    // 浮点边界兜底：最后一个正概率 token
    probs
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &p)| p > 0.0)
        .map(|(i, _)| i)
        .ok_or(LaunchError::Fatal)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn rng_deterministic_and_uniform_bounds() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
            // 两个流各自消费相同数量的状态推进（next_f32_unit 也推进一次）
            let (ua, ub) = (a.next_f32_unit(), b.next_f32_unit());
            assert_eq!(ua, ub);
            assert!((0.0..1.0).contains(&ua));
        }
    }

    #[test]
    fn argmax_respects_mask() {
        let logits = [0.1f32, 2.0, 1.0, 3.0];
        let mask = [true, false, true, true];
        assert_eq!(sample_argmax(&logits, &mask).unwrap(), 3);
        // 全 masked → 错误
        assert!(sample_argmax(&logits, &[false; 4]).is_err());
    }

    #[test]
    fn prob_sampling_matches_cumulative_and_deterministic() {
        // 确定性：同种子同概率 → 同 token
        let probs = [0.25f32, 0.25, 0.25, 0.25];
        let mut r1 = SplitMix64::new(7);
        let mut r2 = SplitMix64::new(7);
        let tok1 = sample_from_probs(&probs, &mut r1).unwrap();
        let tok2 = sample_from_probs(&probs, &mut r2).unwrap();
        assert_eq!(tok1, tok2);
        assert!(tok1 < 4);

        // 全部集中在 token 2 —— 多次采样必为 2
        let spikes = [0.0f32, 0.0, 1.0, 0.0];
        for seed in 1..20u64 {
            let mut r = SplitMix64::new(seed);
            assert_eq!(sample_from_probs(&spikes, &mut r).unwrap(), 2);
        }

        // 全 0 → 错误
        assert!(sample_from_probs(&[0.0, 0.0], &mut SplitMix64::new(1)).is_err());
    }

    #[test]
    fn all_masked_message_nonempty() {
        assert!(!all_masked_message().is_empty());
    }
}
