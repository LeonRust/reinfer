//! KernelProvider 选择链（012 T7 最小落地；评审 A-H1：trait/select 归属本 crate）。
//!
//! 档位序（012 r1 R1 裁决）：`Vendor > Jit > Native`，CPU 参考**不注册**进
//! 运行时选择链（仅供差分/显式 opt-in）——`select` 不会返回 `CpuRef`，
//! 且置"全不匹配/仅 CpuRef → 明确错误"以保证"nvcc 缺失 → Fatal 不静默降级"
//! 裁决成立。

use crate::error::LaunchError;
use reinfer_core::{DType, DeviceId};

/// 提供者档位（显式 discriminant——排序键与未来持久化不动序）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ProviderTier {
    /// 厂商预编译（FlashInfer/aclnn）。
    Vendor = 0,
    /// 引擎自有源码经现场编译（CUDA nvcc / AscendC bisheng）。
    Jit = 1,
    /// 预置原生符号路径（CubeCL/cub/pTX 快速通道；CUDA 侧保留档位）。
    Native = 2,
    /// CPU 参考实现（纯函数；**仅差分/显式 opt-in，select 不得返回**）。
    CpuRef = 3,
}

impl ProviderTier {
    /// 是否可被运行时选择（CpuRef 例外）。
    pub fn selectable(self) -> bool {
        self != ProviderTier::CpuRef
    }
}

/// 类型化算子配置（最小面：本切片只 pin 选择链依赖的字段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpConfig {
    /// 算子名（`vec_add`/`rms_norm`/... —— 与 KernelSource.name 同源）。
    pub op: &'static str,
    /// 设备索引。
    pub device: DeviceId,
    /// 输入 dtype。
    pub in_dt: DType,
    /// 输出 dtype。
    pub out_dt: DType,
    /// 头维度（norm/rope/softmax 相关；标量算子用 0）。
    pub head_dim: usize,
    /// 批大小。
    pub batch: usize,
    /// 序列长度。
    pub seq: usize,
}

/// launch 参数通道（后端无关 marker；实现方 downcast 到具体 Args 结构——
/// 具体字段随 kernel 宿主在 003 T5/T8 定型）。
pub trait LaunchArgs: std::any::Any {}

/// KernelProvider：一个算子的"谁能跑/按什么优先级/怎么调"。
pub trait KernelProvider {
    /// 档位。
    fn tier(&self) -> ProviderTier;
    /// 本实现是否匹配该 cfg（设备/形状/dtype）。
    fn matches(&self, cfg: &OpConfig) -> bool;
    /// 无调优数据时的确定性优先级（同档内排序）。
    fn base_priority(&self, cfg: &OpConfig) -> i32;
    /// 需要的 workspace 字节数（0 = 不需要）。
    fn workspace_size(&self, cfg: &OpConfig) -> usize;
    /// 执行算子。本 crate 为安全层（`forbid(unsafe_code)`）：方法签名本身安全，
    /// 实现方必须把设备/FFI `unsafe` 收敛在 impl 内部并按契约保证不变量——
    /// `args` 与 `cfg` 的形状/dtype/设备上下文一致（实现在 FFI 宿主 crate）。
    fn launch(&self, cfg: &OpConfig, args: &mut dyn LaunchArgs) -> Result<(), LaunchError>;
}

/// 确定性选择：按 tier 升序（Vendor < Jit < Native < CpuRef）-> 取最小 tier，
/// 同档按 base_priority 降序。**CpuRef 与其他 provider 混含时被排除**；
/// 全不匹配或仅 CpuRef → 明确错误（fail-closed，非 panic）。
pub fn select<'a>(
    providers: &[&'a dyn KernelProvider],
    cfg: &OpConfig,
) -> Result<&'a dyn KernelProvider, LaunchError> {
    providers
        .iter()
        .copied()
        .filter(|p| p.matches(cfg) && p.tier().selectable())
        .min_by_key(|p| (p.tier(), std::cmp::Reverse(p.base_priority(cfg))))
        .ok_or(LaunchError::Fatal)
}

/// 最小调优条目（正式 TuneDb/持久化归 006——此处仅类型骨架）。
#[derive(Debug, Clone, PartialEq)]
pub struct TuneEntry {
    /// 算子。
    pub op: &'static str,
    /// 架构（`sm_120a`...）。
    pub arch: String,
    /// 形状哈希（确定性字符串，如 `"h128x_b32"` 编码）。
    pub shape: String,
    /// 测量耗时（微秒；无测量语义时不写）。
    pub us: f64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cfg() -> OpConfig {
        OpConfig {
            op: "vec_add",
            device: DeviceId::new(0),
            in_dt: DType::F32,
            out_dt: DType::F32,
            head_dim: 0,
            batch: 1,
            seq: 0,
        }
    }

    struct Fake(ProviderTier, i32);
    static N: AtomicUsize = AtomicUsize::new(0);
    impl KernelProvider for Fake {
        fn tier(&self) -> ProviderTier {
            self.0
        }
        fn matches(&self, c: &OpConfig) -> bool {
            c.op == "vec_add"
        }
        fn base_priority(&self, _c: &OpConfig) -> i32 {
            self.1
        }
        fn workspace_size(&self, _c: &OpConfig) -> usize {
            0
        }
        fn launch(&self, _c: &OpConfig, _a: &mut dyn LaunchArgs) -> Result<(), LaunchError> {
            // 测试桩：不触碰资源（安全层准则下无 unsafe）
            N.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    impl LaunchArgs for u8 {}

    #[test]
    fn tier_order_wins_over_priority() {
        let native = Fake(ProviderTier::Native, 100);
        let jit = Fake(ProviderTier::Jit, 1);
        let picked = select(&[&native, &jit], &cfg()).unwrap();
        assert_eq!(picked.tier(), ProviderTier::Jit);
        // 同档内按优先级
        let l = Fake(ProviderTier::Jit, 5);
        let h = Fake(ProviderTier::Jit, 9);
        assert_eq!(select(&[&l, &h], &cfg()).unwrap().base_priority(&cfg()), 9);
    }

    #[test]
    fn non_matching_provider_is_not_picked() {
        let vendor = Fake(ProviderTier::Vendor, 0);
        let c = OpConfig { op: "other", ..cfg() };
        assert!(select(&[&vendor], &c).is_err());
    }

    #[test]
    fn emptylist_and_cpuref_only_are_errors() {
        assert!(select(&[], &cfg()).is_err());
        let cpu = Fake(ProviderTier::CpuRef, 0);
        let picked = select(&[&cpu], &cfg());
        assert!(picked.is_err(), "CpuRef 不得被运行时选择");
        // 混含时 CpuRef 被排除，Jit 胜出
        let jit = Fake(ProviderTier::Jit, 1);
        assert_eq!(select(&[&cpu, &jit], &cfg()).unwrap().tier(), ProviderTier::Jit);
    }

    #[test]
    fn tune_entry_fields() {
        let t =
            TuneEntry { op: "vec_add", arch: "sm_120a".into(), shape: "h0x_b1".into(), us: 1.5 };
        assert_eq!(t.op, "vec_add");
        assert!(t.us > 0.0);
    }
}
