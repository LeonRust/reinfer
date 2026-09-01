//! KernelProvider 选择链（012 T7 最小落地；评审 A-H1：trait/select 归属本 crate；
//! 006 T2：TuneDb + select_fmha/select_attn 回退链）。
//!
//! 档位序（012 r1 R1 裁决）：`Vendor > Jit > Native`，CPU 参考**不注册**进
//! 运行时选择链（仅供差分/显式 opt-in）——`select` 不会返回 `CpuRef`，
//! 且置"全不匹配/仅 CpuRef → 明确错误"以保证"nvcc 缺失 → Fatal 不静默降级"
//! 裁决成立。

use crate::error::LaunchError;
use crate::tune::{self, TuneDb};
use reinfer_core::{DType, DeviceId};
use std::collections::HashMap;
use std::sync::Mutex;

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

/// 可用档位集合（006 T2 选择链输入；由后端探测填充——无 GPU/无后端 → 全 false
/// → 选择恒 `JitDense`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderSet {
    /// Vendor 预编译可用（T3 manifest 校验通过）。
    pub vendor: bool,
    /// Jit FMHA 可用（006 T1 编译通过/有设备）。
    pub jit_fmha: bool,
    /// Jit dense/003 可用。
    pub jit_dense: bool,
}

impl ProviderSet {
    /// 全不可用（无 GPU/无后端）。
    pub fn none() -> Self {
        Self::default()
    }

    /// 全部可用（真机）。
    pub fn all() -> Self {
        Self { vendor: true, jit_fmha: true, jit_dense: true }
    }

    /// 是否没有任何可用档。
    pub fn empty(self) -> bool {
        !self.vendor && !self.jit_fmha && !self.jit_dense
    }

    /// 该档位名（tune.json 名）是否可用；未知档位名 → false。
    fn has_tier(self, name: &str) -> bool {
        match name {
            tune::PROVIDER_VENDOR => self.vendor,
            tune::PROVIDER_JIT_FMHA => self.jit_fmha,
            tune::PROVIDER_JIT_DENSE => self.jit_dense,
            _ => false,
        }
    }
}

/// 选择结果（006 D1 链档位：`Vendor > Jit(fmha) > Jit(dense/003)`）。
///
/// **无载荷**：plan 草案的 `JitKey` 载荷由后端（crates/cuda）在解析档位时计算
/// ——kernels 为 safe 层且不得依赖 reinfer-jit（jit 依赖 kernels，环形不可行）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderChoice {
    /// Vendor 预编译（T3 cubins）。
    Vendor,
    /// Jit FMHA（006 T1）。
    JitFmha,
    /// Jit dense/003 回退。
    JitDense,
}

impl ProviderChoice {
    /// tune.json 内稳定档位名（与 TuneDb 记录键一致）。
    pub fn tune_name(self) -> &'static str {
        match self {
            ProviderChoice::Vendor => tune::PROVIDER_VENDOR,
            ProviderChoice::JitFmha => tune::PROVIDER_JIT_FMHA,
            ProviderChoice::JitDense => tune::PROVIDER_JIT_DENSE,
        }
    }
}

/// FMHA（prefill）选择（006 T2）：`Vendor > Jit(fmha) > Jit(dense/003)`，
/// TuneDb 实测数据优先（score 最优）；无任何可用档（无 GPU）→ 恒 `JitDense`。
///
/// `avail`：可用档位集合（plan 接口为 `(cfg, db)`；本实现注入 avail 以满足
/// "无 GPU → 恒 dense" 与 D6 可用性语义）。
pub fn select_fmha(cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
    select_chain("fmha", cfg, db, avail)
}

/// 通用 attention 选择（decode 等非 FMHA 路径；与 `select_fmha` 同链，
/// 调优命名空间独立（`attn`））。
pub fn select_attn(cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
    select_chain("attn", cfg, db, avail)
}

/// 共享回退链（006 D1）。
///
/// 策略：1) 无任何可用档（无 GPU/无后端）→ 恒 `JitDense`（fail-closed：真实
/// launch 由后端报 `Fatal`，不静默降级 CPU）；2) 可用档位中有实测数据 → 取
/// score 最优者（**默认策略：偏好已实测档 > 语义顺序**——未实测的高档位不压制
/// 已实测数据）；3) 均无实测 → 语义顺序取首个可用档。
///
/// **B2 修正（2026-08-31）**：回退档（`jit_dense`）的孤立实测记录不再压制
/// 可用的更高档位。`jit_dense` 是链末回退档——它只会在高档位不可用/失败时
/// 被实测，因此该记录不是与高档位的公平比较；无条件采纳会让高档位**永久
/// 锁死**（选择器永远选回退档 → 高档位永远不会再被实测——B2：s2048 的
/// 942s dense 记录来自 FMHA 装载失败的旧会话，此后 2048 词 prefill 恒走
/// 逐 token 回退，表现为"卡死"）。`jit_dense` 记录仅在以下情况参与竞争：
/// (a) 无更高档位可用（回退档即唯一档），或 (b) 更高可用档位也有实测记录
/// （双方都已实测 → 仍取 score 最优者，保持"已实测 > 语义序"）。
fn select_chain(op: &str, cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
    if avail.empty() {
        return ProviderChoice::JitDense;
    }
    let sk = tune::shape_key(cfg);
    let known = available_names(&avail);
    if let Some(rec) = db.best_in(op, &sk, &known) {
        let fair = rec.provider != tune::PROVIDER_JIT_DENSE
            || db.best_in(op, &sk, &higher_than_dense(&known)).is_some();
        if fair {
            if let Some(c) = choice_of(&rec.provider) {
                return c;
            }
        }
    }
    [ProviderChoice::Vendor, ProviderChoice::JitFmha, ProviderChoice::JitDense]
        .into_iter()
        .find(|c| avail.has_tier(c.tune_name()))
        .unwrap_or(ProviderChoice::JitDense)
}

/// 可用档位名中高于回退档（`jit_dense`）的档位（语义序内的比较集）。
fn higher_than_dense(known: &[&'static str]) -> Vec<&'static str> {
    known.iter().copied().filter(|n| *n != tune::PROVIDER_JIT_DENSE).collect()
}

/// 可用档位的 tune.json 名（按语义序；选择链的数据面）。
fn available_names(avail: &ProviderSet) -> Vec<&'static str> {
    [ProviderChoice::Vendor, ProviderChoice::JitFmha, ProviderChoice::JitDense]
        .into_iter()
        .filter(|c| avail.has_tier(c.tune_name()))
        .map(ProviderChoice::tune_name)
        .collect()
}

/// tune.json 档位名 → 选择档位（未知名 → `None`：foreign 记录被忽略）。
fn choice_of(name: &str) -> Option<ProviderChoice> {
    match name {
        tune::PROVIDER_VENDOR => Some(ProviderChoice::Vendor),
        tune::PROVIDER_JIT_FMHA => Some(ProviderChoice::JitFmha),
        tune::PROVIDER_JIT_DENSE => Some(ProviderChoice::JitDense),
        _ => None,
    }
}

/// 进程内重复选择缓存：同 (op, shape_key) 只计算一次，后续直接返回首测结果。
///
/// 缓存生命周期 = 实例生命周期（引擎进程级）；进程内 db 更新不触发重算
/// （重算需新实例——与"首测慢/二测快"的进程级语义一致）。
#[derive(Debug, Default)]
pub struct SelectionCache {
    map: Mutex<HashMap<String, ProviderChoice>>,
}

impl SelectionCache {
    /// 空缓存。
    pub fn new() -> Self {
        Self::default()
    }

    /// 缓存的 FMHA 选择（键 = `fmha\0<shape_key>`）。
    pub fn select_fmha(&self, cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
        self.chain("fmha", cfg, db, avail)
    }

    /// 缓存的通用 attention 选择。
    pub fn select_attn(&self, cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
        self.chain("attn", cfg, db, avail)
    }

    fn chain(&self, op: &str, cfg: &OpConfig, db: &TuneDb, avail: ProviderSet) -> ProviderChoice {
        let key = format!("{op}\u{0}{}", tune::shape_key(cfg));
        let mut map = self.map.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = map.get(&key) {
            return *c;
        }
        let c = select_chain(op, cfg, db, avail);
        map.insert(key, c);
        c
    }
}

/// 最小调优条目（正式 TuneDb/持久化归 006——此处仅类型骨架；正式记录见
/// `tune::TuneRecord`）。
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

    // ---- 006 T2：TuneDb + select_fmha/select_attn 回退链 ----

    use crate::tune::{PROVIDER_JIT_DENSE, PROVIDER_JIT_FMHA, PROVIDER_VENDOR, TuneDb, shape_key};
    use std::path::PathBuf;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("reinfer-kernels-provider-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn no_gpu_always_dense() {
        let mut db = TuneDb::open_at(tmp_dir("nogpu").join("tune.json"));
        let c = cfg();
        // 即使已有实测数据，无可用档也必须恒 dense
        db.record("fmha", &shape_key(&c), PROVIDER_VENDOR, 1.0);
        assert_eq!(select_fmha(&c, &db, ProviderSet::none()), ProviderChoice::JitDense);
        assert_eq!(select_attn(&c, &db, ProviderSet::none()), ProviderChoice::JitDense);
        let _ = std::fs::remove_dir_all(tmp_dir("nogpu"));
    }

    #[test]
    fn semantic_chain_respects_availability() {
        let db = TuneDb::open_at(tmp_dir("chain").join("tune.json")); // 空库
        let c = cfg();
        let all = ProviderSet::all();
        assert_eq!(select_fmha(&c, &db, all), ProviderChoice::Vendor);
        let no_vendor = ProviderSet { vendor: false, ..all };
        assert_eq!(select_fmha(&c, &db, no_vendor), ProviderChoice::JitFmha);
        let only_dense = ProviderSet { vendor: false, jit_fmha: false, ..all };
        assert_eq!(select_fmha(&c, &db, only_dense), ProviderChoice::JitDense);
        assert_eq!(select_attn(&c, &db, only_dense), ProviderChoice::JitDense);
        let _ = std::fs::remove_dir_all(tmp_dir("chain"));
    }

    #[test]
    fn tuned_data_beats_semantic_order() {
        let mut db = TuneDb::open_at(tmp_dir("data").join("tune.json"));
        let c = cfg();
        let sk = shape_key(&c);
        // 仅 dense 实测（vendor/fmha 未测）→ 回退档孤立实测不压制可用高档
        // （B2 修正：该记录是高档位不可用时测得的，采纳即锁死高档位）
        db.record("fmha", &sk, PROVIDER_JIT_DENSE, 5000.0);
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        // 三档实测 → score 最优
        db.record("fmha", &sk, PROVIDER_VENDOR, 100.0);
        db.record("fmha", &sk, PROVIDER_JIT_FMHA, 50.0);
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::JitFmha);
        // vendor 实测最优但不可用 → 已实测可用者中最优（fmha 50 < dense 5000）
        let no_vendor = ProviderSet { vendor: false, ..ProviderSet::all() };
        assert_eq!(select_fmha(&c, &db, no_vendor), ProviderChoice::JitFmha);
        // 实测档全部不可用 → 可用档中按语义序
        let only_dense = ProviderSet { vendor: false, jit_fmha: false, ..ProviderSet::all() };
        assert_eq!(select_fmha(&c, &db, only_dense), ProviderChoice::JitDense);
        // 新实测更新 → 最优切换
        db.record("fmha", &sk, PROVIDER_VENDOR, 1.0);
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        let _ = std::fs::remove_dir_all(tmp_dir("data"));
    }

    #[test]
    fn b2_fallback_record_does_not_lock_out_available_primary_tier() {
        let dir = tmp_dir("b2");
        let path = dir.join("tune.json");
        let mut db = TuneDb::open_at(&path);
        let c = cfg();
        let sk = shape_key(&c);
        // B2 现场：s2048 仅存 942s dense 记录（FMHA 装载失败旧会话测得），
        // fmha 可用但从未实测 —— 必须走语义链到 fmha，而非逐 token 回退。
        db.record("fmha", &sk, PROVIDER_JIT_DENSE, 941_982_951.0);
        let fmha_avail = ProviderSet { vendor: false, jit_fmha: true, jit_dense: true };
        assert_eq!(select_fmha(&c, &db, fmha_avail), ProviderChoice::JitFmha);
        // fmha 实测后：双方都已实测 → 取 score 最优者
        db.record("fmha", &sk, PROVIDER_JIT_FMHA, 250.0);
        assert_eq!(select_fmha(&c, &db, fmha_avail), ProviderChoice::JitFmha);
        // dense 实测更优 → 已实测档优先（回退档参与公平比较）
        db.record("fmha", &sk, PROVIDER_JIT_DENSE, 50.0);
        assert_eq!(select_fmha(&c, &db, fmha_avail), ProviderChoice::JitDense);
        // 高档位不可用 → 回退档记录即唯一数据 → dense
        let only_dense = ProviderSet { vendor: false, jit_fmha: false, jit_dense: true };
        assert_eq!(select_fmha(&c, &db, only_dense), ProviderChoice::JitDense);
        // 持久化重启后语义一致
        db.save().unwrap();
        let db2 = TuneDb::open_at(&path);
        assert_eq!(select_fmha(&c, &db2, fmha_avail), ProviderChoice::JitDense);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tuning_is_scoped_by_shape_key() {
        let mut db = TuneDb::open_at(tmp_dir("shape").join("tune.json"));
        let c = cfg(); // seq=0, b=1, h=0, f32->f32
        let other = OpConfig { seq: 4096, ..c };
        db.record("fmha", &shape_key(&c), PROVIDER_JIT_FMHA, 1.0);
        // 其他形状无数据 → 语义链；原形状 → 数据
        assert_eq!(select_fmha(&other, &db, ProviderSet::all()), ProviderChoice::Vendor);
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::JitFmha);
        let _ = std::fs::remove_dir_all(tmp_dir("shape"));
    }

    #[test]
    fn first_bench_slow_second_bench_fast() {
        let dir = tmp_dir("wf");
        let path = dir.join("tune.json");
        let mut db = TuneDb::open_at(&path);
        let avail = ProviderSet::all();
        let c = cfg();
        let sk = shape_key(&c);
        // 首测：无实测数据 → 语义链（vendor 优先）
        assert_eq!(select_fmha(&c, &db, avail), ProviderChoice::Vendor);
        // mock bench 计时（首测编译慢/命中快）：dense 5000µs、fmha 250µs
        db.record("fmha", &sk, PROVIDER_JIT_DENSE, 5000.0);
        db.record("fmha", &sk, PROVIDER_JIT_FMHA, 250.0);
        // 二测：实测数据优先 → 快档
        assert_eq!(select_fmha(&c, &db, avail), ProviderChoice::JitFmha);
        db.save().unwrap();
        // 重启进程（新实例重载）→ 仍是快档
        let db2 = TuneDb::open_at(&path);
        assert!(!db2.was_corrupt());
        assert_eq!(select_fmha(&c, &db2, avail), ProviderChoice::JitFmha);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selection_cache_returns_cached_first_choice() {
        let mut db = TuneDb::open_at(tmp_dir("cache").join("tune.json"));
        let c = cfg();
        let sk = shape_key(&c);
        let cache = SelectionCache::new();
        assert_eq!(cache.select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        // db 更新后仍返回首测缓存（进程内缓存语义：同 (op, shape) 只算一次）
        db.record("fmha", &sk, PROVIDER_JIT_FMHA, 1.0);
        assert_eq!(cache.select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        // 新实例（引擎重启）→ 数据优先
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::JitFmha);
        let _ = std::fs::remove_dir_all(tmp_dir("cache"));
    }

    #[test]
    fn attn_and_fmha_tune_namespaces_are_separate() {
        let mut db = TuneDb::open_at(tmp_dir("ns").join("tune.json"));
        let c = cfg();
        db.record("attn", &shape_key(&c), PROVIDER_JIT_DENSE, 1.0);
        // fmha 无数据 → 语义链；attn 仅回退档孤立实测 → 不压制可用高档
        // （B2 修正，同 fmha 语义）
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        assert_eq!(select_attn(&c, &db, ProviderSet::all()), ProviderChoice::Vendor);
        // 回退档即唯一可用档 → 恒 dense（语义链末端）
        let only_dense = ProviderSet { vendor: false, jit_fmha: false, ..ProviderSet::all() };
        assert_eq!(select_attn(&c, &db, only_dense), ProviderChoice::JitDense);
        let _ = std::fs::remove_dir_all(tmp_dir("ns"));
    }

    #[test]
    fn foreign_tune_records_do_not_affect_selection() {
        let mut db = TuneDb::open_at(tmp_dir("foreign").join("tune.json"));
        let c = cfg();
        db.record("fmha", &shape_key(&c), "vendor_evil", 0.0001);
        db.record("fmha", &shape_key(&c), PROVIDER_JIT_FMHA, 100.0);
        assert_eq!(select_fmha(&c, &db, ProviderSet::all()), ProviderChoice::JitFmha);
        let _ = std::fs::remove_dir_all(tmp_dir("foreign"));
    }

    #[test]
    fn tune_names_roundtrip() {
        assert_eq!(ProviderChoice::Vendor.tune_name(), PROVIDER_VENDOR);
        assert_eq!(ProviderChoice::JitFmha.tune_name(), PROVIDER_JIT_FMHA);
        assert_eq!(ProviderChoice::JitDense.tune_name(), PROVIDER_JIT_DENSE);
        assert_eq!(choice_of(PROVIDER_VENDOR), Some(ProviderChoice::Vendor));
        assert_eq!(choice_of(PROVIDER_JIT_FMHA), Some(ProviderChoice::JitFmha));
        assert_eq!(choice_of(PROVIDER_JIT_DENSE), Some(ProviderChoice::JitDense));
        assert_eq!(choice_of("unknown_tier"), None);
    }
}
