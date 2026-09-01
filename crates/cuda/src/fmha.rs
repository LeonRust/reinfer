//! 006 T1 Jit(fmha)：batched-prefill FMHA 内核装载与 launch。
//!
//! 与 dequant.rs/decode.rs 同构的 JitCache 管线，差异点：
//! - `.cu` 内 `include` 的 vendored 头（`vendor/fmha/headers/`，见该目录
//!   README.md 与 version.json）在**运行时**读取并作为 `HeaderFile` 内容
//!   入键（内容哈希 → 头树任何变动自动失效缓存；路径不参与键）；
//! - 头以相对路径落盘（`compile.rs` 按 `HeaderFile.path` 建目录），
//!   构建期 `-M` 闭包按相对路径校验；
//! - launch 需要动态 smem（98304 B）+ `cuFuncSetAttribute` opt-in
//!   （`jit::launch_fmha` / `jit::set_max_dynamic_smem`）。
//!
//! 判据（specs/006 T1）：与 003 dense 双 GEMM 参考逐位/漂移按 003 D7 表；
//! 引擎级 prefill_batch vs 逐 token step 同输入 logits 漂移 ≤ D7、贪心
//! 64 token 文本 100% 一致（`tests/fmha_prefill.rs`，真机 #[ignore] 档）。

use crate::buffer::DeviceBuffer;
use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{HeaderFile, JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

/// vendored 头树根（相对 CARGO_MANIFEST_DIR；JIT 编译机必须能解析该路径）。
const VENDOR_HEADERS: &str = "vendor/fmha/headers";

/// 编译期即嵌入 `.cu` 源码；头文件运行时读取（键 = 内容哈希）。
const KERNEL_SRC: &str = include_str!("../kernels/fmha_kernels.cu");

/// 一个 FMHA 内核变体（S1-7 启发式调优）：同一 flash-attn 算法的不同 tile
/// 几何（kBlockM × kBlockN × 128，kNWarps），数值同构（同 math、不同
/// 分块/线程数）。四个形状符号按 `Is_even_MN = seqlen % block_m == 0` ×
/// `Is_even_K = seqlen % 64 == 0` 选择。
#[derive(Debug, Clone)]
pub struct FmhaVariant {
    mn_even_k_even: KernelFn,
    mn_even_k_odd: KernelFn,
    mn_odd_k_even: KernelFn,
    mn_odd_k_odd: KernelFn,
    /// kBlockM：每个 CTA 的 Q 行数（grid 除数 + LSE 取整块）。
    pub block_m: u32,
    /// 每个 CTA 的线程数（kNWarps × 32）。
    pub threads: u32,
    /// 动态 smem 字节（kSmemQ + kSmemK + kSmemV，d=128）。
    pub smem: u32,
}

impl FmhaVariant {
    /// 按 `(Is_even_MN, Is_even_K)` 取符号。
    fn symbol(&self, mn_even: bool, k_even: bool) -> KernelFn {
        match (mn_even, k_even) {
            (true, true) => self.mn_even_k_even,
            (true, false) => self.mn_even_k_odd,
            (false, true) => self.mn_odd_k_even,
            (false, false) => self.mn_odd_k_odd,
        }
    }
}

/// Mirror of `reinfer_fmha::Flash_fwd_params` (fmha_kernels.cu), passed to
/// the kernel by value as a **single** argument.
///
/// Why a whole-struct argument instead of field-by-field kernelParams
/// entries: the driver slots each kernelParams entry into the kernel's
/// param space by its own per-entry rules, which do not reproduce the C++
/// struct layout — for a 73-field struct every field past the first three
/// pointers lands at the wrong offset (observed on the 595.84 driver as
/// garbage seqlen/strides/h and all-zero O/LSE with the q/k/v pointers
/// still correct). Passing one pointer to a byte-exact copy of the struct
/// is the documented usage for by-value struct params.
///
/// Layout contract: `#[repr(C)]` with field-for-field parity — same natural
/// alignment rules as C++ (ptr/i64 @ 8, i32/f32 @ 4, bool/u8 @ 1), so
/// offsets, padding and total size (456 B) coincide. Keep in sync with
/// fmha_kernels.cu; `debug_assert!`ed in launch_batched_prefill.
#[repr(C)]
struct FlashFwdParams {
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    q_batch_stride: i64,
    k_batch_stride: i64,
    v_batch_stride: i64,
    q_row_stride: i64,
    k_row_stride: i64,
    v_row_stride: i64,
    q_head_stride: i64,
    k_head_stride: i64,
    v_head_stride: i64,
    h: i32,
    h_k: i32,
    h_h_k_ratio: i32,
    o_ptr: *mut c_void,
    oaccum_ptr: *mut c_void,
    o_batch_stride: i64,
    o_row_stride: i64,
    o_head_stride: i64,
    p_ptr: *mut c_void,
    softmax_lse_ptr: *mut c_void,
    softmax_lseaccum_ptr: *mut c_void,
    b: i32,
    seqlen_q: i32,
    seqlen_k: i32,
    seqlen_knew: i32,
    d: i32,
    seqlen_q_rounded: i32,
    seqlen_k_rounded: i32,
    d_rounded: i32,
    rotary_dim: i32,
    total_q: i32,
    scale_softmax: f32,
    scale_softmax_log2: f32,
    cu_seqlens_q: *mut c_void,
    cu_seqlens_k: *mut c_void,
    leftpad_k: *mut c_void,
    seqused_k: *mut c_void,
    blockmask: *mut c_void,
    knew_ptr: *mut c_void,
    vnew_ptr: *mut c_void,
    knew_batch_stride: i64,
    vnew_batch_stride: i64,
    knew_row_stride: i64,
    vnew_row_stride: i64,
    knew_head_stride: i64,
    vnew_head_stride: i64,
    rotary_cos_ptr: *mut c_void,
    rotary_sin_ptr: *mut c_void,
    cache_batch_idx: *mut c_void,
    block_table: *mut c_void,
    block_table_batch_stride: i64,
    page_block_size: i32,
    p_dropout: f32,
    p_dropout_in_uint8_t: u8,
    rp_dropout: f32,
    scale_softmax_rp_dropout: f32,
    window_size_left: i32,
    window_size_right: i32,
    softcap: f32,
    philox_seed: u64,
    philox_offset: u64,
    rng_state: *mut c_void,
    is_bf16: bool,
    is_causal: bool,
    is_seqlens_k_cumulative: bool,
    is_rotary_interleaved: bool,
    num_splits: i32,
    alibi_slopes_ptr: *mut c_void,
    alibi_slopes_batch_stride: i64,
    unpadded_lse: bool,
    seqlenq_ngroups_swapped: bool,
}

/// S1-7 cold-context warmup scratch（`warmup_context` 的落点；保活至
/// Drop——warmup launch 是异步的，过早 free 会与排队中的内核竞争）。
#[derive(Debug)]
struct WarmupScratch {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    o: DeviceBuffer,
    lse: DeviceBuffer,
}

/// FMHA 装载单元：多内核变体（各含四形态：Is_even_MN × Is_even_K），
/// 按 launch 形状取核（变体由 `pick` 启发式选择，见下）。
#[derive(Debug)]
pub struct FmhaKernels {
    /// 加载的 cubin（保活——kernel fn 为模块内符号）。
    #[allow(dead_code)]
    lib: JLib,
    /// 变体表（下标 = 变体 id；`pick` 返回其引用）。
    variants: Vec<FmhaVariant>,
    stream: CudaStream,
    arch: String,
    /// 冷上下文 warmup scratch（见 `new()`；`None` = 设备拒绝 v0 时跳过）。
    warmup: Option<WarmupScratch>,
    /// S1-7 preflight state: (geometry + buffer-identity) keys already
    /// primed by a preflight v0 launch on those exact buffers. A fresh-
    /// context first launch on an unprimed buffer set deterministically
    /// corrupts the last Q block (see the preflight comment in
    /// `launch_batched_prefill_variant_smem`); the cure is per BUFFER SET,
    /// not per geometry — two launch sequences with identical geometry but
    /// different q/k/v/o/lse allocations behave independently (cure probe
    /// fmha_engine_config_kernel_probe: a dense-draw cell primes its
    /// geometry, yet a gate-draw cell at the SAME geometry on its OWN
    /// buffers still fails until its own preflight fires). The engine's
    /// `self.fmha` caches one instance for its lifetime and reallocates
    /// per prefill call; per-set priming makes every call's first FMHA
    /// launch safe, and layers within a call share one set (no per-layer
    /// preflight). `RefCell` — the launch path only holds `&self`;
    /// single-threaded use.
    primed: std::cell::RefCell<
        std::collections::HashSet<(u32, u32, u32, u32, u32, usize, usize, usize, usize, usize)>,
    >,
}

/// 读 vendored 头树全部文件（相对路径 → 内容），按字典序。
fn load_vendor_headers(root: &Path) -> Result<Vec<HeaderFile>, LaunchError> {
    let mut out = Vec::new();
    if !root.is_dir() {
        eprintln!("reinfer-cuda: vendored FMHA headers not found at {}", root.display());
        return Err(LaunchError::Fatal);
    }
    let mut entries: Vec<PathBuf> = Vec::new();
    collect_files(root, root, &mut entries)?;
    entries.sort();
    for p in entries {
        let rel = p.strip_prefix(root).map_err(|_| LaunchError::Fatal)?;
        let content = std::fs::read(&p).map_err(|e| {
            eprintln!("reinfer-cuda: cannot read vendored header {}: {e}", p.display());
            LaunchError::Fatal
        })?;
        out.push(HeaderFile { path: rel.to_string_lossy().replace('\\', "/"), content });
    }
    Ok(out)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), LaunchError> {
    let rd = std::fs::read_dir(dir).map_err(|e| {
        eprintln!("reinfer-cuda: vendored header dir scan failed: {e}");
        LaunchError::Fatal
    })?;
    for e in rd {
        let e = e.map_err(|_| LaunchError::Fatal)?;
        let p = e.path();
        if p.is_dir() {
            collect_files(root, &p, out)?;
        } else if p.file_name().is_some() {
            out.push(p);
        }
    }
    Ok(())
}

impl FmhaKernels {
    /// Loader constructor（工具链 → 编译/缓存 → 取核 ×4 → smem opt-in）。
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        // 头树读取路径：编译期注入（option_env!）。运行时 env（std::env::var）
        // 在 cargo test 下被注入、shell 启动的 serve 下没有——正是 BLOCKER-B
        // 根因（测试绿 / serve 失败）。编译期绝对路径要求本地开发源码树存在
        // vendored 头（packaging 期把头树随二进制分发——见 version.json/将来产物）。
        let manifest = option_env!("CARGO_MANIFEST_DIR").ok_or(LaunchError::Fatal)?;
        let headers = load_vendor_headers(&Path::new(&manifest).join(VENDOR_HEADERS))?;
        let src = KernelSource {
            name: "fmha_fwd_kernels",
            src: KERNEL_SRC,
            headers,
            // Upstream flash-attn requires relaxed constexpr (its CMake sets
            // --expt-relaxed-constexpr): flash_fwd_kernel.h uses std::min in
            // device code (n_block_max clamp).
            flags: {
                let mut f = gencode_flags(arch)?;
                f.push("--expt-relaxed-constexpr".to_string());
                f
            },
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(cache_dir)?;
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc))?;
        let bytes = std::fs::read(&cubin_path).map_err(|_| LaunchError::Fatal)?;
        let lib = JLib::from_bytes(bytes)?;
        // Variant table (S1-7): geometry constants mirror fmha_kernels.cu.
        // smem = 2*d*(block_m + 2*block_n) with d=128 (kSmemQ + kSmemK +
        // kSmemV, f16).
        let mut variants = Vec::new();
        for (id, (bm, bn, warps)) in
            [(128u32, 128u32, 4u32), (128, 128, 8), (128, 64, 4), (256, 128, 8)].iter().enumerate()
        {
            let v = format!("fmha_v{id}");
            let sym = |mn: &str, k: &str| -> Result<KernelFn, LaunchError> {
                lib.kernel(&format!("{v}_mn_{mn}_k_{k}"))
            };
            let variant = FmhaVariant {
                mn_even_k_even: sym("even", "even")?,
                mn_even_k_odd: sym("even", "odd")?,
                mn_odd_k_even: sym("odd", "even")?,
                mn_odd_k_odd: sym("odd", "odd")?,
                block_m: *bm,
                threads: *warps * 32,
                // Dynamic smem declared to the driver: the compact-layout
                // formula 2*d*(bm + 2*bn) is the minimum; the swizzled
                // cutlass layouts can round up (observed: the 128x64
                // variant needs the 128x128-class amount), so cap the
                // declaration at the baseline 98304 B unless the compact
                // estimate is larger (v3). Over-declaring is safe (the
                // kernel only touches its own region).
                smem: (2 * 128 * (*bm + 2 * *bn)).max(98304),
            };
            // Opt-in the dynamic-smem ceiling per kernel. A device whose
            // per-SM shared-memory limit can't host this variant's smem
            // (e.g. the 128 KiB v3 on consumer Blackwell) rejects the
            // attribute — drop the variant instead of failing the load
            // (pick()/bench then just see a shorter table).
            let mut ok = true;
            for s in [
                variant.mn_even_k_even,
                variant.mn_even_k_odd,
                variant.mn_odd_k_even,
                variant.mn_odd_k_odd,
            ] {
                if let Err(e) = super::jit::set_max_dynamic_smem(s, variant.smem) {
                    eprintln!(
                        "reinfer-cuda: FMHA variant v{id} ({bm}x{bn}x128, {warps} warps) \
                         needs {} B smem — device rejected opt-in ({e:?}); variant skipped",
                        variant.smem
                    );
                    ok = false;
                    break;
                }
            }
            if ok {
                variants.push(variant);
            }
        }
        // S1-7 driver store-drop guard. The v2 variant (128x64x4w)
        // deterministically fails to write O/LSE for its second-half
        // x-major grid CTAs (blockIdx.x >= gM/2, rows >= 128 at gM=2) —
        // PROVEN a driver-level artifact, not a race: printf traces show
        // every CTA entering AND exiting (full kernel, epilogue included),
        // the SASS epilogue is unconditional STG.E.128 with correct affine
        // addresses (no CTA-discriminating branch), pattern-fill probes
        // leave the second-half rows byte-identical stale, and the same
        // CTAs' printf ring writes land. The drop is tied to the v2's
        // declared-smem (65536 B) vs launch-smem (98304 B) mismatch:
        // v0/v1 declare exactly what they launch and write every block at
        // every grid size (see fmha_last_cta_anatomy). pick_variant
        // therefore selects v1 for the engine; the v2 remains loadable for
        // microbenchmarks only. The scratch warmup + per-buffer-set v0
        // preflight (launch_batched_prefill_variant_smem) stay as cheap
        // insurance (one extra v0 launch per new (shape, address) key;
        // v0/v1 bit-identical math makes the stale-readback mix harmless).
        // This scratch launch gives the context one early 98304-smem launch
        // (~10 µs per FmhaKernels instance); its own launch does NOT
        // qualify as the preflight (wrong buffers), so it runs with the
        // preflight disarmed (see new()). The engine never launches FMHA
        // first (embed/rms/gemm/rope precede it), but standalone
        // FMHA-first callers (tests, microbenches) do. The scratch config
        // mirrors the preflight (heads=8, kv_heads=4).
        let warmup = match variants.first().filter(|v| v.block_m == 128) {
            Some(_) => {
                let dev = stream.device();
                let n = (256usize * 8 * 128) * 2;
                let nk = (256usize * 4 * 128) * 2;
                Some(WarmupScratch {
                    q: DeviceBuffer::alloc(dev, n)?,
                    k: DeviceBuffer::alloc(dev, nk)?,
                    v: DeviceBuffer::alloc(dev, nk)?,
                    o: DeviceBuffer::alloc(dev, n)?,
                    lse: DeviceBuffer::alloc(dev, 8 * 256 * 4)?,
                })
            }
            None => None,
        };
        let this = Self {
            lib,
            variants,
            stream,
            arch: arch.to_string(),
            warmup,
            // The primed-key set starts EMPTY: the constructor's scratch
            // warmup below must not mark its own geometry as primed (a
            // scratch-buffer launch does not prime the caller's buffers —
            // see the preflight comment), so after it the set is cleared
            // and the first real launch at ANY geometry fires its own
            // preflight with the caller's arguments.
            primed: std::cell::RefCell::new(std::collections::HashSet::new()),
        };
        this.warmup_context()?;
        this.primed.borrow_mut().clear();
        Ok(this)
    }

    /// Fire the cold-context warmup launch (async, on our stream — ordered
    /// before every later launch on the same stream). No-op when the device
    /// rejected v0 at load.
    fn warmup_context(&self) -> Result<(), LaunchError> {
        let Some(w) = &self.warmup else { return Ok(()) };
        let dev: u32 = self.stream.device().into();
        self.launch_batched_prefill_variant(
            dev,
            0,
            w.q.as_ptr().cast(),
            w.k.as_ptr().cast(),
            w.v.as_ptr().cast(),
            w.o.as_ptr().cast::<u16>() as *mut u16,
            w.lse.as_ptr().cast::<f32>() as *mut f32,
            256,
            1,
            8,
            4,
            128,
        )
    }

    /// Target architecture (diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Block until the launch stream drains.
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// Pick the variant for a prefill shape (S1-7 heuristics; all variants
    /// are numerically identical — bit-identical on this device, see
    /// fmha_variant_numeric_identity — so the pick only affects speed).
    ///
    /// Measured on RTX 5090 Laptop (sm_120a), causal GQA 16/8 d=128,
    /// median of 24 cudaEvent-timed launches (tests/fmha_heuristics_bench):
    /// ```text
    ///  seq   |  v0 128x128 w4 |  v1 128x128 w8 |  v2 128x64 w4
    ///   255  |          27.8  |          23.6  |        11.3
    ///   256  |          23.6  |          19.2  |         9.4
    ///   512  |          44.1  |          33.8  |        13.5
    ///  1023  |         142.3  |         111.3  |        39.9
    ///  1024  |         121.4  |          89.3  |        31.6
    ///  2047  |         354.8  |         271.2  |        87.0
    ///  2048  |         356.4  |         257.8  |        80.4
    ///  4095  |        1235.7  |        1022.0  |       304.9
    ///  4096  |        1145.7  |         963.1  |       275.6
    /// ```
    /// (µs; v3 256x128 w8 is dropped at load — its 128 KiB smem opt-in
    /// exceeds the GB203 128 KiB/SM limit.)
    ///
    /// v2 (kBlockN=64, same smem budget, same grid) is 2.5-4.4x faster than
    /// the v0 baseline at every measured shape (2047: 4.1x) — but it is
    /// NOT usable on this driver: its second-half x-major grid CTAs never
    /// land their O/LSE stores (see the new() guard note for the full
    /// proof). v1 (8 warps) is ~1.38x faster than the v0 baseline and
    /// writes every block at every grid size, so it is the pick for all
    /// seqlen; the ~3.2x FMHA-per-call gap vs v2 costs only ~2% of the
    /// prefill wall (the GEMM legs dominate). Fall back to v0 only if a
    /// device rejected v1 at load.
    fn pick_variant(&self, seqlen: u32) -> &FmhaVariant {
        let _ = seqlen;
        // v1 = fastest variant whose stores land on every device observed.
        // All of v0..v2 probe the same 98304 B opt-in, so in practice
        // v0..v1 load together; the defensive fallback walks down to v0.
        let idx = self.variants.len().saturating_sub(1).min(1);
        &self.variants[idx]
    }

    /// 所选变体的 `kBlockM`（引擎按它取整 LSE 缓冲；pick 变体变化时联动）。
    pub fn block_m(&self, seqlen: u32) -> u32 {
        self.pick_variant(seqlen).block_m
    }

    /// 全部变体（微基准 / 诊断用）。
    pub fn variants(&self) -> &[FmhaVariant] {
        &self.variants
    }

    /// 诊断：尝试把某变体的动态 smem opt-in 上限设为 `bytes`（微基准用
    /// 来探测设备上限——v3 的 128 KiB 请求被拒后需要实测天花板）。
    pub fn probe_set_max_smem(&self, variant: usize, bytes: u32) -> Result<(), LaunchError> {
        let v = self.variants.get(variant).ok_or(LaunchError::Fatal)?;
        super::jit::set_max_dynamic_smem(v.mn_even_k_even, bytes)
    }

    /// 指定变体 launch（微基准用）；默认路径经 `pick_variant` 转发。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_prefill_variant(
        &self,
        dev: u32,
        variant_idx: usize,
        q: *const u16,
        k: *const u16,
        v: *const u16,
        o: *mut u16,
        lse: *mut f32,
        seqlen: u32,
        batch: u32,
        heads: u32,
        kv_heads: u32,
        d: u32,
    ) -> Result<(), LaunchError> {
        self.launch_batched_prefill_variant_smem(
            dev,
            variant_idx,
            q,
            k,
            v,
            o,
            lse,
            seqlen,
            batch,
            heads,
            kv_heads,
            d,
            None,
        )
    }

    /// `launch_batched_prefill_variant` + 动态 smem 覆盖（S1-7 排障探针：
    /// 验证 v2 冷上下文零输出是否与 98304 B 过度声明相关；`None` = 变体
    /// 声明值）。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_prefill_variant_smem(
        &self,
        dev: u32,
        variant_idx: usize,
        q: *const u16,
        k: *const u16,
        v: *const u16,
        o: *mut u16,
        lse: *mut f32,
        seqlen: u32,
        batch: u32,
        heads: u32,
        kv_heads: u32,
        d: u32,
        smem_override: Option<u32>,
    ) -> Result<(), LaunchError> {
        let variant = self.variants.get(variant_idx).ok_or(LaunchError::Fatal)?;
        // S1-7 preflight. On a fresh CUDA context, the first 98304-smem
        // FMHA launch against a fresh buffer set deterministically
        // corrupts the last Q block: even MN — the whole second Q block of
        // the last CTA (rows 128.. at seqlen 256, rows 64..127 at 128), O
        // stays zero/stale; odd MN — only the first row of the last
        // block's second sub-block (row 128 at 129, row 64 at 65). The
        // kernel logic is exonerated (bit-identical and correct once
        // primed); the proven cure is one v0 launch with the CALLER'S OWN
        // arguments on THE SAME BUFFERS, completed (sync), immediately
        // before the real launch. Evidence (tests/fmha_heuristics_bench.rs
        // fmha_engine_config_kernel_probe, RTX 5090 sm_120a): with a
        // shared instance whose preflight fires once (the old design),
        // only the first geometry/buffer set is correct — every later
        // geometry fails deterministically, all 28 consecutive launches
        // bit-identical; the ctor's scratch-buffer warmup at the same
        // geometry does NOT prime a caller's buffers (same geometry stayed
        // broken without a caller-buffer preflight); with a preflight per
        // (geometry, buffer set) ALL cells pass (seq 65/128/129/256 x
        // 16/8 heads, dense and gate draws, nbad=0, worst drift 9.8e-4 =
        // f16-P quantization). The key therefore binds the geometry AND
        // the five buffer addresses. The recursive call below sees the key
        // already inserted — no loop.
        let key = (
            seqlen,
            batch,
            heads,
            kv_heads,
            d,
            q as usize,
            k as usize,
            v as usize,
            o as usize,
            lse as usize,
        );
        let first = self.primed.borrow_mut().insert(key);
        if first {
            self.launch_batched_prefill_variant_smem(
                dev, 0, q, k, v, o, lse, seqlen, batch, heads, kv_heads, d, None,
            )?;
            self.stream.synchronize()?;
        }
        let mn = seqlen % variant.block_m == 0;
        let kk = seqlen % 64 == 0;
        let kernel = variant.symbol(mn, kk);
        let _guard = CtxGuard::set_current(dev)?;
        debug_assert!(kv_heads > 0 && heads % kv_heads == 0);
        debug_assert!(d == 128);
        // 参数打包：C3 纪律（局部变量取址）；整个 struct 作单一 by-value
        // 参数（kernelParams 逐字段打 73 槽会与 C++ 布局错位，见上）。
        debug_assert!(std::mem::size_of::<FlashFwdParams>() == 456);
        let params = FlashFwdParams {
            q_ptr: q as *mut c_void,
            k_ptr: k as *mut c_void,
            v_ptr: v as *mut c_void,
            // Layout contract: Q/O are [S*B, nqk] (nqk = heads*d); K/V are
            // [S*B, kvk] (kvk = kv_heads*d, GQA) — different row widths, so
            // K/V strides use kvk, not nqk (a heads*d stride there reads
            // K row s as row 2s — scores only coincidentally right for s=0).
            q_batch_stride: (heads as i64) * (d as i64),
            k_batch_stride: (kv_heads as i64) * (d as i64),
            v_batch_stride: (kv_heads as i64) * (d as i64),
            q_row_stride: (batch as i64) * (heads as i64) * (d as i64),
            k_row_stride: (batch as i64) * (kv_heads as i64) * (d as i64),
            v_row_stride: (batch as i64) * (kv_heads as i64) * (d as i64),
            q_head_stride: d as i64,
            k_head_stride: d as i64,
            v_head_stride: d as i64,
            h: heads as i32,
            h_k: kv_heads as i32,
            h_h_k_ratio: (heads / kv_heads) as i32,
            o_ptr: o as *mut c_void,
            oaccum_ptr: std::ptr::null_mut(),
            // O shares the Q layout [S*B, nqk]: batch stride = nqk (heads*d),
            // row stride = B*nqk. (Both previously set to B*nqk — identical
            // for batch=1, so batch>1 mis-slotted O rows: batch b's row s was
            // written to slot (s+b)*B*nqk, racing with other batches' rows.)
            o_batch_stride: (heads as i64) * (d as i64),
            o_row_stride: (batch as i64) * (heads as i64) * (d as i64),
            o_head_stride: d as i64,
            p_ptr: std::ptr::null_mut(),
            softmax_lse_ptr: lse as *mut c_void,
            softmax_lseaccum_ptr: std::ptr::null_mut(),
            b: batch as i32,
            seqlen_q: seqlen as i32,
            seqlen_k: seqlen as i32,
            seqlen_knew: 0,
            d: d as i32,
            seqlen_q_rounded: (seqlen.div_ceil(variant.block_m) * variant.block_m) as i32,
            seqlen_k_rounded: (seqlen.div_ceil(64) * 64) as i32,
            d_rounded: 128,
            rotary_dim: 0,
            total_q: (seqlen * batch) as i32,
            // q is pre-scaled by 1/sqrt(d) by the caller (engine convention),
            // so the kernel must weight scores with exp(x) — but its softmax
            // is exp2f(x * scale_softmax_log2), hence 1/ln2 ≈ 1.4427 to make
            // 2^(x/ln2) == e^x. (With scale 1.0/log2 1.0 the weights are
            // 2^x, off by a factor ln2 in the exponent — invisible only for
            // single-key rows like s=0.)
            scale_softmax: 1.0,
            scale_softmax_log2: 1.4426950408889634,
            cu_seqlens_q: std::ptr::null_mut(),
            cu_seqlens_k: std::ptr::null_mut(),
            leftpad_k: std::ptr::null_mut(),
            seqused_k: std::ptr::null_mut(),
            blockmask: std::ptr::null_mut(),
            knew_ptr: std::ptr::null_mut(),
            vnew_ptr: std::ptr::null_mut(),
            knew_batch_stride: 0,
            vnew_batch_stride: 0,
            knew_row_stride: 0,
            vnew_row_stride: 0,
            knew_head_stride: 0,
            vnew_head_stride: 0,
            rotary_cos_ptr: std::ptr::null_mut(),
            rotary_sin_ptr: std::ptr::null_mut(),
            cache_batch_idx: std::ptr::null_mut(),
            block_table: std::ptr::null_mut(),
            block_table_batch_stride: 0,
            page_block_size: 0,
            p_dropout: 0.0,
            p_dropout_in_uint8_t: 0,
            rp_dropout: 1.0,
            scale_softmax_rp_dropout: 1.0,
            // Right window must be 0: the causal mask (mask.h
            // apply_mask_local) computes col_idx_limit_right = row + 1 +
            // window_size_right — a -1 excludes the diagonal, fully masking
            // row 0 (LSE=+inf, O=0). Left is unused for Is_local=false.
            window_size_left: -1,
            window_size_right: 0,
            softcap: 0.0,
            philox_seed: 0,
            philox_offset: 0,
            rng_state: std::ptr::null_mut(),
            is_bf16: false,
            is_causal: true,
            is_seqlens_k_cumulative: false,
            is_rotary_interleaved: false,
            num_splits: 1,
            alibi_slopes_ptr: std::ptr::null_mut(),
            alibi_slopes_batch_stride: 0,
            unpadded_lse: false,
            seqlenq_ngroups_swapped: false,
        };
        let mut args: [*mut c_void; 1] = [(&params as *const FlashFwdParams) as *mut c_void];
        let smem = smem_override.unwrap_or(variant.smem);
        unsafe {
            super::jit::launch_fmha(
                kernel,
                &self.stream,
                dev,
                seqlen.div_ceil(variant.block_m),
                batch,
                heads,
                variant.threads,
                smem,
                args.as_mut_ptr(),
            )
        }
    }

    /// 按形状选核（变体 + Is_even_MN × Is_even_K）。
    fn pick(&self, seqlen: u32) -> (usize, KernelFn) {
        let v = self.pick_variant(seqlen);
        let mn = seqlen % v.block_m == 0;
        let kk = seqlen % 64 == 0;
        // The selected variant's index (heuristics bookkeeping).
        let idx = self.variants.iter().position(|x| std::ptr::eq(x, v)).unwrap_or(0);
        (idx, v.symbol(mn, kk))
    }

    /// Batched-prefill FMHA（一次 forward 的注意力段）：
    /// `O = softmax(Q Kᵀ / √d) V`，causal，GQA（`kv_heads` 组复用）。
    ///
    /// 布局契约（affine-stride 技巧，无转置）——Q/K/V/O 均为
    /// **连续 [S×B, nqk] f16 行主序**，即 (s, b, h) 元素在
    /// `s*(B*nqk) + b*nqk + h*d`：`row_stride=B*nqk, batch_stride=nqk,
    /// head_stride=d`。内核按 (bidb, bidh) 直接索引，K/V 头复用
    /// `bidh / (nqk/kv_heads)`。
    ///
    /// - `o`：输出 f16 [S×B, nqk]（与 q/k/v 同布局）；
    /// - `lse`：scratch f32 [B×nqk×S]，内核**无条件**写（行内非因果块
    ///   也为 LSE 留位）；引擎可忽略其值；
    /// - `d` 固定 128（编译期 kHeadDim）。
    ///
    /// grid = (ceil(S/block_m), B, nqk)；block = 变体线程数；动态 smem =
    /// 变体字节数（见 `pick_variant`）。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_prefill(
        &self,
        dev: u32,
        q: *const u16,
        k: *const u16,
        v: *const u16,
        o: *mut u16,
        lse: *mut f32,
        seqlen: u32,
        batch: u32,
        heads: u32,
        kv_heads: u32,
        d: u32,
    ) -> Result<(), LaunchError> {
        let (idx, _) = self.pick(seqlen);
        self.launch_batched_prefill_variant(
            dev, idx, q, k, v, o, lse, seqlen, batch, heads, kv_heads, d,
        )
    }
}

/// Batched-prefill companion kernels (006 T1)：逐行 rms/rope/gather 与
/// 批 KV 写——与 dense_kernels.cu 的逐 token 版本逐行同数学（确定性锚：
/// FMHA 路径与逐 token 路径同输入逐位一致）。
#[derive(Debug)]
pub struct PrefillKernels {
    /// 加载的 cubin（保活）。
    #[allow(dead_code)]
    lib: JLib,
    rms_rows: KernelFn,
    rope_rows: KernelFn,
    gather_rows: KernelFn,
    kv_seq: KernelFn,
    cast_split_qkv: KernelFn,
}

impl PrefillKernels {
    /// Loader constructor（无 vendored 头；自包含编译单元）。
    /// launch 侧与 DiffKernels 同风格：流由调用方显式传入。
    pub fn new(arch: &str, cache_dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "prefill_kernels",
            src: include_str!("../kernels/prefill_kernels.cu"),
            headers: vec![],
            flags: gencode_flags(arch)?,
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(cache_dir)?;
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc))?;
        let bytes = std::fs::read(&cubin_path).map_err(|_| LaunchError::Fatal)?;
        let lib = JLib::from_bytes(bytes)?;
        let rms_rows = lib.kernel("rms_norm_rows_f16")?;
        let rope_rows = lib.kernel("rope_neox_rows_f16")?;
        let gather_rows = lib.kernel("gather_rows_f16")?;
        let kv_seq = lib.kernel("kv_write_seq_rows")?;
        let cast_split_qkv = lib.kernel("cast_split_qkv_f16")?;
        Ok(Self { lib, rms_rows, rope_rows, gather_rows, kv_seq, cast_split_qkv })
    }

    /// 逐行 RMSNorm（grid = rows；每行与单行版同数学）。
    pub fn launch_rms_norm_rows(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const u16,
        out: *mut u16,
        w: *const u16,
        rows: u32,
        n: u32,
        eps: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 3] = [rows as i32, n as i32, eps.to_bits() as i32];
        let mut args: [*mut c_void; 6] = [
            (&x as *const *const u16) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&w as *const *const u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
        ];
        unsafe { super::jit::launch_rows(self.rms_rows, stream, dev, rows, 256, args.as_mut_ptr()) }
    }

    /// 批 RoPE（行主序 [seqlen×heads]；pos = 行内 seq）。S1-7: rows-per-CTA
    /// 合并——每 CTA 处理 `256/half` 行（half=64 → 4 行，256 线程全忙）；
    /// 逐元素数学与逐行版逐位一致（见 prefill_kernels.cu 注释）。
    pub fn launch_rope_rows(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut u16,
        half: u32,
        heads: u32,
        seqlen: u32,
        eta: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // rows_per_cta = 256/half fills the block; grid covers all rows.
        let rpc = (256 / half).max(1);
        let nv: [i32; 4] = [half as i32, heads as i32, seqlen as i32, rpc as i32];
        let mut args: [*mut c_void; 6] = [
            (&x as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&eta as *const f32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
        ];
        let rows = seqlen * heads;
        unsafe {
            super::jit::launch_rows(
                self.rope_rows,
                stream,
                dev,
                rows.div_ceil(rpc),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 批 embed 行拷贝（grid = rows；`toks` 为设备侧 token 数组）。
    pub fn launch_gather_rows(
        &self,
        dev: u32,
        stream: &CudaStream,
        src: *const u16,
        dst: *mut u16,
        toks: *const u32,
        rows: u32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [rows as i32, n as i32];
        let mut args: [*mut c_void; 5] = [
            (&src as *const *const u16) as *mut c_void,
            (&dst as *const *mut u16) as *mut c_void,
            (&toks as *const *const u32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        let blocks = (rows as usize * n as usize).div_ceil(256) as u32;
        unsafe {
            super::jit::launch_rows(self.gather_rows, stream, dev, blocks, 256, args.as_mut_ptr())
        }
    }

    /// 批 KV 写（token s → 页 (s/block_len, s%block_len)；与逐 token
    /// kv_write_row 同地址同值——页序 = li*pp + s/32 连续；`page_base`
    /// 为层页基址 li*pp（逐 token 路径显式 phys 的等价物）。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_write_seq(
        &self,
        dev: u32,
        stream: &CudaStream,
        k_rows: *const u16,
        v_rows: *const u16,
        kv: *mut u16,
        seqlen: u32,
        block_len: u32,
        kv_heads: u32,
        d: u32,
        page_base: u32,
        total_pages: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 6] = [
            seqlen as i32,
            block_len as i32,
            kv_heads as i32,
            d as i32,
            page_base as i32,
            total_pages as i32,
        ];
        let mut args: [*mut c_void; 9] = [
            (&k_rows as *const *const u16) as *mut c_void,
            (&v_rows as *const *const u16) as *mut c_void,
            (&kv as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
        ];
        let per_tok = kv_heads * d;
        let blocks_per_tok = per_tok.div_ceil(256);
        unsafe {
            super::jit::launch_rows(
                self.kv_seq,
                stream,
                dev,
                seqlen * blocks_per_tok,
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 融合 QKV cast（S1-7）：fused [s×(nqk+2kvk)] f32 GEMM 输出 → 三个
    /// 连续 per-section f16 缓冲（q [s·nqk]、k/v [s·kvk]）——与分离路径的
    /// cast 产物同布局，逐元素转换逐位相同（同一 truncation 实现），故
    /// 后续所有下游内核（rms_heads/rope/scale/FMHA/kv_write）无布局特判。
    /// grid = (ceil(N/256), s)；block = 256。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_cast_split_qkv(
        &self,
        dev: u32,
        stream: &CudaStream,
        c: *const f32,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        seqlen: u32,
        nqk: u32,
        kvk: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [nqk as i32, kvk as i32];
        let mut args: [*mut c_void; 6] = [
            (&c as *const *const f32) as *mut c_void,
            (&q as *const *mut u16) as *mut c_void,
            (&k as *const *mut u16) as *mut c_void,
            (&v as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        let cols = nqk + 2 * kvk;
        unsafe {
            super::jit::launch_grid(
                self.cast_split_qkv,
                stream,
                dev,
                cols.div_ceil(256),
                seqlen,
                256,
                1,
                args.as_mut_ptr(),
            )
        }
    }
}

/// FMHA 设备缓冲（引擎 prefill_batch 的一次调用内复用）。
///
/// `qkv` 为 [S×B, nqk] f16 行主序（S 行连续 → 内核 affine-stride 直接
/// 消费）；`o` 同布局 f16；`lse` 为 [B, nqk, S] f32 scratch（d=128 隐含
/// 在 nqk = n_heads × 128 中）。
pub struct FmhaBuffers {
    /// [max_seq×max_batch, nqk] f16（Q/K/V 复用同一块）。
    pub qkv: DeviceBuffer,
    /// [max_seq×max_batch, nqk] f16。
    pub o: DeviceBuffer,
    /// [max_batch, nqk, max_seq] f32（LSE 落点）。
    pub lse: DeviceBuffer,
}

impl std::fmt::Debug for FmhaBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FmhaBuffers")
    }
}

impl FmhaBuffers {
    /// 按 (max_seq, max_batch, nqk) 分配（d 由 nqk/n_heads 隐含，不参与尺寸）。
    pub fn alloc(
        dev: reinfer_core::DeviceId,
        max_seq: usize,
        max_batch: usize,
        nqk: usize,
    ) -> Result<Self, LaunchError> {
        let n = max_seq * max_batch * nqk;
        Ok(Self {
            qkv: DeviceBuffer::alloc(dev, n * 2)?,
            o: DeviceBuffer::alloc(dev, n * 2)?,
            lse: DeviceBuffer::alloc(dev, max_batch * nqk * max_seq * 4)?,
        })
    }
}
