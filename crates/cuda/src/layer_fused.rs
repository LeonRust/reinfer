//! S1-10: decode-step deep fusion — the whole layer in ONE kernel launch.
//!
//! The S1-9 fused step still runs 8 kernels per layer (229 nodes/step for
//! Qwen3-0.6B). Every kernel boundary pays the launch-gap + grid-ramp
//! overhead, and the small stages (p2_qkv: 8 blocks; p2_o: 1 block) leave
//! the device mostly idle. S1-10 merges the eight kernels into one
//! persistent kernel per layer (kernels/decode_layer_fused_kernels.cu): a
//! fixed grid of co-resident blocks runs the eight stages in sequence,
//! separated by device-side grid barriers (arrive/spin on a generation
//! counter — plain cuLaunchKernel, no -rdc, graph-capture friendly). Per
//! step: 28 layer kernels + the lm_head phase pair = 30 nodes.
//!
//! Stage 0 (gather + attn rms) of the S1-9 sequence is folded into the
//! layer-0 kernel (redundant idempotent compute across blocks, byte-
//! identical to gather_row + rms_norm_row_f16) — the step drops the two
//! separate launches entirely. The lm_head stays its own phase pair (it
//! runs after the last layer's p2_down and is bandwidth-bound on its own).
//!
//! Bit-level contract: every stage preserves the exact per-column /
//! per-tile arithmetic and accumulation orders of the S1-9 kernels it
//! replaces (see the kernel header for the per-stage contracts). The
//! mechanical differences are: 512-thread blocks (a 512-col tile replaces
//! two 256-col tiles — per-(col, slab) values are computed by exactly one
//! thread with the same k-walk), grid-strided stage tiles (tile t to block
//! t % G deterministically), the add_rms stages using threads 0..255 with
//! the rest idle at the same syncs, head-norm guards scaled to 512/d heads
//! per block, and the 512-col gate/up stripes (build gate below).
//!
//! Co-residency gate: the grid barriers spin and deadlock unless every
//! block is co-resident. The loader queries
//! cuOccupancyMaxActiveBlocksPerMultiprocessor (with the launch's dynamic
//! smem) and launches grid = min(stage tile max, occupancy * SM count);
//! the stages grid-stride so any co-resident grid is correct. A build
//! failure fails open: the engine keeps the S1-9 fused path (and the
//! split path behind it).

use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
use crate::fused::{FusedGeom, PlanRow};
use crate::jit::{CtxGuard, JLib, KernelFn, launch_fmha};
use crate::stream::CudaStream;
use cudarc::driver::sys as dsys;
use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::PathBuf;

/// S1-11 (specs/017): the block-width switch (default **2** — the widened
/// stages' 512-col tiles are split into 256-col tile PAIRS, one per block;
/// `REINFER_FUSED_BW=1`/off words keep the S1-10 single-width kernel
/// byte-for-byte). Widths above 2 (the plan's W=4) are not implemented
/// yet — they clamp to the shipped width 2.
const BW_ENV: &str = "REINFER_FUSED_BW";

/// 017-d: the per-thread column width of the W=2 kernel's phase-1 stages
/// (`REINFER_FUSED_WC=2`/`4`; default 1 keeps the S1-11 kernel verbatim —
/// each thread then owns WC consecutive columns fetched as one vector
/// load, see decode_layer_fused_kernels.cu). Only meaningful with block
/// width 2; W=1 forces WC=1.
const WC_ENV: &str = "REINFER_FUSED_WC";

/// 018 P1a (specs/018-decode-pipeline): the two-tree pipe switch
/// (`REINFER_FUSED_PIPE=1` — the W=2/WC=1 kernel with the A/B barrier
/// trees and the P_edge_add_o / P_edge_down data edges). Default 0 = the
/// S1-11 kernel byte-identical (the pipe entry is only instantiated at
/// W=2/WC=1, so other width combinations silently keep the S1-11
/// kernel). Orthogonal to REINFER_FUSED_BW/WC.
const PIPE_ENV: &str = "REINFER_FUSED_PIPE";

/// REINFER_FUSED_BW parsing: unset -> **2** (default on); off words mirror
/// the opt-out convention (like REINFER_FUSED) and select width 1.
#[must_use]
fn block_width_from_env(value: Option<&str>) -> u32 {
    match value {
        None => 2,
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "0" | "off" | "false" | "no" => 1,
            _ => 2,
        },
    }
}

/// REINFER_FUSED_WC parsing: unset -> 1; only 2 and 4 are implemented.
#[must_use]
fn col_width_from_env(value: Option<&str>) -> u32 {
    match value {
        None => 1,
        Some(v) => match v.trim() {
            "2" => 2,
            "4" => 4,
            _ => 1,
        },
    }
}

/// REINFER_FUSED_PIPE parsing: unset -> off; "1"/on-words -> the two-tree
/// pipe entry (specs/018 P1a). The reserved P2a/P3a values (2/3) are not
/// implemented — they stay off.
#[must_use]
fn pipe_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" => true,
            _ => false,
        },
    }
}

/// The kernel's tw-col tile mapping, derived from the plan shapes
/// (tw = 512 for the plain stages, 256 for the S1-11 widened stages).
fn tiles_w(n: usize, nslabs: u32, tw: u32) -> u32 {
    n.div_ceil(tw as usize) as u32 * nslabs
}

/// The kernel's 512-col tile mapping (the plain stages — o, down, and all
/// of W=1), derived from the plan shapes.
fn tiles512(n: usize, nslabs: u32) -> u32 {
    tiles_w(n, nslabs, 512)
}

/// Host mirror of the kernel-side `LayerFusedConst` (repr(C) — the bytes
/// are uploaded verbatim). Pointer fields first, then the value block,
/// exactly as declared in decode_layer_fused_kernels.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LayerFusedConst {
    /// gather source (embed row table).
    embed: *const u16,
    /// Residual stream (mutated in place).
    x: *mut u16,
    /// Norm output (the next plan group's layer input).
    xn: *mut u16,
    /// q projection output (f16).
    q16: *mut u16,
    /// k projection output (f16).
    k16: *mut u16,
    /// v projection output (f16).
    v16: *mut u16,
    /// Flash attention output (o-projection input).
    attn: *mut u16,
    /// KV cache base (K region then V region).
    kv: *mut u16,
    /// [B] kv lengths (per-step upload).
    lens: *const u32,
    /// Identity page table base (li*pp offset inside the kernel).
    pages: *const u32,
    /// Layer-0 attn_norm (the folded rms0 stage).
    wnorm0: *const u16,
    /// Per-(layer, stage) clock64 slots [n_layers * 9] (block 0; null when
    /// profiling is off).
    stage_ts: *mut u32,
    /// Hidden size.
    h: i32,
    /// q projection columns (q_heads * d).
    nqk: i32,
    /// k/v projection columns (kv_heads * d).
    kvk: i32,
    /// Head dim.
    d: i32,
    /// d / 2 (rope pair split).
    half: i32,
    /// FFN hidden size.
    ffn: i32,
    /// q plan slab count (ascending-slab phase-2 sums).
    nslabs_q: i32,
    /// k plan slab count.
    nslabs_k: i32,
    /// v plan slab count.
    nslabs_v: i32,
    /// o plan slab count.
    nslabs_o: i32,
    /// gate plan slab count (== up).
    nslabs_g: i32,
    /// down plan slab count.
    nslabs_d: i32,
    /// p1_qkv stage tiles (sum over the q/k/v rows).
    tiles_qkv: i32,
    /// p1_o stage tiles.
    tiles_o: i32,
    /// p1_gu stage tiles.
    tiles_gu: i32,
    /// p2_gu_d stage tiles (the down plan's tile grid).
    tiles_gu_d: i32,
    /// q head count.
    q_heads: i32,
    /// kv head count.
    kv_heads: i32,
    /// q_heads / kv_heads (GQA ratio).
    ratio: i32,
    /// KV cache page length.
    block_len: i32,
    /// Token-capacity bound (pp * block_len — dynamic smem sizing).
    max_kv: i32,
    /// Total physical pages over all layers.
    total_pages: i32,
    /// Pages per layer.
    pp: i32,
    /// RoPE theta.
    eta: f32,
    /// 1 / sqrt(d) (q rope scale).
    scale_q: f32,
    /// RMSNorm epsilon.
    eps: f32,
    /// head-norm on (q_norm/k_norm present).
    hn: i32,
    /// 018 P1a pipe sets (only the pipe entry reads them; Qwen3-0.6B
    /// W=2: 144/96).
    /// bar0 participants (the p1_qkv producer prefix).
    pipe_pa0: i32,
    /// B-group prefix (edge + B-tree participants).
    pipe_pb: i32,
}

/// Per-layer geometry + buffer pointers the const is built from. Plain
/// data so both the engine and the kernel-level tests can construct it
/// (the test fabricates its own buffer set over fabricated plans).
#[derive(Debug, Clone, Copy)]
pub struct LayerFusedSpec {
    /// Hidden size.
    pub h: usize,
    /// q projection columns.
    pub nqk: usize,
    /// k/v projection columns.
    pub kvk: usize,
    /// FFN hidden size.
    pub ffn: usize,
    /// Head dim (gates: 512 % d == 0, 64 <= d <= 256, even).
    pub d: usize,
    /// q head count.
    pub q_heads: usize,
    /// kv head count.
    pub kv_heads: usize,
    /// KV cache page length.
    pub block_len: usize,
    /// Token-capacity bound (pp * BLOCK_LEN) — dynamic smem sizing.
    pub max_kv: usize,
    /// Total physical pages over all layers.
    pub total_pages: usize,
    /// Pages per layer.
    pub pp: usize,
    /// RoPE theta.
    pub eta: f32,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Whether q/k head-norms are present.
    pub head_norm: bool,
    /// gather source (embed row table).
    pub embed: *const u16,
    /// Residual stream (mutated in place).
    pub x: *mut u16,
    /// Norm output (the next plan group's layer input).
    pub xn: *mut u16,
    /// q projection output (f16).
    pub q16: *mut u16,
    /// k projection output (f16).
    pub k16: *mut u16,
    /// v projection output (f16).
    pub v16: *mut u16,
    /// Flash attention output (o-projection input).
    pub attn: *mut u16,
    /// KV cache base (K region then V region).
    pub kv: *mut u16,
    /// [B] kv lengths (per-step upload).
    pub lens: *const u32,
    /// Identity page table base (li*pp offset inside the kernel).
    pub pages: *const u32,
    /// Layer-0 attn_norm [h] (the folded rms0 stage).
    pub wnorm0: *const u16,
}

/// The S1-10 layer-fused unit: the persistent kernel + its stable
/// argument buffers (the const blob, the barrier slots, the per-stage
/// clock64 timestamps) + the co-residency grid.
///
/// S1-11 (specs/017): two kernel entries are loaded — the W=1 S1-10
/// kernel (512-col stage tiles) and the W=2 `decode_step_layer_fused_bw2`
/// (256-col widened-stage tiles, __launch_bounds__(512, 2) so two blocks
/// fit per SM). `build` picks the width from REINFER_FUSED_BW and the
/// occupancy gate; `launch`/`kernel_name`/`grid` follow the selection.
///
/// 017-d: the W=2 phase-1 stages additionally widen per-thread columns
/// (`REINFER_FUSED_WC`): WC=2/4 entries `decode_step_layer_fused_bw2_wc2`
/// /`..._wc4` load each thread's WC consecutive columns as one vector
/// load. WC=1 is the S1-11 entry verbatim.
///
/// 018 P1a (specs/018-decode-pipeline): a fifth entry
/// `decode_step_layer_fused_bw2_pipe` (selected by `REINFER_FUSED_PIPE=1`
/// at W=2/WC=1) runs the same stage program with the barriers split into
/// two trees (A: gather/qkv/flash/o-p1; B: add_rms(o)/gu/down) plus the
/// P_edge_add_o / P_edge_down data edges — same arithmetic, different
/// participant sets (uploaded in the const as pipe_pa0/pipe_pb).
#[derive(Debug)]
pub struct LayerFusedKernels {
    lib: JLib,
    /// The selected kernel entry (per the width + column-width gates).
    kernel: KernelFn,
    /// The W=2 entry (`decode_step_layer_fused_bw2`).
    kernel_w2: KernelFn,
    /// 017-d: the W=2 WC=2/4 entries.
    kernel_w2_wc2: KernelFn,
    kernel_w2_wc4: KernelFn,
    /// 018 P1a: the two-tree pipe entry (`decode_step_layer_fused_bw2_pipe`).
    kernel_w2_pipe: KernelFn,
    /// Selected block width (1 = S1-10 behavior, 2 = S1-11 widened).
    block_width: u32,
    /// Selected per-thread column width of the phase-1 stages (1/2/4;
    /// 1 when block width 1).
    col_width: u32,
    /// 018 P1a: the two-tree pipe mode is on (W=2/WC=1 pipe entry).
    pipe: bool,
    dev: u32,
    /// The uploaded `LayerFusedConst` blob (stable).
    cbuf: DeviceBuffer,
    /// 20 u32 barrier slots (10 counters + 10 generations), zeroed once at
    /// build — every barrier self-resets, so the state is clean across
    /// launches and graph replays.
    bar: DeviceBuffer,
    /// Per-(layer, stage) clock64 marks [n_layers * 9] (block 0 only).
    stage_ts: DeviceBuffer,
    /// Co-resident grid size (occupancy-gated).
    grid: u32,
    /// Dynamic shared memory of the launch ((d + max_kv) * 4).
    shared: u32,
    /// Stage-clock aggregation window for `profile_accumulate`.
    ts_acc: [f64; 9],
    ts_steps: u32,
}

impl LayerFusedKernels {
    /// Load the layer-fused kernel (standard JitCache pipeline).
    pub fn new(dev: u32, arch: &str, cache_dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "decode_layer_fused",
            src: include_str!("../kernels/decode_layer_fused_kernels.cu"),
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
        let kernel = lib.kernel("decode_step_layer_fused")?;
        let kernel_w2 = lib.kernel("decode_step_layer_fused_bw2")?;
        let kernel_w2_wc2 = lib.kernel("decode_step_layer_fused_bw2_wc2")?;
        let kernel_w2_wc4 = lib.kernel("decode_step_layer_fused_bw2_wc4")?;
        let kernel_w2_pipe = lib.kernel("decode_step_layer_fused_bw2_pipe")?;
        Ok(Self {
            lib,
            kernel,
            kernel_w2,
            kernel_w2_wc2,
            kernel_w2_wc4,
            kernel_w2_pipe,
            block_width: 1,
            col_width: 1,
            pipe: false,
            dev,
            cbuf: DeviceBuffer::alloc(DeviceId::new(dev), 4)?, // placeholder
            bar: DeviceBuffer::alloc(DeviceId::new(dev), 4)?,  // placeholder
            stage_ts: DeviceBuffer::alloc(DeviceId::new(dev), 4)?, // placeholder
            grid: 1,
            shared: 0,
            ts_acc: [0.0; 9],
            ts_steps: 0,
        })
    }

    /// Build the const blob, the barrier buffer, the stage-clock buffer
    /// and the co-residency grid from `spec` over the fused geometry `g`
    /// (the S1-9 plan table + shared partials segments — the layer kernel
    /// reads the SAME plan rows and partials the S1-9 fused path uses).
    ///
    /// Merge-validity gate (the 512-col stripe version of the S1-9 gate):
    /// stage 7's block redundantly writes the 512-col stripe its phase-2
    /// k-range lies in — every down slab's k-range must lie inside
    /// [(s/per_stripe)*512, +512) with per_stripe = 512/slab_k_d (slab_k
    /// in {512, 256, 128, 64}), and the stripes must cover the whole down
    /// k-range. Configs that violate this (or any other gate) fail the
    /// build (Fatal) — the engine falls back to the S1-9 fused path.
    pub fn build(
        &mut self,
        dev: DeviceId,
        g: &FusedGeom,
        spec: &LayerFusedSpec,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev.index())?;
        // 512-thread geometry gates: the p2_qkv stage runs 512/d heads per
        // block over 128-slot smem regions (s_sh[1024] -> d >= 64), the
        // flash stage needs the decode_flash_kernels.cu contracts
        // (d <= 256, d even), the add_rms stages keep the n <= 1024 bound.
        if 512 % spec.d != 0 || spec.d < 64 || spec.d > 256 || spec.d % 2 != 0 {
            return Err(LaunchError::Fatal);
        }
        if spec.h > 1024 || spec.max_kv == 0 {
            return Err(LaunchError::Fatal);
        }
        // Stage-7 merge gate (see the doc above; mirrors the kernel's c1
        // mapping with 512-col stripes).
        let slab_k_d = (spec.ffn as u32).div_ceil(g.nslabs_d as u32);
        let mut merged_ok = slab_k_d <= 512 && 512 % slab_k_d == 0;
        if merged_ok {
            let per_stripe = 512 / slab_k_d;
            if (g.nslabs_d as u32 / per_stripe) * 512 >= spec.ffn as u32 {
                for s in 0..g.nslabs_d {
                    let s = s as u64;
                    let ks = s * slab_k_d as u64;
                    let ke = (ks + slab_k_d as u64).min(spec.ffn as u64);
                    let lo = (s / per_stripe as u64) * 512;
                    if ks < lo || ke > lo + 512 {
                        merged_ok = false;
                        break;
                    }
                }
            } else {
                merged_ok = false;
            }
        }
        if !merged_ok {
            return Err(LaunchError::Fatal);
        }

        // S1-11 (specs/017): block width from REINFER_FUSED_BW (default
        // 2). The widened stages (p1_qkv, p2_qkv, p1_gu) split their
        // 512-col tiles into 512/W-col tiles (W=2 -> 256), one adjacent
        // PAIR per block; the plain stages (o, down) and all of W=1 keep
        // the 512-col tiles. The const's tile counts feed the kernel's
        // stage loops AND the barrier participant sets (P1/P4/P5), so they
        // must match the entry's tiles — the counts are computed only
        // after the width is settled (the fallback below).
        let mut w = block_width_from_env(std::env::var(BW_ENV).ok().as_deref());
        // 017-d: per-thread column width from REINFER_FUSED_WC (default
        // 1 = the S1-11 kernel verbatim; 2/4 widen the W=2 phase-1 b
        // loads). Only the W=2 kernel has widened entries — W=1 forces
        // WC=1.
        let mut wc =
            if w == 2 { col_width_from_env(std::env::var(WC_ENV).ok().as_deref()) } else { 1 };
        // 018 P1a: the two-tree pipe mode (REINFER_FUSED_PIPE=1). Only
        // the W=2/WC=1 entry has a pipe instantiation (specs/018 P1a is
        // measured at the default width) — any other combination keeps
        // the S1-11 kernel (the switch stays orthogonal to BW/WC).
        let mut pipe = pipe_from_env(std::env::var(PIPE_ENV).ok().as_deref());
        if pipe && (w != 2 || wc != 1) {
            eprintln!(
                "reinfer-cuda: REINFER_FUSED_PIPE needs block width 2 / column width 1 \
                 (got {w}/{wc}) — pipe off (S1-11 kernel)"
            );
            pipe = false;
        }
        // Co-residency gate: the exact resident block count for the real
        // launch geometry (512 threads, the launch's dynamic smem), times
        // the SM count. The grid is capped at max_tiles (the largest
        // stage's tile count — extra blocks would idle everywhere).
        let shared = ((spec.d + spec.max_kv) * 4) as u32;
        // sm_120 (consumer Blackwell) per-block dynamic smem opt-in cap:
        // 101376 B. The flash stage stages (d + kv_len) floats; with a
        // context beyond ~24K tokens the request cannot fit — fail open
        // to the S1-9 fused path (its flash kernel caps at 98304 B and
        // never stages the whole kv length). Query the real cap so the
        // gate tracks the device, not a hard-coded number.
        let mut smem_cap: i32 = 0;
        let r = unsafe {
            dsys::cuDeviceGetAttribute(
                &mut smem_cap,
                dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev.index() as dsys::CUdevice,
            )
        };
        if r != dsys::CUresult::CUDA_SUCCESS || smem_cap <= 0 {
            return Err(LaunchError::Fatal);
        }
        if shared > smem_cap as u32 {
            return Err(LaunchError::Fatal); // ctx too long for the layer kernel
        }
        // >48KB opt-in: the launch smem exceeds the default per-block
        // limit (49152) — declare the cap on all kernel handles once
        // (they share the launch's dynamic smem).
        if shared > 49152 {
            super::jit::set_max_dynamic_smem(self.kernel, shared)?;
            super::jit::set_max_dynamic_smem(self.kernel_w2, shared)?;
            super::jit::set_max_dynamic_smem(self.kernel_w2_wc2, shared)?;
            super::jit::set_max_dynamic_smem(self.kernel_w2_wc4, shared)?;
            super::jit::set_max_dynamic_smem(self.kernel_w2_pipe, shared)?;
        }
        // W=2 needs TWO co-resident blocks per SM (its
        // __launch_bounds__(512, 2) caps the registers at 64/thread);
        // when the occupancy query cannot fit that (smem/register
        // pressure), the width falls back to W=1 — the S1-10 behavior
        // verbatim (REINFER_FUSED_BW=off keeps it). The 017-d column
        // width falls back WC -> 1 first (same gate), then width.
        let entry = |w: u32, wc: u32, pipe: bool| -> KernelFn {
            match (w, wc) {
                (1, _) => self.kernel,
                (2, 1) => {
                    if pipe {
                        self.kernel_w2_pipe
                    } else {
                        self.kernel_w2
                    }
                }
                (2, 2) => self.kernel_w2_wc2,
                (2, 4) => self.kernel_w2_wc4,
                _ => self.kernel, // unreachable (env parsing clamps)
            }
        };
        let occ_of = |kern: KernelFn| -> Result<i32, LaunchError> {
            let mut occ: i32 = 0;
            let r = unsafe {
                dsys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut occ,
                    kern.raw(),
                    512,
                    shared as usize,
                )
            };
            if r != dsys::CUresult::CUDA_SUCCESS || occ <= 0 {
                return Err(LaunchError::Fatal);
            }
            Ok(occ)
        };
        let mut occ = occ_of(entry(w, wc, pipe))?;
        if w == 2 && wc > 1 && occ < 2 {
            eprintln!(
                "reinfer-cuda: layer-fused column width {wc} not co-resident (occ {occ}) — \
                 falling back to column width 1"
            );
            wc = 1;
            occ = occ_of(entry(w, 1, false))?;
        }
        if w == 2 && occ < 2 {
            eprintln!(
                "reinfer-cuda: layer-fused block width 2 not co-resident (occ {occ}) — \
                 falling back to width 1"
            );
            w = 1;
            wc = 1;
            pipe = false;
            occ = occ_of(self.kernel)?;
        }
        let mut sms: i32 = 0;
        let r = unsafe {
            dsys::cuDeviceGetAttribute(
                &mut sms,
                dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                dev.index() as dsys::CUdevice,
            )
        };
        if r != dsys::CUresult::CUDA_SUCCESS || sms <= 0 {
            return Err(LaunchError::Fatal);
        }
        // Stage tile counts for the settled width: tw-col for the widened
        // stages (p1_qkv/p2_qkv/p1_gu), 512-col for the plain stages
        // (p1_o, p2_gu_d) — grid-strided over the grid. `tiles_p2` mirrors
        // the kernel's cq + 2*ck (the stage-2 loop bound).
        //
        // 017-d: with WC > 1 the W=2 phase-1 tiles are 256*WC columns wide
        // (tw below) and stage 7's down tiles 512*WC (tw7); phase-2 keeps
        // its own 256-col (W=2) tiling regardless of WC, so `tiles_p2`
        // stays 256-based.
        let tw = if w == 2 { 256 * wc } else { 512 };
        let tw7 = if w == 2 && wc > 1 { 512 * wc } else { 512 };
        let tw_p2 = if w == 2 { 256 } else { 512 };
        let tiles_qkv = tiles_w(spec.nqk, g.nslabs_q as u32, tw)
            + tiles_w(spec.kvk, g.nslabs_k as u32, tw)
            + tiles_w(spec.kvk, g.nslabs_v as u32, tw);
        let tiles_o = tiles512(spec.h, g.nslabs_o as u32);
        let tiles_gu = tiles_w(spec.ffn, g.nslabs_g as u32, tw) * 2;
        let tiles_gu_d = tiles_w(spec.h, g.nslabs_d as u32, tw7);
        let tiles_p2 =
            (spec.nqk.div_ceil(tw_p2 as usize) + 2 * spec.kvk.div_ceil(tw_p2 as usize)) as u32;
        let max_tiles = tiles_qkv
            .max(tiles_p2)
            .max(spec.q_heads as u32)
            .max(tiles_o)
            .max(1)
            .max(tiles_gu)
            .max(tiles_gu_d);
        let grid = (occ * sms).min(max_tiles as i32).max(1) as u32;

        // 018 P1a pipe sets (uploaded in the const; only the pipe entry
        // reads them — the kernel derives nothing, the host owns the
        // sets): pa0 = the p1_qkv producer prefix (ceil(tiles_qkv/2) —
        // one PAIR of tw-col tiles per block at W=2); pb = the B-group
        // prefix, the max over every stage-2..8 producer prefix (the
        // p2_qkv pairs, the flash heads, the p1_o tiles, the p1_gu
        // pairs, the p2_gu_d tiles — the add stripes then repartition
        // over pb). W=2/Qwen3-0.6B: 144/96. Only the W=2 build has a
        // pipe entry; other widths upload 0 (never read).
        let pipe_pa0 = if w == 2 { tiles_qkv.div_ceil(2).min(grid) as i32 } else { 0 };
        let pipe_pb = if w == 2 {
            tiles_gu
                .div_ceil(2)
                .max(tiles_gu_d)
                .max(tiles_o)
                .max(spec.q_heads as u32)
                .max(tiles_p2.div_ceil(2))
                .min(grid) as i32
        } else {
            0
        };

        // Buffers: stage-clock slots (allocated FIRST — the const's
        // stage_ts pointer addresses them), the const blob, the barrier
        // slots (zeroed once; the barriers self-reset).
        let n_layers = g.grid_qkv.len();
        self.stage_ts = DeviceBuffer::alloc(dev, n_layers * 9 * 4)?;
        let hn = if spec.head_norm { 1 } else { 0 };
        let c = LayerFusedConst {
            embed: spec.embed,
            x: spec.x,
            xn: spec.xn,
            q16: spec.q16,
            k16: spec.k16,
            v16: spec.v16,
            attn: spec.attn,
            kv: spec.kv,
            lens: spec.lens,
            pages: spec.pages,
            wnorm0: spec.wnorm0,
            stage_ts: self.stage_ts.as_ptr() as *mut u32,
            h: spec.h as i32,
            nqk: spec.nqk as i32,
            kvk: spec.kvk as i32,
            d: spec.d as i32,
            half: (spec.d / 2) as i32,
            ffn: spec.ffn as i32,
            nslabs_q: g.nslabs_q,
            nslabs_k: g.nslabs_k,
            nslabs_v: g.nslabs_v,
            nslabs_o: g.nslabs_o,
            nslabs_g: g.nslabs_g,
            nslabs_d: g.nslabs_d,
            tiles_qkv: tiles_qkv as i32,
            tiles_o: tiles_o as i32,
            tiles_gu: tiles_gu as i32,
            tiles_gu_d: tiles_gu_d as i32,
            q_heads: spec.q_heads as i32,
            kv_heads: spec.kv_heads as i32,
            ratio: (spec.q_heads / spec.kv_heads) as i32,
            block_len: spec.block_len as i32,
            max_kv: spec.max_kv as i32,
            total_pages: spec.total_pages as i32,
            pp: spec.pp as i32,
            eta: spec.eta,
            scale_q: 1.0 / (spec.d as f32).sqrt(),
            eps: spec.eps,
            hn,
            pipe_pa0,
            pipe_pb,
        };
        self.cbuf = DeviceBuffer::alloc(dev, std::mem::size_of::<LayerFusedConst>())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &c as *const LayerFusedConst as *const u8,
                std::mem::size_of::<LayerFusedConst>(),
            )
        };
        let hb = HostBuffer::alloc(bytes.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), hb.as_ptr() as *mut u8, bytes.len());
        }
        copy(&mut MemRef::Device(&self.cbuf), &MemRef::Host(&hb), bytes.len(), None)?;

        self.bar = DeviceBuffer::alloc(dev, 20 * 4)?;
        let zeros = HostBuffer::alloc(20 * 4)?;
        unsafe {
            std::ptr::write_bytes(zeros.as_ptr() as *mut u8, 0, 20 * 4);
        }
        copy(&mut MemRef::Device(&self.bar), &MemRef::Host(&zeros), 20 * 4, None)?;

        // Persist the width, the column width, the pipe mode and the
        // selected entry.
        self.block_width = w;
        self.col_width = wc;
        self.pipe = pipe;
        self.kernel = entry(w, wc, pipe);
        self.grid = grid;
        self.shared = shared;
        self.ts_acc = [0.0; 9];
        self.ts_steps = 0;
        Ok(())
    }

    /// The selected block width (1 = S1-10 behavior, 2 = S1-11 widened).
    #[must_use]
    pub fn block_width(&self) -> u32 {
        self.block_width
    }

    /// The selected kernel entry's cubin symbol name (graph declaration
    /// takes the `CUkernel` handle by name — it must match the width the
    /// launch uses).
    #[must_use]
    pub fn kernel_name(&self) -> &'static str {
        match (self.block_width, self.col_width, self.pipe) {
            (1, _, _) => "decode_step_layer_fused",
            (2, 1, true) => "decode_step_layer_fused_bw2_pipe",
            (2, 1, false) => "decode_step_layer_fused_bw2",
            (2, 2, _) => "decode_step_layer_fused_bw2_wc2",
            (2, 4, _) => "decode_step_layer_fused_bw2_wc4",
            _ => "decode_step_layer_fused_bw2",
        }
    }

    /// Whether the two-tree pipe variant is selected (018 P1a).
    #[must_use]
    pub fn pipe(&self) -> bool {
        self.pipe
    }

    /// The co-resident grid size (graph declaration geometry).
    #[must_use]
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Dynamic shared memory of the launch (graph declaration geometry).
    #[must_use]
    pub fn shared(&self) -> u32 {
        self.shared
    }

    /// Raw cubin library handle (graph declaration takes the `CUkernel`
    /// handle via `cu_kernel_of`).
    pub fn raw_lib(&self) -> dsys::CUlibrary {
        self.lib.raw()
    }

    /// The stable const-blob device pointer.
    #[must_use]
    pub fn const_ptr(&self) -> *const c_void {
        self.cbuf.as_ptr() as *const c_void
    }

    /// The stable barrier-slot device pointer.
    #[must_use]
    pub fn bar_ptr(&self) -> *const c_void {
        self.bar.as_ptr() as *const c_void
    }

    /// The stable stage-clock device pointer (the profiler reads it back).
    #[must_use]
    pub fn stage_ts_ptr(&self) -> *const c_void {
        self.stage_ts.as_ptr() as *const c_void
    }

    /// One layer launch (the persistent kernel, 512 threads, grid =
    /// occupancy-gated, dynamic smem = (d + max_kv) * 4). `table` = the
    /// layer's 7-row plan pointer (from the fused geometry), `wnext` = the
    /// next layer's attn_norm (or final_norm for the last layer), `wffn` =
    /// this layer's ffn_norm, `wq`/`wk` = this layer's q/k head-norm
    /// weights (null when head_norm is off — the per-layer weights, like
    /// the S1-9 per-layer p2_qkv launches). `pos`/`token` are per-step
    /// (token is only read by stage 0 — layer 0).
    #[allow(clippy::too_many_arguments)] // kernel launch arg matrix (C3)
    pub fn launch(
        &self,
        stream: &CudaStream,
        table: *const PlanRow,
        wnext: *const u16,
        wffn: *const u16,
        wq: *const u16,
        wk: *const u16,
        li: i32,
        n_layers: i32,
        pos: u32,
        token: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let c_v: *const c_void = self.cbuf.as_ptr() as *const c_void;
        let t_v: *const PlanRow = table;
        let wn_v: *const u16 = wnext;
        let wf_v: *const u16 = wffn;
        let wq_v: *const u16 = wq;
        let wk_v: *const u16 = wk;
        let bar_v: *const c_void = self.bar.as_ptr() as *const c_void;
        let li_v: i32 = li;
        let nl_v: i32 = n_layers;
        let pos_v: u32 = pos;
        let tok_v: u32 = token;
        let mut args: [*mut c_void; 11] = [
            (&c_v as *const *const c_void) as *mut c_void,
            (&t_v as *const *const PlanRow) as *mut c_void,
            (&wn_v as *const *const u16) as *mut c_void,
            (&wf_v as *const *const u16) as *mut c_void,
            (&wq_v as *const *const u16) as *mut c_void,
            (&wk_v as *const *const u16) as *mut c_void,
            (&bar_v as *const *const c_void) as *mut c_void,
            (&li_v as *const i32) as *mut c_void,
            (&nl_v as *const i32) as *mut c_void,
            (&pos_v as *const u32) as *mut c_void,
            (&tok_v as *const u32) as *mut c_void,
        ];
        unsafe {
            launch_fmha(
                self.kernel,
                stream,
                self.dev,
                self.grid,
                1,
                1,
                512,
                self.shared,
                args.as_mut_ptr(),
            )
        }
    }

    /// Fold the per-stage clock64 marks of the last step into the
    /// aggregation window and print the 20-step mean table (called once
    /// per step when the decode profiler is active — the stream is
    /// already synchronized by `finalize`).
    pub fn profile_accumulate(&mut self, stream: &CudaStream) -> Result<(), LaunchError> {
        let n_layers = self.stage_ts.size() / (9 * 4);
        if n_layers == 0 {
            return Ok(());
        }
        let hb = HostBuffer::alloc(n_layers * 9 * 4)?;
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&self.stage_ts), n_layers * 9 * 4, None)?;
        stream.synchronize()?;
        let marks: Vec<u32> =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const u32, n_layers * 9).to_vec() };
        // Per-stage delta per layer: marks[li*9 + stage+1] - marks[li*9 +
        // stage]; the layer-0 marks also cover stage 0 (gather/rms0).
        let mut acc = [0u64; 9];
        let mut valid = 0u32;
        for li in 0..n_layers {
            let base = li * 9;
            if marks[base] == 0 {
                continue; // no mark for this layer (never ran)
            }
            valid += 1;
            for st in 0..9 {
                let a = marks[base + st];
                let b = if st + 1 < 9 { marks[base + st + 1] } else { marks[base + st] };
                acc[st] += b.wrapping_sub(a) as u64;
            }
        }
        if valid == 0 {
            return Ok(());
        }
        self.ts_steps += 1;
        for st in 0..9 {
            self.ts_acc[st] += acc[st] as f64 / (n_layers as f64);
        }
        if self.ts_steps.is_multiple_of(20) {
            let n = self.ts_steps as f64;
            let names = [
                "gather/rms0",
                "p1_qkv",
                "p2_qkv",
                "flash",
                "p1_o",
                "p2_o",
                "p1_gu",
                "p2_gu_d",
                "p2_down",
            ];
            println!(
                "[reinfer-cuda] layer-fused stage means (mean over {} steps, us):",
                self.ts_steps
            );
            let mut total = 0.0;
            for st in 0..9 {
                let ms = self.ts_acc[st] / n;
                total += ms;
                println!("  {:>10} {:8.3} us", names[st], ms * 1000.0);
            }
            println!("  {:>10} {:8.3} us (per layer)", "layer", total * 1000.0);
            self.ts_acc = [0.0; 9];
            self.ts_steps = 0;
        }
        Ok(())
    }
}
