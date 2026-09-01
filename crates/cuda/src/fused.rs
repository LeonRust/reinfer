//! S1-9: fused decode-step kernels — loader, plan tables and launches.
//!
//! The fused step replaces the split per-layer launch sequence (27 nodes)
//! with 8 nodes by merging the two-phase gemv launches and the small dense
//! kernels into one kernel per group (kernels/decode_fused_kernels.cu,
//! kernels/decode_flash_kernels.cu):
//!
//!   1. `gemv_m1_f16f32_multi`  — phase-1 of the m=1 plans in four
//!      launches per layer around one device-side `PlanRow` table:
//!      p1_qkv (q, k, v) at layer start reading xn (the layer input);
//!      p1_o (o) after the flash node, reading attn (the full attention
//!      row — its own node because the flash blocks each write only their
//!      own head's output); p1_gu (gate, up) after p2_o, reading xn
//!      (p2_o's ffn-normed x) — the exact points where the split path
//!      reads these buffers. 288 + 16 + 192 blocks for Qwen3-0.6B instead
//!      of 14 phase-1 launches;
//!   2. `gemv_p2_qkv_cast_hn_rope` — q/k/v phase-2 reductions + f16 casts +
//!      q/k head-norm + RoPE (one kernel; 16 blocks);
//!   3. `gemv_p2_gu_p1d_swiglu`   — gate/up phase-2 reductions + the fused
//!      cast-SiLU-GLU + the down phase-1 in one kernel (ncols_d * nslabs_d
//!      blocks — 96 for Qwen3-0.6B, the split's full down phase-1 tile
//!      grid): each block redundantly writes the 256-col phase-1 stripe
//!      its own phase-2 k-range lies in, then computes the
//!      (bx / nslabs_d, bx % nslabs_d) down tile (valid iff every slab's
//!      k-range fits in its block's stripe — a build_plans gate);
//!   4. `gemv_p2_add_rms`         — phase-2 reduction + residual add
//!      (exact add_cast semantics) + RMSNorm row; used for the o
//!      projection (residual into x, norm into xn with ffn_norm) and the
//!      down projection (residual into x, norm into xn with the next
//!      layer's attn_norm / final_norm);
//!   5. `decode_step_gqa_flash_fused` — kv write of the current slot
//!      (k16/v16 -> kv) + the flash decode attention in one kernel
//!      (16 blocks; the additions are block-local — see
//!      decode_flash_kernels.cu).
//!
//! So the fused layer is 8 nodes: [p1_qkv, p2_qkv, flash_fused (kv_write +
//! attention), p1_o, p2_o, p1_gu, p2_gu_d (swiglu + down phase-1),
//! p2_down] — 4 + 8*n_layers total (228 for Qwen3-0.6B vs 760).
//!
//! Bit-level identity: every fused kernel preserves the split kernels'
//! fixed accumulation orders (per-column ascending-slab phase-2 sums, the
//! per-thread strided rms sums with the 128/256-tree, the f16
//! round/widen/round chains) and the identical software RNE f16
//! conversions — see the kernel header for the per-fusion contracts.
//!
//! The slab-partials layout is shared: all layers reuse one segment
//! layout (each layer's p1 runs before its p2, layers are stream-serial),
//! and the lm_head row overlaps the same segments (it runs last). The
//! plan table holds 7 rows per layer (q, k, v, o, gate, up, down) plus
//! the lm_head row, uploaded once.

use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
use crate::gemm::Jgemm;
use crate::jit::{CtxGuard, JLib, KernelFn, launch_rows};
use crate::stream::CudaStream;
use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::PathBuf;

/// Host mirror of the kernel-side `PlanRow` (40 bytes, 8-aligned):
/// { a, b, partials, n, k, nslabs, col_off } — one row per gemv plan.
/// The multi phase-1 kernel decodes its (col, slab) tile from the row, so
/// every per-(col, slab) computation is identical to the single-plan
/// `gemv_m1_f16f32`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlanRow {
    /// [k] f16 activation row (the gemv input).
    pub a: *const u16,
    /// [k x n] f16 row-major weight matrix.
    pub b: *const u16,
    /// [nslabs x n] s-major slab partials (partials[slab*n + col]).
    pub partials: *mut f32,
    /// Output column count.
    pub n: i32,
    /// Reduction length.
    pub k: i32,
    /// k slabs (phase-1 grid split).
    pub nslabs: i32,
    /// Linearized (ncols*nslabs) block offset of this plan's segment in
    /// the multi-kernel grid.
    pub col_off: i32,
}

/// Stable fused geometry: segment pointers into the shared partials
/// buffer, the per-layer grids and the plan-table pointers. Everything is
/// fixed once `build_plans` runs, so the graph declaration records these
/// values in its argument cells (they never move).
#[derive(Debug, Clone)]
pub struct FusedGeom {
    /// Per-layer phase-1 grid of the q/k/v group (blocks) —
    /// `gemv_m1_f16f32_multi`, 3 rows, at layer start.
    pub grid_qkv: Vec<u32>,
    /// Per-layer phase-1 grid of the gate/up group (blocks) —
    /// `gemv_m1_f16f32_multi`, 2 rows, after p2_o.
    pub grid_gu: Vec<u32>,
    /// Per-layer phase-1 grid of the o plan (blocks) —
    /// `gemv_m1_f16f32_multi`, 1 row, after the flash node.
    pub grid_o: Vec<u32>,
    /// lm_head phase-1 grid (= ceil(vocab/256)).
    pub grid_lm: u32,
    /// `gemv_p2_qkv_cast_hn_rope` grid = ncols_q + ncols_k + ncols_v.
    pub grid_qkv_p2: u32,
    /// `gemv_p2_gu_swiglu` grid = ncols_ffn.
    pub grid_gu_p2: u32,
    /// Per-plan slab counts (phase-2 ascending-slab sums).
    pub nslabs_q: i32,
    /// k plan slab count.
    pub nslabs_k: i32,
    /// v plan slab count.
    pub nslabs_v: i32,
    /// o plan slab count.
    pub nslabs_o: i32,
    /// gate plan slab count (== up).
    pub nslabs_g: i32,
    /// down plan slab count.
    pub nslabs_d: i32,
    /// Shared partials segments (f32; all layers + lm reuse one layout).
    pub pq: *mut f32,
    /// k partials segment.
    pub pk: *mut f32,
    /// v partials segment.
    pub pv: *mut f32,
    /// o partials segment.
    pub po: *mut f32,
    /// gate partials segment.
    pub pg: *mut f32,
    /// up partials segment.
    pub pu: *mut f32,
    /// down partials segment.
    pub pd: *mut f32,
    /// lm_head partials segment.
    pub plm: *mut f32,
    /// Base of the device plan table (7 rows per layer, then the lm row).
    pub tables: *mut PlanRow,
    /// The lm_head row pointer (a = xn, b = lm_head, nslabs = 1).
    pub lm_table: *mut PlanRow,
}

/// The fused decode-step unit: the `decode_fused_kernels.cu` kernels plus
/// the shared partials buffer, the plan table and the geometry. The
/// lm_head phase-2 reduction reuses the existing
/// `gemv_m1_f16f32_reduce` kernel (`reduce`), passed in from the loaded
/// `Jgemm` — the fused path never loads gemm_m1.cu itself.
#[derive(Debug)]
pub struct FusedDecodeKernels {
    lib: JLib,
    p1: KernelFn,
    p2_qkv: KernelFn,
    p2_gu_d: KernelFn,
    p2_add_rms: KernelFn,
    reduce: KernelFn,
    dev: u32,
    /// Shared slab-partials layout (all layers + lm, see `FusedGeom`).
    partials: DeviceBuffer,
    /// Plan table bytes (7 rows per layer + 1 lm row).
    tables: DeviceBuffer,
    geom: FusedGeom,
}

impl FusedDecodeKernels {
    /// Load the fused kernels (standard JitCache pipeline — same shape as
    /// `Jgemm::new`). `reduce` is the lm_head phase-2 kernel from the
    /// engine's loaded `Jgemm`.
    pub fn new(
        dev: u32,
        arch: &str,
        cache_dir: Option<PathBuf>,
        reduce: KernelFn,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "decode_fused",
            src: include_str!("../kernels/decode_fused_kernels.cu"),
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
        let p1 = lib.kernel("gemv_m1_f16f32_multi")?;
        let p2_qkv = lib.kernel("gemv_p2_qkv_cast_hn_rope")?;
        let p2_gu_d = lib.kernel("gemv_p2_gu_p1d_swiglu")?;
        let p2_add_rms = lib.kernel("gemv_p2_add_rms")?;
        Ok(Self {
            lib,
            p1,
            p2_qkv,
            p2_gu_d,
            p2_add_rms,
            reduce,
            dev,
            partials: DeviceBuffer::alloc(DeviceId::new(dev), 4)?, // placeholder
            tables: DeviceBuffer::alloc(DeviceId::new(dev), 4)?,   // placeholder
            geom: FusedGeom {
                grid_qkv: Vec::new(),
                grid_gu: Vec::new(),
                grid_o: Vec::new(),
                grid_lm: 1,
                grid_qkv_p2: 1,
                grid_gu_p2: 1,
                nslabs_q: 1,
                nslabs_k: 1,
                nslabs_v: 1,
                nslabs_o: 1,
                nslabs_g: 1,
                nslabs_d: 1,
                pq: std::ptr::null_mut(),
                pk: std::ptr::null_mut(),
                pv: std::ptr::null_mut(),
                po: std::ptr::null_mut(),
                pg: std::ptr::null_mut(),
                pu: std::ptr::null_mut(),
                pd: std::ptr::null_mut(),
                plm: std::ptr::null_mut(),
                tables: std::ptr::null_mut(),
                lm_table: std::ptr::null_mut(),
            },
        })
    }

    /// Build the plan table + shared partials layout + geometry from the
    /// engine's decode plans (all layers share the shapes of layer 0; the
    /// lm_head row is appended after the 7*n_layers rows).
    ///
    /// Merge-validity gate: the `gemv_p2_gu_p1d_swiglu` fusion runs on
    /// the down plan's full phase-1 tile grid (ncols_d * nslabs_d blocks);
    /// block bx redundantly writes the 256-col phase-1 stripe
    /// [(bx % nslabs_d)/(256/slab_k_d) * 256, ...) and its phase-2
    /// k-range (slab bx % nslabs_d) must lie inside that stripe, with
    /// the stripes covering the whole down k-range (the divisor mirrors
    /// the kernel's c1 mapping — S1-9b admits slab_k in {256, 128, 64}).
    /// Configs that violate this fall back to the split path (Fatal,
    /// the dispatcher falls through).
    pub fn build_plans(
        &mut self,
        dev: DeviceId,
        jg: &Jgemm,
        plans: &crate::engine::DecodeGemmPlans,
    ) -> Result<(), LaunchError> {
        let n_layers = plans.layers.len();
        let l0 = &plans.layers[0];
        let nq = l0.q.n as usize;
        let nk = l0.k.n as usize;
        let nv = l0.v.n as usize;
        let h = l0.o.n as usize;
        let ffn = l0.gate.n as usize;
        let vocab = plans.lm_head.n as usize;
        let (ncols_q, nslabs_q) = jg.shape(l0.q.n, l0.q.k);
        let (ncols_k, nslabs_k) = jg.shape(l0.k.n, l0.k.k);
        let (ncols_v, nslabs_v) = jg.shape(l0.v.n, l0.v.k);
        let (ncols_o, nslabs_o) = jg.shape(l0.o.n, l0.o.k);
        let (ncols_g, nslabs_g) = jg.shape(l0.gate.n, l0.gate.k);
        let (_, nslabs_u) = jg.shape(l0.up.n, l0.up.k);
        let (ncols_d, nslabs_d) = jg.shape(l0.down.n, l0.down.k);
        debug_assert_eq!(nslabs_u, nslabs_g, "gate/up share the shape");
        // Merge-validity gate for `gemv_p2_gu_p1d_swiglu` (see the kernel
        // doc): the down plan's full phase-1 tile grid runs in one kernel.
        // Block bx's phase-2 k-range (its slab's [ks, ke)) must lie inside
        // the phase-1 stripe [(slab/per_stripe)*256, ...+256) the block
        // writes itself (block-local after __syncthreads), with
        // per_stripe = 256 / slab_k_d (the kernel's c1 divisor; S1-9b
        // admits slab_k in {256, 128, 64} — 1/2/4 slabs per 256-stripe),
        // and the stripes must cover the whole down k-range (phase-1
        // computes rd.a[0..k)).
        let slab_k_d = (l0.down.k as u32).div_ceil(nslabs_d);
        let mut merged_ok = slab_k_d <= 256 && 256 % slab_k_d == 0;
        if merged_ok {
            let per_stripe = 256 / slab_k_d;
            if (nslabs_d / per_stripe) * 256 >= l0.down.k as u32 {
                for s in 0..nslabs_d {
                    let s = s as u64;
                    let ks = s * slab_k_d as u64;
                    let ke = (ks + slab_k_d as u64).min(l0.down.k as u64);
                    let lo = (s / per_stripe as u64) * 256;
                    if ks < lo || ke > lo + 256 {
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

        // Shared segment offsets (f32 elements). Every layer reuses the
        // same layout (stream-serial: a layer's p2 consumes its p1
        // partials before the next layer's p1 overwrites them); the lm
        // head runs after the last layer and overlaps the same segments.
        let off_q = 0usize;
        let off_k = off_q + nq * nslabs_q as usize;
        let off_v = off_k + nk * nslabs_k as usize;
        let off_o = off_v + nv * nslabs_v as usize;
        let off_g = off_o + h * nslabs_o as usize;
        let off_u = off_g + ffn * nslabs_g as usize;
        let off_d = off_u + ffn * nslabs_u as usize;
        let off_lm = off_d + h * nslabs_d as usize;
        let total = off_lm + vocab; // lm nslabs == 1

        let _guard = CtxGuard::set_current(dev.index())?;
        self.partials = DeviceBuffer::alloc(dev, total * 4)?;
        self.tables =
            DeviceBuffer::alloc(dev, (n_layers * 7 + 1) * std::mem::size_of::<PlanRow>())?;
        let base = self.partials.as_ptr() as *mut f32;
        let pq = unsafe { base.add(off_q) };
        let pk = unsafe { base.add(off_k) };
        let pv = unsafe { base.add(off_v) };
        let po = unsafe { base.add(off_o) };
        let pg = unsafe { base.add(off_g) };
        let pu = unsafe { base.add(off_u) };
        let pd = unsafe { base.add(off_d) };
        let plm = unsafe { base.add(off_lm) };

        // Per-layer segment block offsets (cumulative ncols) and the row
        // table bytes. One row per (layer, plan): the exact a/b/n/k the
        // split path would launch, plus the shared segment's partials
        // pointer and the segment's grid offset.
        let row = |a: *const u16,
                   b: *const u16,
                   partials: *mut f32,
                   n: usize,
                   k: usize,
                   nslabs: u32,
                   col_off: i32|
         -> PlanRow {
            PlanRow {
                a,
                b,
                partials,
                n: n as i32,
                k: k as i32,
                nslabs: nslabs as i32,
                col_off,
            }
        };
        // Multi-kernel block offsets per plan: each plan's segment is
        // ncols*nslabs tiles, and the kernel decodes its tile as
        // local = bx - col_off, so the col_offs must be cumulative BLOCK
        // counts (ncols*nslabs per plan), not ncols counts. The seven
        // rows are split into four launches (q/k/v, o, gate/up, down —
        // the down launch is the phase-2 of the merged p2_gu_d kernel)
        // with col_offs relative to each launch's own grid.
        let b_q = ncols_q * nslabs_q;
        let b_k = b_q + ncols_k * nslabs_k;
        let b_v = b_k + ncols_v * nslabs_v;
        let b_o = ncols_o * nslabs_o; // o grid (its own launch)
        let b_g = ncols_g * nslabs_g; // gate/up group, relative to its grid
        let b_u = b_g + ncols_g * nslabs_g; // gate/up share the shape
        let mut grid_qkv = Vec::with_capacity(n_layers);
        let mut grid_gu = Vec::with_capacity(n_layers);
        let mut grid_o = Vec::with_capacity(n_layers);
        let mut bytes = Vec::with_capacity((n_layers * 7 + 1) * std::mem::size_of::<PlanRow>());
        for pl in &plans.layers {
            let (nslabs, seg, n, k, col_off) = (nslabs_q, pq, nq, l0.q.k as usize, 0i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(pl.q.a as *const u16, pl.q.b as *const u16, seg, n, k, nslabs, col_off)
                        as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_k, pk, nk, l0.k.k as usize, b_q as i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(pl.k.a as *const u16, pl.k.b as *const u16, seg, n, k, nslabs, col_off)
                        as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_v, pv, nv, l0.v.k as usize, b_k as i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(pl.v.a as *const u16, pl.v.b as *const u16, seg, n, k, nslabs, col_off)
                        as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_o, po, h, l0.o.k as usize, 0i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(pl.o.a as *const u16, pl.o.b as *const u16, seg, n, k, nslabs, col_off)
                        as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_g, pg, ffn, l0.gate.k as usize, 0i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(
                        pl.gate.a as *const u16,
                        pl.gate.b as *const u16,
                        seg,
                        n,
                        k,
                        nslabs,
                        col_off,
                    ) as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_u, pu, ffn, l0.up.k as usize, b_g as i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(pl.up.a as *const u16, pl.up.b as *const u16, seg, n, k, nslabs, col_off)
                        as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            let (nslabs, seg, n, k, col_off) =
                (nslabs_d, pd, h, l0.down.k as usize, 0i32);
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &row(
                        pl.down.a as *const u16,
                        pl.down.b as *const u16,
                        seg,
                        n,
                        k,
                        nslabs,
                        col_off,
                    ) as *const PlanRow as *const u8,
                    std::mem::size_of::<PlanRow>(),
                )
            });
            grid_qkv.push(b_v);
            grid_o.push(b_o);
            grid_gu.push(b_u);
        }
        // lm_head row: a = xn, b = lm_head weight, nslabs = 1, its own
        // grid offset 0 (the row's own segment in the multi grid).
        let (ncols_lm, nslabs_lm) = jg.shape(plans.lm_head.n, plans.lm_head.k);
        debug_assert_eq!(nslabs_lm, 1);
        // NOTE: the row must be a named local before the byte view — reading
        // the address of the closure-call rvalue (`&row(...)`) through a raw
        // slice returned garbage at opt-level 3 (the rvalue was never
        // materialized; the kernel then saw n=-1/k=-1, early-returned, and
        // the lm_head logits came out all-zero in release builds).
        let lm_row = row(
            plans.lm_head.a as *const u16,
            plans.lm_head.b as *const u16,
            plm,
            vocab,
            plans.lm_head.k as usize,
            nslabs_lm,
            0,
        );
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &lm_row as *const PlanRow as *const u8,
                std::mem::size_of::<PlanRow>(),
            )
        });

        // Host -> device upload of the table.
        let hb = HostBuffer::alloc(bytes.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), hb.as_ptr() as *mut u8, bytes.len());
        }
        copy(&mut MemRef::Device(&self.tables), &MemRef::Host(&hb), bytes.len(), None)?;

        let tables = self.tables.as_ptr() as *mut PlanRow;
        let lm_table = unsafe { tables.add(n_layers * 7) };
        self.geom = FusedGeom {
            grid_qkv,
            grid_gu,
            grid_o,
            grid_lm: ncols_lm,
            grid_qkv_p2: ncols_q + ncols_k + ncols_v,
            grid_gu_p2: ncols_d * nslabs_d,
            nslabs_q: nslabs_q as i32,
            nslabs_k: nslabs_k as i32,
            nslabs_v: nslabs_v as i32,
            nslabs_o: nslabs_o as i32,
            nslabs_g: nslabs_g as i32,
            nslabs_d: nslabs_d as i32,
            pq,
            pk,
            pv,
            po,
            pg,
            pu,
            pd,
            plm,
            tables,
            lm_table,
        };
        Ok(())
    }

    /// Stable geometry (segment pointers, grids, table pointers).
    #[must_use]
    pub fn geom(&self) -> &FusedGeom {
        &self.geom
    }

    /// Raw cubin library handle — the graph declaration takes the
    /// `CUkernel` handles via `cu_kernel_of` (the handle form capture
    /// records for `cuLibraryLoadData` kernels, graph.rs `FN_TYPE`).
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }

    /// Phase-1 multi kernel launch: `table` = the device row pointer of
    /// this layer (or the lm row), `nplans` rows, `grid` = the segment's
    /// total block count. C3 discipline: all args are locals.
    pub fn launch_p1(
        &self,
        stream: &CudaStream,
        table: *const PlanRow,
        nplans: u32,
        grid: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let t_v: *const PlanRow = table;
        let n_v: i32 = nplans as i32;
        let mut args: [*mut c_void; 2] = [
            (&t_v as *const *const PlanRow) as *mut c_void,
            (&n_v as *const i32) as *mut c_void,
        ];
        unsafe { launch_rows(self.p1, stream, self.dev, grid, 256, args.as_mut_ptr()) }
    }

    /// q/k/v phase-2 + casts + head-norm + rope kernel launch. `wq`/`wk`
    /// are the head-norm weight pointers (may be null when `hn == 0`).
    #[allow(clippy::too_many_arguments)] // kernel launch arg matrix (C3)
    pub fn launch_p2_qkv(
        &self,
        stream: &CudaStream,
        pq: *const f32,
        pk: *const f32,
        pv: *const f32,
        q16: *mut u16,
        k16: *mut u16,
        v16: *mut u16,
        wq: *const u16,
        wk: *const u16,
        nq: u32,
        nk: u32,
        nv: u32,
        nslabs_q: i32,
        nslabs_k: i32,
        nslabs_v: i32,
        d: u32,
        half: u32,
        pos: u32,
        eta: f32,
        scale_q: f32,
        scale_k: f32,
        eps: f32,
        hn: u32,
        grid: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let pq_v = pq;
        let pk_v = pk;
        let pv_v = pv;
        let q16_v = q16;
        let k16_v = k16;
        let v16_v = v16;
        let wq_v = wq;
        let wk_v = wk;
        let nq_v = nq;
        let nk_v = nk;
        let nv_v = nv;
        let nsq_v = nslabs_q;
        let nsk_v = nslabs_k;
        let nsv_v = nslabs_v;
        let d_v = d;
        let half_v = half;
        let pos_v = pos;
        let eta_v = eta;
        let sq_v = scale_q;
        let sk_v = scale_k;
        let eps_v = eps;
        let hn_v = hn;
        let mut args: [*mut c_void; 22] = [
            (&pq_v as *const *const f32) as *mut c_void,
            (&pk_v as *const *const f32) as *mut c_void,
            (&pv_v as *const *const f32) as *mut c_void,
            (&q16_v as *const *mut u16) as *mut c_void,
            (&k16_v as *const *mut u16) as *mut c_void,
            (&v16_v as *const *mut u16) as *mut c_void,
            (&wq_v as *const *const u16) as *mut c_void,
            (&wk_v as *const *const u16) as *mut c_void,
            (&nq_v as *const u32) as *mut c_void,
            (&nk_v as *const u32) as *mut c_void,
            (&nv_v as *const u32) as *mut c_void,
            (&nsq_v as *const i32) as *mut c_void,
            (&nsk_v as *const i32) as *mut c_void,
            (&nsv_v as *const i32) as *mut c_void,
            (&d_v as *const u32) as *mut c_void,
            (&half_v as *const u32) as *mut c_void,
            (&pos_v as *const u32) as *mut c_void,
            (&eta_v as *const f32) as *mut c_void,
            (&sq_v as *const f32) as *mut c_void,
            (&sk_v as *const f32) as *mut c_void,
            (&eps_v as *const f32) as *mut c_void,
            (&hn_v as *const u32) as *mut c_void,
        ];
        unsafe { launch_rows(self.p2_qkv, stream, self.dev, grid, 256, args.as_mut_ptr()) }
    }

    /// Merged gate/up phase-2 + cast-SiLU-GLU + down phase-1 launch.
    /// `table` = the gate row pointer (rows g, u, d); `grid` = the
    /// gate/up p2 grid (ncols_g blocks) — each block's down phase-1
    /// slabs lie inside its own phase-1 columns (build_plans gate).
    pub fn launch_p2_gu_d(
        &self,
        stream: &CudaStream,
        table: *const PlanRow,
        grid: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let t_v: *const PlanRow = table;
        let n_v: i32 = 3;
        let mut args: [*mut c_void; 2] = [
            (&t_v as *const *const PlanRow) as *mut c_void,
            (&n_v as *const i32) as *mut c_void,
        ];
        unsafe { launch_rows(self.p2_gu_d, stream, self.dev, grid, 256, args.as_mut_ptr()) }
    }

    /// Phase-2 + residual add + RMSNorm row launch (single block 256; the
    /// o and down variants share this kernel — see the kernel header).
    pub fn launch_p2_add_rms(
        &self,
        stream: &CudaStream,
        partials: *const f32,
        x: *mut u16,
        out: *mut u16,
        w: *const u16,
        n: u32,
        nslabs: i32,
        eps: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let p_v = partials;
        let x_v = x;
        let out_v = out;
        let w_v = w;
        let n_v = n;
        let ns_v = nslabs;
        let eps_v = eps;
        let mut args: [*mut c_void; 7] = [
            (&p_v as *const *const f32) as *mut c_void,
            (&x_v as *const *mut u16) as *mut c_void,
            (&out_v as *const *mut u16) as *mut c_void,
            (&w_v as *const *const u16) as *mut c_void,
            (&n_v as *const u32) as *mut c_void,
            (&ns_v as *const i32) as *mut c_void,
            (&eps_v as *const f32) as *mut c_void,
        ];
        unsafe { launch_rows(self.p2_add_rms, stream, self.dev, 1, 256, args.as_mut_ptr()) }
    }

    /// lm_head phase-2 reduction (the shared `gemv_m1_f16f32_reduce`
    /// kernel): ascending-slab sum of the lm row's partials into logits.
    pub fn launch_reduce(
        &self,
        stream: &CudaStream,
        partials: *const f32,
        c: *mut f32,
        n: u32,
        nslabs: i32,
        grid: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let p_v = partials;
        let c_v = c;
        let n_v = n;
        let ns_v = nslabs;
        let mut args: [*mut c_void; 4] = [
            (&p_v as *const *const f32) as *mut c_void,
            (&c_v as *const *mut f32) as *mut c_void,
            (&n_v as *const u32) as *mut c_void,
            (&ns_v as *const i32) as *mut c_void,
        ];
        unsafe { launch_rows(self.reduce, stream, self.dev, grid, 256, args.as_mut_ptr()) }
    }
}
