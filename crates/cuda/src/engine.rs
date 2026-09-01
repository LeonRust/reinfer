//! 014 L3 单请求 CUDA 引擎：dense decoder 全链（embed → RMSNorm → QKV →
//! q/k head norm（可选）→ RoPE → paged KV → decode → o → 残差 → FFN →
//! final norm → lm_head → logits）。
//!
//! 输入面：safetensors 权重（BF16/F16/F32——主机侧转 f16 位上传）+ `LlamaConfig`
//! （`arch::from_hf_config` / `from_gguf_meta`）；**无模型名/架构名硬编码**
//! （head norm 由 `cfg.head_norm` 驱动；张量名取 HF 通用键族）。
//!
//! 数值档：f16 存储层 + f32 数学（GEMM `CUBLAS_COMPUTE_32F` 16F-in；小内核
//! f32 累加——非 16F-acc 记录档）。单 token 步进（prefill 批量化
//! [T7 prefill_attention 已就] 在性能后议清单——首版正确性优先）。
//!
//! 生成语义（014 T9 必备块）：EOS 停 / `max_tokens` 硬限 / logits 全 NaN 显式
//! 错误 / embedding OOV → 错误 / `temperature == 0` argmax（tie-break 首个
//! 最大——012 语义；temperature > 0 暂不支持，显式错误避免静默）。

use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
use crate::decode::DecodeKernels;
use crate::diff::DiffKernels;
use crate::error::from_runtime_error;
use crate::event::CudaEvent;
use crate::fmha::{FmhaKernels, PrefillKernels};
use crate::fused::{FusedDecodeKernels, PlanRow};
use crate::gemm::{Gemm, GemmPlan, GpuMat, Jgemm};
use crate::graph::{
    GraphExec, GraphPool, KernelSpec, NodeRole, ParamLayout, PtrRole, PtrUpdate, bucket_index,
    layout_slots,
};
use crate::jit::{CtxGuard, JLib, KernelFn, cu_kernel_of, launch_fmha};
use crate::stream::CudaStream;
use cudarc::cublas::sys as blas;
use cudarc::driver::sys as dsys;
use reinfer_arch::llama::{LlamaConfig, from_hf_config};
use reinfer_core::{DType, DeviceId};
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::{LaunchError, OpConfig, ProviderChoice, ProviderSet, SelectionCache, TuneDb};
use reinfer_safetensors::{SafeFile, StDtype};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// 引擎错误面。
#[derive(Debug)]
pub enum EngineError {
    /// 底层 launch/mem 错误。
    Launch(LaunchError),
    /// 配置/架构解析错误。
    Arch(reinfer_arch::llama::ArchError),
    /// 输入面 IO 错误。
    Io(std::io::Error),
    /// safetensors 数据错误。
    Sts(String),
    /// 缺失张量。
    MissingTensor(String),
    /// 权重形状不一致。
    WeightShape(String),
    /// 不支持 dtype。
    UnsupportedDtype(String),
    /// embedding 越界 token。
    EmbeddingOov(u32),
    /// logits 全量 NaN（生成语义：显式错误）。
    NaNLogits,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Launch(e) => write!(f, "launch: {e}"),
            EngineError::Arch(e) => write!(f, "arch: {e}"),
            EngineError::Io(e) => write!(f, "io: {e}"),
            EngineError::Sts(s) => write!(f, "safetensors: {s}"),
            EngineError::MissingTensor(t) => write!(f, "missing tensor: {t}"),
            EngineError::WeightShape(s) => write!(f, "weight shape: {s}"),
            EngineError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {d}"),
            EngineError::EmbeddingOov(t) => write!(f, "embedding OOV token: {t}"),
            EngineError::NaNLogits => write!(f, "logits contain only NaN — refuse to sample"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LaunchError> for EngineError {
    fn from(e: LaunchError) -> Self {
        EngineError::Launch(e)
    }
}
impl From<reinfer_arch::llama::ArchError> for EngineError {
    fn from(e: reinfer_arch::llama::ArchError) -> Self {
        EngineError::Arch(e)
    }
}
impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e)
    }
}

/// dense 小内核装载单元（rms/head-rms/rope/残差加/swiglu/kv-write/embed 行拷贝）。
#[derive(Debug)]
/// dense 前向小内核装载器集合。
pub struct DenseKernels {
    /// 加载的 cubin（保活）。
    #[allow(dead_code)]
    lib: JLib,
    rms_row: KernelFn,
    rms_heads: KernelFn,
    rope: KernelFn,
    /// S1-2: batched per-head rope (one launch per q/k buffer; optional
    /// element scale folded in — replaces the per-head rope + scale passes).
    rope_heads: KernelFn,
    add: KernelFn,
    /// S1-2: fused f32->f16 cast + residual add (o/ffn residuals).
    add_cast: KernelFn,
    swiglu: KernelFn,
    /// S1-4: fused f32->f16 cast + SiLU-GLU (gate/up GEMM outputs in).
    cast_swiglu: KernelFn,
    /// S1-4: fused residual add + RMSNorm row (o residual + ffn norm).
    add_rms: KernelFn,
    kv_write: KernelFn,
    gather: KernelFn,
    scale: KernelFn,
    // 014 S0-3b: parity-f32 tier — f32 variants (gather/scale/swiglu; the
    // f32 rms/rope come from DiffKernels' criterion kernels).
    gather_f32: KernelFn,
    scale_f32: KernelFn,
    swiglu_f32: KernelFn,
}

impl DenseKernels {
    /// 装载（JitCache 管线；与 decode.rs 同构）。
    pub fn new(arch: &str, cache_dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "dense_kernels",
            src: include_str!("../kernels/dense_kernels.cu"),
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
        let rms_row = lib.kernel("rms_norm_row_f16")?;
        let rms_heads = lib.kernel("rms_norm_heads_f16")?;
        let rope = lib.kernel("rope_neox_f16")?;
        let rope_heads = lib.kernel("rope_heads_f16")?;
        let add = lib.kernel("add_f16_to_f16")?;
        let add_cast = lib.kernel("add_cast_f16")?;
        let swiglu = lib.kernel("swiglu_f16")?;
        let cast_swiglu = lib.kernel("fused_cast_swiglu_f16")?;
        let add_rms = lib.kernel("fused_add_rms_f16")?;
        let kv_write = lib.kernel("kv_write_row")?;
        let gather = lib.kernel("gather_row")?;
        let scale = lib.kernel("scale_f16")?;
        let gather_f32 = lib.kernel("gather_row_f32")?;
        let scale_f32 = lib.kernel("scale_f32")?;
        let swiglu_f32 = lib.kernel("swiglu_f32")?;
        Ok(Self {
            lib,
            rms_row,
            rms_heads,
            rope,
            rope_heads,
            add,
            add_cast,
            swiglu,
            cast_swiglu,
            add_rms,
            kv_write,
            gather,
            scale,
            gather_f32,
            scale_f32,
            swiglu_f32,
        })
    }

    /// 原始 cubin 库句柄（S1-3 graph spec 声明取 `CUkernel` 用——capture
    /// 记录的是 `CUkernel` 形态，见 `JLib::cu_kernel`）。
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }

    /// RMSNorm 行（单 block 256；n ≤ 1024）。
    pub fn launch_rms_norm(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const u16,
        out: *mut u16,
        w: *const u16,
        n: u32,
        eps: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [n as i32, eps.to_bits() as i32];
        let mut args: [*mut c_void; 5] = [
            (&x as *const *const u16) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&w as *const *const u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        unsafe { crate::jit::launch_row(self.rms_row, stream, dev, 256, args.as_mut_ptr()) }
    }

    /// 每头 RMSNorm（grid = rows × 256；w 共享）。
    pub fn launch_rms_heads(
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
        let nv: [i32; 3] = [n as i32, eps.to_bits() as i32, rows as i32];
        let mut args: [*mut c_void; 6] = [
            (&x as *const *const u16) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&w as *const *const u16) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_grid(self.rms_heads, stream, dev, rows, 1, 256, 1, args.as_mut_ptr())
        }
    }

    /// RoPE（NEOX 半分割就地单 head，grid 1 block；half ≤ 1024）。
    pub fn launch_rope_row(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut u16,
        half: u32,
        pos: u32,
        eta: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 3] = [half as i32, pos as i32, eta.to_bits() as i32];
        let mut args: [*mut c_void; 4] = [
            (&x as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
        ];
        unsafe { crate::jit::launch_row(self.rope, stream, dev, 256, args.as_mut_ptr()) }
    }

    /// 批量逐头 RoPE（S1-2：grid = heads，block 256；`scale` 折叠进 q 趟
    /// （k 趟传 1.0）——替代每头一 launch + 独立 scale 趟；位等价见内核注）。
    pub fn launch_rope_heads(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut u16,
        heads: u32,
        half: u32,
        pos: u32,
        eta: f32,
        scale: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 5] =
            [heads as i32, half as i32, pos as i32, eta.to_bits() as i32, scale.to_bits() as i32];
        let mut args: [*mut c_void; 6] = [
            (&x as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(self.rope_heads, stream, dev, heads, 256, args.as_mut_ptr())
        }
    }

    /// 残差加：out += x。
    pub fn launch_add(
        &self,
        dev: u32,
        stream: &CudaStream,
        out: *mut u16,
        x: *const u16,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 3] = [
            (&out as *const *mut u16) as *mut c_void,
            (&x as *const *const u16) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(self.add, stream, dev, n.div_ceil(256), 256, args.as_mut_ptr())
        }
    }

    /// 融合 cast + 残差加（S1-2）：out[i] += f16(x[i])——替代
    /// cast_f32_f16 + add_f16_to_f16 两 launch（位等价见内核注）。
    pub fn launch_add_cast(
        &self,
        dev: u32,
        stream: &CudaStream,
        out: *mut u16,
        x: *const f32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 3] = [
            (&out as *const *mut u16) as *mut c_void,
            (&x as *const *const f32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.add_cast,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// S1-4: fused f32->f16 cast + SiLU-GLU (FFN gate/up GEMM outputs in,
    /// f16 SiLU-GLU product out). Replaces cast_gate + cast_up + swiglu
    /// (3 launches -> 1); bit-identical — see the kernel note.
    pub fn launch_fused_cast_swiglu(
        &self,
        dev: u32,
        stream: &CudaStream,
        gate: *const f32,
        up: *const f32,
        out: *mut u16,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 4] = [
            (&gate as *const *const f32) as *mut c_void,
            (&up as *const *const f32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.cast_swiglu,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// S1-4: fused residual add + RMSNorm row (o-projection residual into
    /// the ffn norm): x[i] = x[i] + f16(c[i]) in place, out = rms(x) with
    /// weight w. Replaces add_cast_f16 + rms_norm_row_f16 (2 launches -> 1);
    /// bit-identical — see the kernel note.
    #[allow(clippy::too_many_arguments)] // kernel launch arg matrix (C3 discipline)
    pub fn launch_fused_add_rms(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut u16,
        c: *const f32,
        out: *mut u16,
        w: *const u16,
        n: u32,
        eps: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [n as i32, eps.to_bits() as i32];
        let mut args: [*mut c_void; 6] = [
            (&x as *const *mut u16) as *mut c_void,
            (&c as *const *const f32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&w as *const *const u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        unsafe { crate::jit::launch_row(self.add_rms, stream, dev, 256, args.as_mut_ptr()) }
    }

    /// SiLU-GLU：out = silu(gate)*up。
    pub fn launch_swiglu(
        &self,
        dev: u32,
        stream: &CudaStream,
        gate: *const u16,
        up: *const u16,
        out: *mut u16,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 4] = [
            (&gate as *const *const u16) as *mut c_void,
            (&up as *const *const u16) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.swiglu,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// KV 行写。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_write(
        &self,
        dev: u32,
        stream: &CudaStream,
        k_row: *const u16,
        v_row: *const u16,
        kv: *mut u16,
        phys: u32,
        off: u32,
        block_len: u32,
        kv_heads: u32,
        d: u32,
        total_pages: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 5] = [phys as i32, off as i32, block_len as i32, kv_heads as i32, d as i32];
        let mut args: [*mut c_void; 9] = [
            (&k_row as *const *const u16) as *mut c_void,
            (&v_row as *const *const u16) as *mut c_void,
            (&kv as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&total_pages as *const u32) as *mut c_void,
        ];
        let per_tok = (kv_heads * d) as usize;
        unsafe {
            crate::jit::launch_rows(
                self.kv_write,
                stream,
                dev,
                per_tok.div_ceil(256) as u32,
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 元素缩放（x *= scale；grid n/256）。
    pub fn launch_scale(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut u16,
        n: u32,
        scale: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [n as i32, scale.to_bits() as i32];
        let mut args: [*mut c_void; 3] = [
            (&x as *const *mut u16) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.scale,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// embed 行拷贝。
    pub fn launch_gather(
        &self,
        dev: u32,
        stream: &CudaStream,
        src: *const u16,
        dst: *mut u16,
        row: u32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 4] = [
            (&src as *const *const u16) as *mut c_void,
            (&dst as *const *mut u16) as *mut c_void,
            (&row as *const u32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.gather,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 014 S0-3b: f32 embed 行拷贝（parity-f32 档；src 为 f32 embed）。
    pub fn launch_gather_f32(
        &self,
        dev: u32,
        stream: &CudaStream,
        src: *const f32,
        dst: *mut f32,
        row: u32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 4] = [
            (&src as *const *const f32) as *mut c_void,
            (&dst as *const *mut f32) as *mut c_void,
            (&row as *const u32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.gather_f32,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 014 S0-3b: f32 元素缩放（parity-f32 档；注意力 1/sqrt(d) 缩放点）。
    pub fn launch_scale_f32(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *mut f32,
        n: u32,
        scale: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let nv: [i32; 2] = [n as i32, scale.to_bits() as i32];
        let mut args: [*mut c_void; 3] = [
            (&x as *const *mut f32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.scale_f32,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 014 S0-3b: f32 SiLU-GLU（parity-f32 档；与 swiglu_f16 同数学）。
    pub fn launch_swiglu_f32(
        &self,
        dev: u32,
        stream: &CudaStream,
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let mut args: [*mut c_void; 4] = [
            (&gate as *const *const f32) as *mut c_void,
            (&up as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe {
            crate::jit::launch_rows(
                self.swiglu_f32,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }
}

/// 层的设备权重（f16 位；矩阵已转置为 [k×n] 行主序——gemm B 约定）。
#[derive(Debug)]
pub struct LayerWeights {
    attn_norm: DeviceBuffer,
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
    /// S1-7: fused QKV weight [h x (nqk+2kvk)] row-major = q_proj ++ k_proj
    /// ++ v_proj **column join** (row r = q row r ++ k row r ++ v row r;
    /// all three share the k=h contraction dim). Built per-row at load on
    /// the f16 channel only (None on the parity-f32 tier, which routes
    /// prefill away from this path); the separated buffers stay for decode.
    qkv_proj: Option<DeviceBuffer>,
    o_proj: DeviceBuffer,
    q_norm: Option<DeviceBuffer>,
    k_norm: Option<DeviceBuffer>,
    ffn_norm: DeviceBuffer,
    gate_proj: DeviceBuffer,
    up_proj: DeviceBuffer,
    down_proj: DeviceBuffer,
}

/// 单请求解码引擎。
#[derive(Debug)]
pub struct Engine {
    dev: u32,
    cfg: LlamaConfig,
    gemm: Gemm,
    /// JIT m=1 GEMM (JitGemm): the decode step's m=1 projections run the
    /// `gemv_m1_f16f32` kernel instead of cublas (REINFER_JGEMM, default on;
    /// `None` when disabled by env, when the graph pool is active — the
    /// graph declaration still declares cublas nodes this wave — or when
    /// the kernel failed to load (fail-open to cublas with a note)).
    jgemm: Option<Jgemm>,
    /// JIT-gemm launch failures that fell back to cublas (interior
    /// mutable — `gemm_exec_plan` is &self; observable for tests).
    jgemm_fallbacks: AtomicU64,
    /// S1-9: fused decode-step kernels (REINFER_FUSED, default on). When
    /// loaded, the decode step runs 8 nodes/layer instead of 27
    /// (bit-identical — see fused.rs), and the graph declaration mirrors
    /// the fused sequence. `None` when disabled by env, when the jgemm
    /// path is off (the fused kernels are the fused jgemm phases), on the
    /// parity-f32 tier, when the naive decode attention is forced, when
    /// the fused geometry is unsupported (head_dim must divide the
    /// 256-thread block, >= 32; hidden <= 1024), or when the kernels
    /// failed to load (fail-open to the split path with a note).
    fused: Option<FusedDecodeKernels>,
    kernels: DenseKernels,
    diff: DiffKernels,
    decode: DecodeKernels,
    /// 006 T1：FMHA 批 prefill 装载（惰性；装载失败 → 逐 token 回退）。
    fmha: Option<FmhaKernels>,
    /// 006 T1：批 prefill 伴生内核（逐行 rms/rope/gather + 批 KV 写）。
    prefill: Option<PrefillKernels>,
    /// 目标架构（FMHA 惰性装载用）。
    arch: String,
    stream: CudaStream,
    embed: DeviceBuffer,
    lm_head: DeviceBuffer,
    final_norm: DeviceBuffer,
    layers: Vec<LayerWeights>,
    kv: crate::decode::KvStore,
    pp: usize,
    /// S1-2: static identity page table — `pages_dev[i] = i` for all
    /// n_layer*pp pages. Layer li owns physical pages [li*pp, (li+1)*pp)
    /// (contiguous KvStore allocation; kv_write phys = li*pp + lp), so the
    /// decode page table of every layer is the identity mapping li*pp + j:
    /// uploaded once at load, never touched per step (removes the per-layer
    /// table H2D uploads and their pinned staging — the former race notes
    /// in `upload_pages` no longer apply, the table is static).
    pages_dev: DeviceBuffer,
    lens_dev: DeviceBuffer,
    // f16 中间缓冲
    x: DeviceBuffer,
    xn: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn: DeviceBuffer,
    // S1-4: gate/up f16 staging buffers removed — fused_cast_swiglu_f16 reads
    // the f32 GEMM outputs (c_g/c_u) directly and writes f16 into `down`.
    down: DeviceBuffer,
    // 014 S0-3b: parity-f32 tier — dedicated f32 activation buffers. The
    // q/k/v/gate/up/down activations live in the existing f32 GEMM output
    // buffers (c_q..c_d); only the residual stream, the norm output and the
    // attention output need separate f32 storage (tiny — always allocated).
    x32: DeviceBuffer,
    xn32: DeviceBuffer,
    attn32: DeviceBuffer,
    // f32 GEMM 输出临时（m=1 → 行主序线性连续）
    c_q: DeviceBuffer,
    c_k: DeviceBuffer,
    c_v: DeviceBuffer,
    c_o: DeviceBuffer,
    c_g: DeviceBuffer,
    c_u: DeviceBuffer,
    c_d: DeviceBuffer,
    logits: DeviceBuffer,
    logits_host: HostBuffer,
    scores: DeviceBuffer,
    /// 006 T2: tune db (REINFER_TUNE_DB > XDG > HOME/.cache; the selector's
    /// data surface — measured scores are recorded here per successful run).
    db: TuneDb,
    /// 006 T2: process-stable selection cache ("first measure / second
    /// replay" semantics; same key space as the db).
    sel: SelectionCache,
    /// 006 T3E / S1-3: decode-step graph pool (REINFER_GRAPH, default off —
    /// opt-in; no-GPU slow path -> `GraphPool::disabled()` -> always eager).
    /// Captured execs are retained per bucket in `graph_execs` (Send
    /// wrapper — raw CUDA graph pointers) and served by replay when the
    /// refresh-safety guards allow; otherwise replay fails closed and the
    /// bucket falls back to eager (see `GraphStepDecl`).
    graph: GraphPool,
    /// 006 T3E: buckets whose capture failed (diagnostic once per bucket;
    /// no in-process retry of the same bucket, new buckets still attempt).
    graph_failed: HashSet<usize>,
    /// 006 T3E: total eager fallbacks served by the graph path (capture
    /// failures + any hypothetical replay absence; observable for tests).
    graph_eager_fallbacks: u64,
    /// 006-2 T2: decode-attn flash 档回退计数（smem 预算/launch 失败 →
    /// naive paged GQA 核；回退≠错误——选择链透明，诊断/测试可见）。
    decode_flash_fallbacks: u64,
    lens_hb: HostBuffer,
    /// 014 S0-3b: parity-f32 criterion tier switch (REINFER_PARITY_F32,
    /// default off — read once at load; when on, decode/attn intermediates
    /// run through the f32 channel, see `step_decode_launches_f32`).
    parity_f32: bool,
    /// S1-2: decode-step profiler (REINFER_DECODE_PROFILE, default off —
    /// inert unless armed; see `DecodeProfiler`).
    prof: DecodeProfiler,
    /// S1-7: prefill profiler (REINFER_PREFILL_PROFILE, default off — inert
    /// unless armed; see `PrefillProfiler`). cudaEvent marks at every prefill
    /// launch site, per-layer/per-kernel GPU attribution.
    prefill_prof: PrefillProfiler,
    /// S1-2: stable GEMM parameter grids for the decode step — every gemm1
    /// call is a fixed m/n/k + fixed buffer-pointer layout, prebuilt at load
    /// into `GemmPlan` cells (see `DecodeGemmPlans`). The decode step only
    /// executes plans; the S1-3 graph wave addresses these cells via
    /// `kernel_spec`/`PtrUpdate` without re-deriving parameters.
    plans: DecodeGemmPlans,
    /// S1-3: decode-step graph declaration (f16 channel only; `None` on the
    /// parity-f32 tier or when the graph pool is disabled). Built once at
    /// load — specs/cells/updates never move afterwards (the C3 cells are
    /// addressed by the constant replay update list).
    graph_decl: Option<GraphStepDecl>,
    /// S1-3: captured execs per bucket (bound after a successful capture,
    /// served by replay). `GraphExec` holds raw CUDA graph handles — the
    /// store is a Send wrapper with a device-bound Drop (serve.rs moves the
    /// Engine across threads).
    graph_execs: GraphExecStore,
    /// S1-3: successful capture + bucket-bind count (diagnostics/tests).
    graph_captures: u64,
    /// S1-3: successful replay count (diagnostics/tests).
    graph_replays: u64,
}

const BLOCK_LEN: usize = 32;
/// Env var for the decode-step graph integration (default off — fail-
/// closed until the cublas-node replay path is proven bitwise; on words:
/// "1"/"on"/"true"/"yes" — mirror of the graph.rs env convention).
const GRAPH_ENV: &str = "REINFER_GRAPH";
/// 014 S0-3b: parity-f32 criterion tier env (default **off** — the f16
/// channel stays the production path; on words mirror the graph convention).
const PARITY_F32_ENV: &str = "REINFER_PARITY_F32";
/// S1-9: fused decode-step env (default **on** — the fused kernels are the
/// production decode path; `REINFER_FUSED=off` keeps the split launch
/// sequence as the A/B reference arm). Off words mirror the opt-out
/// convention (like REINFER_DECODE_FLASH).
const FUSED_ENV: &str = "REINFER_FUSED";

/// REINFER_FUSED parsing: unset -> **on** (default); off words mirror the
/// opt-out convention (like REINFER_DECODE_FLASH).
#[must_use]
fn fused_decode_disabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
    }
}

/// REINFER_GRAPH parsing: unset -> **off** (default) — capture still runs a
/// probe attempt only when explicitly `on`; the decode-step replay is pending
/// the cublas-node KernelSpec work (see bench/notes T-305), and a failed
/// capture pollutes the current stream (eager launch fails afterwards), so
/// graph stays opt-in until the wiring is complete. Explicit on words also accepted.
#[must_use]
fn graph_enabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
        }
    }
}

/// REINFER_PARITY_F32 parsing: unset -> **off** (default). The parity-f32
/// criterion tier is opt-in by design — it changes the numerics of every
/// decode/prefill step (f32 intermediates) and loads f32 weight copies.
#[must_use]
fn parity_f32_enabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
        }
    }
}

/// 006-2 T2: decode-attn flash tier env (default **on** — the flash kernel is
/// the production decode path; `REINFER_DECODE_FLASH=off` selects the naive
/// paged GQA kernel directly). Unlike the opt-in gates above this one is
/// opt-out, so the A/B test surface (text/attn comparison vs the S1-5 naive
/// baseline) can drive the old tier without un-wiring the engine.
const DECODE_FLASH_ENV: &str = "REINFER_DECODE_FLASH";

/// REINFER_DECODE_FLASH parsing: unset -> **on** (default); off words mirror
/// the graph convention inverted ("0"/"off"/"false"/"no").
#[must_use]
fn decode_flash_disabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no")
        }
    }
}

/// JIT m=1 GEMM (JitGemm) env var (default **on** — the decode step's
/// m=1 projections run the `gemv_m1_f16f32` kernel instead of cublas;
/// `REINFER_JGEMM=off` restores the cublas path — the A/B surface and the
/// fallback when the kernel cannot be loaded).
const JGEMM_ENV: &str = "REINFER_JGEMM";

/// REINFER_JGEMM parsing: unset -> **on** (default; opt-out like
/// REINFER_DECODE_FLASH); off words mirror the flash convention.
#[must_use]
fn jgemm_disabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no")
        }
    }
}

// ---------------------------------------------------------------------------
// S1-2 decode-step profiler (REINFER_DECODE_PROFILE, default off).
//
// cudaEvent-timed segments around the decode step's phase groups, aggregated
// over 20 steps with mean per segment, plus per-step launch counts and the
// host-side wall time — the S1-1 attribution surface that drove the S1-2
// launch-count wave (host/launch overhead was the dominant cost, not GEMM
// bytes). Environment-gated: when off, all counters stay at their initial
// values and no timing/printing work is done (zero probe overhead).
//
// Segment categories (mirror the task's six groups):
//   SMALL    - RMSNorms, head norms, gather, rope, scales, casts (the many
//              small kernels that dominated the launch count)
//   QKV      - q/k/v GEMMs + their f32->f16 casts
//   ATTN     - kv_write + paged decode attention (+ scale on the f16 path)
//   O        - o-projection GEMM + residual add/cast
//   FFN      - gate/up/down GEMMs + casts + swiglu + residual add
//   LM_HEAD  - final RMSNorm + lm_head GEMM (+ softmax/logits copy)
//
// A segment may REOPEN (e.g. the ffn RMSNorm is attributed to SMALL while
// inside the FFN group): `end_segment` then records a boundary event; the
// per-step window for a segment = first start .. last end. Events are
// stream-ordered; the per-step timeline is closed by `finalize`, which
// synchronizes the stream once and folds every completed step's segments.
//
// Event records are ILLEGAL inside a CUDA-graph capture window — when the
// graph pool is active the profiler refuses to start (stays inert, so the
// REINFER_GRAPH probe path is unaffected). Bounded memory: at most 512
// events live on the device per window; beyond that the profiler deactivates
// itself with a note rather than growing.
// ---------------------------------------------------------------------------

/// REINFER_DECODE_PROFILE parsing: unset -> **off** (default). The decode
/// profiler is a diagnostic probe — never on unless explicitly requested
/// (same on-word convention as REINFER_GRAPH / REINFER_PARITY_F32).
#[must_use]
fn decode_profile_enabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
        }
    }
}

/// Decode-step profiler env var name (public for tests).
pub const DECODE_PROFILE_ENV: &str = "REINFER_DECODE_PROFILE";

const SEG_SMALL: u8 = 0;
const SEG_QKV: u8 = 1;
const SEG_ATTN: u8 = 2;
const SEG_O: u8 = 3;
/// FFN gate/up phase-1 (the gemv m=1 phase-1 of the gate and up plans).
const SEG_FFN_GU: u8 = 4;
/// FFN gate/up phase-2 + SiLU-GLU + down phase-1 (fused path) / the split
/// path's swiglu + down gemv pair.
const SEG_FFN_D: u8 = 5;
/// FFN down phase-2 + residual add + next-layer attn rms (fused path) / the
/// split path's ffn-norm/add_cast launches.
const SEG_FFN_RMS: u8 = 6;
const SEG_LM_HEAD: u8 = 7;
const SEG_COUNT: usize = 8;
const SEG_NAMES: [&str; SEG_COUNT] =
    ["small", "qkv", "attn", "o", "ffn_gu", "ffn_d", "ffn_rms", "lm_head"];

/// Mean of a segment across the aggregation window (ms/step).
#[derive(Debug, Clone, Copy)]
pub struct SegmentMean {
    /// Segment name (SEG_NAMES entry).
    pub name: &'static str,
    /// Mean GPU time per step attributed to this segment.
    pub ms: f32,
    /// Share of the total GPU busy time (all segments).
    pub share: f32,
    /// Launch count attributed to this segment (one per `count` call).
    pub launches: u32,
}

/// Aggregated decode-step profile after `finalize` (per-step means).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeProfile {
    /// Steps folded into this aggregate.
    pub steps: u32,
    /// Per-segment means (None for segments never opened).
    pub segments: [Option<SegmentMean>; SEG_COUNT],
    /// Host-side wall time per step (ms) — launch/sync overhead view.
    pub wall_ms: f32,
    /// Total GPU busy time per step (ms) = sum of segment means.
    pub gpu_ms: f32,
    /// Total kernel launches per step across all segments.
    pub launches: u32,
}

/// One closed [start, end) interval on the stream for a segment.
#[derive(Debug)]
struct Interval {
    start: CudaEvent,
    end: CudaEvent,
    seg: u8,
}

/// Per-step accumulation window for the profiler.
#[derive(Default, Debug)]
struct StepWindow {
    /// Closed intervals (start/end events already recorded, stream-ordered).
    intervals: Vec<Interval>,
    /// Whether a segment has an open start (end still to come).
    open: [bool; SEG_COUNT],
    /// The open start event of each segment (valid while `open`).
    open_ev: [Option<CudaEvent>; SEG_COUNT],
    /// Per-segment elapsed accumulators (sum over the step's intervals).
    seg_ms: [f32; SEG_COUNT],
    /// Per-segment launch counts.
    seg_launches: [u32; SEG_COUNT],
    /// Total launches this step.
    launches: u32,
}

/// S1-2 decode-step profiler (owned by Engine; inert unless env-enabled).
#[derive(Debug)]
pub struct DecodeProfiler {
    active: bool,
    dev: DeviceId,
    steps: u32,
    /// Window aggregate: mean-accumulated segment ms (reset every 20 steps).
    agg_seg_ms: [f32; SEG_COUNT],
    agg_seg_launches: [u32; SEG_COUNT],
    agg_launches: u32,
    agg_wall_ms: f32,
    /// Current step's boundaries (completed by `finalize`).
    cur: StepWindow,
    /// Event budget (bounded — deactivate on overflow).
    max_events: usize,
    /// Host-side wall clock for the current step.
    step_t0: Option<std::time::Instant>,
}

impl DecodeProfiler {
    /// Create an inert profiler (env off) or an armed one (env on).
    pub fn new(dev: DeviceId) -> Self {
        let active =
            decode_profile_enabled_from_env(std::env::var(DECODE_PROFILE_ENV).ok().as_deref());
        Self {
            active,
            dev,
            steps: 0,
            agg_seg_ms: [0.0; SEG_COUNT],
            agg_seg_launches: [0; SEG_COUNT],
            agg_launches: 0,
            agg_wall_ms: 0.0,
            cur: StepWindow::default(),
            max_events: 512,
            step_t0: None,
        }
    }

    /// Whether the profiler is armed (env-on and not deactivated).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Start the step window (host wall clock + first event). Called once
    /// per decode step, BEFORE any kernel of the step. No-op when off.
    pub fn begin_step(&mut self, stream: &CudaStream) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        self.step_t0 = Some(std::time::Instant::now());
        self.cur.launches = 0;
        self.cur.seg_ms = [0.0; SEG_COUNT];
        self.cur.seg_launches = [0; SEG_COUNT];
        self.cur.open = [false; SEG_COUNT];
        for slot in self.cur.open_ev.iter_mut() {
            *slot = None;
        }
        // Reuse the previous window's interval vector (events are dropped
        // here — their device lifetime ends at the end of finalize).
        self.cur.intervals.clear();
        self.cur.intervals.reserve(SEG_COUNT * 2 + 4);
        // A leading event pins the step's GPU start; used as the default
        // "unattributed" anchor only if a segment never closes (defensive).
        let ev = CudaEvent::new(self.dev)?;
        ev.record(stream)?;
        self.cur.open_ev[SEG_SMALL as usize] = Some(ev);
        self.cur.open[SEG_SMALL as usize] = true;
        Ok(())
    }

    /// Count one kernel launch attributed to `seg` (call site = each launch).
    /// Launch count is host-side; no events involved.
    #[inline]
    pub fn count(&mut self, seg: u8) {
        if !self.active {
            return;
        }
        self.cur.launches += 1;
        self.cur.seg_launches[seg as usize] += 1;
    }

    /// Open (or reopen) `seg` with a start boundary. `end_segment` must be
    /// called in stream order; callers may interleave segments freely.
    pub fn start_segment(&mut self, seg: u8, stream: &CudaStream) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        if self.cur.open[seg as usize] {
            return Ok(()); // already open — nested start is a no-op
        }
        if self.event_budget_exceeded(1) {
            self.deactivate_overflow();
            return Ok(());
        }
        let ev = CudaEvent::new(self.dev)?;
        ev.record(stream)?;
        self.cur.open_ev[seg as usize] = Some(ev);
        self.cur.open[seg as usize] = true;
        Ok(())
    }

    /// Close `seg` with an end boundary (stream-ordered). Pairs with the
    /// matching `start_segment`; a reopen produces multiple intervals.
    pub fn end_segment(&mut self, seg: u8, stream: &CudaStream) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        let start = match self.cur.open_ev[seg as usize].take() {
            Some(ev) => ev,
            None => return Ok(()), // unbalanced close — ignore
        };
        if self.event_budget_exceeded(1) {
            self.deactivate_overflow();
            return Ok(());
        }
        let ev = CudaEvent::new(self.dev)?;
        ev.record(stream)?;
        self.cur.open[seg as usize] = false;
        self.cur.intervals.push(Interval { start, end: ev, seg });
        Ok(())
    }

    /// End-of-step host wall clock (after the last kernel launch of the step;
    /// the GPU timeline is still in flight — `finalize` waits on the stream).
    pub fn end_wall(&mut self) {
        if !self.active {
            return;
        }
        if let Some(t0) = self.step_t0.take() {
            self.agg_wall_ms += t0.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// Close the current step: synchronize the stream once, fold the event
    /// deltas into the aggregation window, and print the 20-step mean table.
    /// Called once per step, after the step's kernels are queued.
    pub fn finalize(&mut self, stream: &CudaStream) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        // Close any dangling segments (defensive — callers keep pairs).
        for seg in 0..SEG_COUNT {
            if let Some(start) = self.cur.open_ev[seg].take() {
                if !self.event_budget_exceeded(1) {
                    let ev = CudaEvent::new(self.dev)?;
                    ev.record(stream)?;
                    self.cur.intervals.push(Interval { start, end: ev, seg: seg as u8 });
                }
                self.cur.open[seg] = false;
            }
        }
        // One synchronize per step — the events are stream-ordered, so the
        // stream's completion implies every recorded event completed.
        stream.synchronize()?;
        // Fold intervals into the 20-step aggregation window.
        for iv in self.cur.intervals.iter() {
            let ms = iv.start.elapsed_ms(&iv.end)?;
            self.agg_seg_ms[iv.seg as usize] += ms;
        }
        for seg in 0..SEG_COUNT {
            self.agg_seg_launches[seg] += self.cur.seg_launches[seg];
        }
        self.agg_launches += self.cur.launches;
        self.steps += 1;
        if self.steps.is_multiple_of(20) {
            self.print_table();
        }
        self.cur.intervals.clear();
        Ok(())
    }

    /// Event-budget guard: `pending` events still to create. Bounded memory —
    /// a pathological step must not grow the pool unboundedly.
    fn event_budget_exceeded(&self, pending: usize) -> bool {
        let live = self.cur.intervals.len() * 2
            + self.cur.open_ev.iter().filter(|e| e.is_some()).count()
            + pending;
        live > self.max_events
    }

    fn deactivate_overflow(&mut self) {
        self.active = false;
        eprintln!(
            "[reinfer-cuda] REINFER_DECODE_PROFILE: event budget exceeded \
             (>{max} events/step) — profiler disabled for the remaining steps",
            max = self.max_events
        );
    }

    fn print_table(&mut self) {
        let n = 20;
        let nf = n as f32;
        println!("[reinfer-cuda] decode profile (mean over {} steps):", n);
        let mut gpu_ms = 0.0;
        let mut rows = [SegmentMean { name: "", ms: 0.0, share: 0.0, launches: 0 }; SEG_COUNT];
        for seg in 0..SEG_COUNT {
            let ms = self.agg_seg_ms[seg] / nf;
            rows[seg] = SegmentMean {
                name: SEG_NAMES[seg],
                ms,
                share: 0.0,
                launches: self.agg_seg_launches[seg] / n,
            };
            gpu_ms += ms;
        }
        for r in rows.iter_mut() {
            r.share = if gpu_ms > 0.0 { r.ms / gpu_ms * 100.0 } else { 0.0 };
        }
        let mut sorted = rows;
        sorted.sort_by(|a, b| b.ms.total_cmp(&a.ms));
        for r in sorted.iter() {
            println!(
                "  {:>8} {ms:8.3} ms  {share:5.1}%  {:>3} launches",
                r.name,
                r.launches,
                ms = r.ms,
                share = r.share,
            );
        }
        println!(
            "  gpu busy {gpu_ms:.3} ms/step, {launch} launches/step, \
             host wall {wall:.3} ms/step",
            gpu_ms = gpu_ms,
            launch = self.agg_launches / n,
            wall = self.agg_wall_ms / nf
        );
        // Reset the aggregation window.
        self.agg_seg_ms = [0.0; SEG_COUNT];
        self.agg_seg_launches = [0; SEG_COUNT];
        self.agg_launches = 0;
        self.agg_wall_ms = 0.0;
    }
}

impl Default for DecodeProfiler {
    fn default() -> Self {
        // Inert — Engine::load arms it from the env (tests get the inert
        // default unless they set the env explicitly).
        Self {
            active: false,
            dev: DeviceId::new(0),
            steps: 0,
            agg_seg_ms: [0.0; SEG_COUNT],
            agg_seg_launches: [0; SEG_COUNT],
            agg_launches: 0,
            agg_wall_ms: 0.0,
            cur: StepWindow::default(),
            max_events: 512,
            step_t0: None,
        }
    }
}

// ---------------------------------------------------------------------------
// S1-2 stable GEMM parameter grids (graph prelude).
//
// Every gemm1/gemm1r call in the decode/prefill paths is a fixed m/n/k with
// fixed buffer pointers — prebuild those into `GemmPlan` cells at load so the
// step bodies only execute plans (`Gemm::execute`). Cells are addressed by
// stable device pointers (buffers never move after load), which is exactly
// the staging-seed pattern graph.rs needs (`KernelSpec`/`PtrUpdate` — S1-3
// wave). Numerics are untouched: `execute` expands the plan into the same
// cublasGemmEx arguments the old gemm1/gemm1r built inline.
// ---------------------------------------------------------------------------

/// Decode-step GEMM plans for one layer (f16 or f32 channel per `parity_f32`).
#[derive(Debug, Clone, Copy)]
pub struct LayerGemmPlans {
    /// q projection: [1 x nqk] = xn x W_q.
    pub q: GemmPlan,
    /// k projection: [1 x kvk] = xn x W_k.
    pub k: GemmPlan,
    /// v projection: [1 x kvk] = xn x W_v.
    pub v: GemmPlan,
    /// o projection: [1 x h] = attn x W_o.
    pub o: GemmPlan,
    /// gate projection: [1 x ffn] = xn x W_gate.
    pub gate: GemmPlan,
    /// up projection: [1 x ffn] = xn x W_up.
    pub up: GemmPlan,
    /// down projection: [1 x h] = down x W_down.
    pub down: GemmPlan,
}

/// Decode-step GEMM plan grid: per-layer plans + lm_head (built once at load;
/// `layers[li]` addresses the li-th layer's stable cells).
#[derive(Debug, Clone)]
pub struct DecodeGemmPlans {
    /// Per-layer cells, indexed by layer id.
    pub layers: Vec<LayerGemmPlans>,
    /// Final lm_head: [1 x vocab] = xn x W_lm_head.
    pub lm_head: GemmPlan,
}

impl DecodeGemmPlans {
    /// Build the decode grid for `eng`. Each plan captures the exact
    /// A/B/C buffers + m/n/k the step body would use, for the loaded channel:
    ///
    /// - f16 channel (production): A = `xn` (or `attn` for o, `down` for the
    ///   down projection), C = `c_q..c_d` — matches the old `gemm1` calls.
    /// - parity-f32 channel: A = `xn32` (or `attn32` for o, `c_g` for the
    ///   down projection — swiglu is in-place on gate), C = `c_q..c_o` —
    ///   matches the old `gemm1_32f` calls.
    ///
    /// `execute` expands each plan into the same cublasGemmEx arguments the
    /// old wrappers built inline, so numerics are bit-identical.
    fn build(eng: &Engine) -> Self {
        let h = eng.cfg.hidden_size;
        let nqk = eng.cfg.q_heads * eng.cfg.head_dim;
        let kvk = eng.cfg.kv_heads * eng.cfg.head_dim;
        let ffn = eng.cfg.ffn_hidden;
        let vocab = eng.cfg.vocab_size;
        let layers = if eng.parity_f32 {
            let rm = |a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize| {
                GemmPlan::row_major_f32(a, b, c, m, n, k)
            };
            let xn = eng.xn32.as_ptr() as *const f32;
            eng.layers
                .iter()
                .map(|w| LayerGemmPlans {
                    q: rm(
                        xn,
                        w.q_proj.as_ptr() as *const f32,
                        eng.c_q.as_ptr() as *mut f32,
                        1,
                        nqk,
                        h,
                    ),
                    k: rm(
                        xn,
                        w.k_proj.as_ptr() as *const f32,
                        eng.c_k.as_ptr() as *mut f32,
                        1,
                        kvk,
                        h,
                    ),
                    v: rm(
                        xn,
                        w.v_proj.as_ptr() as *const f32,
                        eng.c_v.as_ptr() as *mut f32,
                        1,
                        kvk,
                        h,
                    ),
                    o: rm(
                        eng.attn32.as_ptr() as *const f32,
                        w.o_proj.as_ptr() as *const f32,
                        eng.c_o.as_ptr() as *mut f32,
                        1,
                        h,
                        nqk,
                    ),
                    gate: rm(
                        xn,
                        w.gate_proj.as_ptr() as *const f32,
                        eng.c_g.as_ptr() as *mut f32,
                        1,
                        ffn,
                        h,
                    ),
                    up: rm(
                        xn,
                        w.up_proj.as_ptr() as *const f32,
                        eng.c_u.as_ptr() as *mut f32,
                        1,
                        ffn,
                        h,
                    ),
                    down: rm(
                        eng.c_g.as_ptr() as *const f32,
                        w.down_proj.as_ptr() as *const f32,
                        eng.c_o.as_ptr() as *mut f32,
                        1,
                        h,
                        ffn,
                    ),
                })
                .collect()
        } else {
            let rm = |a: *const u16, b: *const u16, c: *mut f32, m: usize, n: usize, k: usize| {
                GemmPlan::row_major_f16(a, b, c, m, n, k)
            };
            let xn = eng.xn.as_ptr() as *const u16;
            eng.layers
                .iter()
                .map(|w| LayerGemmPlans {
                    q: rm(
                        xn,
                        w.q_proj.as_ptr() as *const u16,
                        eng.c_q.as_ptr() as *mut f32,
                        1,
                        nqk,
                        h,
                    ),
                    k: rm(
                        xn,
                        w.k_proj.as_ptr() as *const u16,
                        eng.c_k.as_ptr() as *mut f32,
                        1,
                        kvk,
                        h,
                    ),
                    v: rm(
                        xn,
                        w.v_proj.as_ptr() as *const u16,
                        eng.c_v.as_ptr() as *mut f32,
                        1,
                        kvk,
                        h,
                    ),
                    o: rm(
                        eng.attn.as_ptr() as *const u16,
                        w.o_proj.as_ptr() as *const u16,
                        eng.c_o.as_ptr() as *mut f32,
                        1,
                        h,
                        nqk,
                    ),
                    gate: rm(
                        xn,
                        w.gate_proj.as_ptr() as *const u16,
                        eng.c_g.as_ptr() as *mut f32,
                        1,
                        ffn,
                        h,
                    ),
                    up: rm(
                        xn,
                        w.up_proj.as_ptr() as *const u16,
                        eng.c_u.as_ptr() as *mut f32,
                        1,
                        ffn,
                        h,
                    ),
                    down: rm(
                        eng.down.as_ptr() as *const u16,
                        w.down_proj.as_ptr() as *const u16,
                        eng.c_d.as_ptr() as *mut f32,
                        1,
                        h,
                        ffn,
                    ),
                })
                .collect()
        };
        let lm_head = if eng.parity_f32 {
            GemmPlan::row_major_f32(
                eng.xn32.as_ptr() as *const f32,
                eng.lm_head.as_ptr() as *const f32,
                eng.logits.as_ptr() as *mut f32,
                1,
                vocab,
                h,
            )
        } else {
            GemmPlan::row_major_f16(
                eng.xn.as_ptr() as *const u16,
                eng.lm_head.as_ptr() as *const u16,
                eng.logits.as_ptr() as *mut f32,
                1,
                vocab,
                h,
            )
        };
        Self { layers, lm_head }
    }
}

/// Expected kernel-node count of the f16 decode step (mirror of
/// `step_decode_launches`): gather + per-layer [rms_attn, q/k/v gemms and
/// casts, head norms (when `head_norm`), rope q/k, kv_write, decode,
/// gemm o, fused_add_rms (o residual + ffn norm, S1-4), gate/up gemms,
/// fused_cast_swiglu (S1-4), gemm down, add_cast] + final rms + lm_head
/// = 20 (18 without head norm) per layer, plus the 3 bookend nodes. The
/// declaration self-check compares against this; a divergence would fail
/// every capture closed.
/// Declared spec count for the f16 decode step (`GraphStepDecl::build`'s
/// launch-order self-check; Graph V2): the gather, per-layer small
/// kernels (13 with head_norm / 11 without) plus the m=1 projection
/// GEMMs (7 per layer + lm_head), each counted TWICE when the JitGemm
/// path is loaded — the jgemm phase pair (`gemv_m1_f16f32` +
/// `gemv_m1_f16f32_reduce`). Every decode plan matches `Jgemm::matches`,
/// so the formula is all-or-none on the jgemm load; a non-matching plan
/// would only trigger the (diagnostic) self-check warning, and the
/// capture node-count guard remains the hard boundary.
const fn expected_node_count(
    n_layers: usize,
    head_norm: bool,
    jgemm: bool,
    fused: bool,
) -> usize {
    if fused {
        // S1-9: gather + rms_attn(0) + per-layer 8 fused nodes (p1_qkv,
        // p2_qkv, flash_fused [kv write + attention], p1_o, p2_o, p1_gu,
        // p2_gu_d [swiglu + down phase-1], p2_down) + the lm_head phase
        // pair (the last layer's p2_add_rms computes the final norm — no
        // separate final-rms node).
        return 4 + 8 * n_layers;
    }
    let per_layer = if head_norm { 13 } else { 11 } + if jgemm { 14 } else { 7 };
    // gather + per-layer + final rms_norm + lm_head gemm(s)
    2 + per_layer * n_layers + if jgemm { 2 } else { 1 }
}

/// Cell word for a device pointer (C3 discipline: the cell *holds* the
/// pointer value; the graph node's kernelParams point at the cell and the
/// driver reads its current value at every replay launch).
fn cell_ptr<T>(p: *const T) -> u64 {
    p as usize as u64
}

// ---------------------------------------------------------------------------
// S1-3 decode-step graph declaration
// ---------------------------------------------------------------------------

/// S1-3 / Graph V2: the decode step's CUDA-graph declaration — the
/// `KernelSpec` list for the whole step (an exact launch-order mirror of
/// `step_decode_launches`), plus the stable C3 argument cells the graph
/// reads at every replay launch.
///
/// Built once at load for the f16 channel (the parity-f32 tier stays
/// eager — its launch sequence differs). Every kernel node is declared as
/// a `CustomKernel` (the Graph V2 unlock — no cublas node is left in the
/// step, so the 13.x V2 read-back is never needed and every node is
/// refresh-safe by construction):
///
/// - small fused kernels (`Fixed{slots}`, `CUkernel` handle, FULL
///   pointer-slot coverage — refresh rewrites the whole kernelParams
///   array, so a partial coverage would launch null cells);
/// - each m=1 projection GEMM as the TWO jgemm phase kernels when the
///   JitGemm path is loaded (`gemv_m1_f16f32` phase 1: 6 slots
///   (a, b, partials, n, k, nslabs), `gemv_m1_f16f32_reduce` phase 2:
///   4 slots (partials, c, n, nslabs)) — the launch-order mirror of
///   `Jgemm::launch`, including the slab-partials scratch pointer
///   pre-ensured (stable) before capture and the per-plan nslabs
///   geometry; grid/block match the phase launches;
/// - cublas gemm nodes (only when JitGemm is off) as
///   `Gemm{slots: 22, m: 0, n: 1, k: 2}` with the m/n/k value cells at
///   slots 0/1/2 — handle/geometry recovered by the post-capture
///   read-back; such nodes are not refresh-safe on a V2-only (13.x)
///   read-back, so replay fails closed (the REINFER_JGEMM=off boundary).
///
/// Cells are one u64 per declared slot, node-major. A `PtrUpdate.ptr` is
/// the cell *address*; the driver dereferences it at every launch (C3),
/// so per-step variable changes (token/pos/phys/off) are plain cell
/// stores and the update list itself is constant — built once, never
/// rebuilt. The captured lens H2D memcpy node re-reads the pinned
/// `lens_hb` at replay; the engine writes it host-side per step.
#[derive(Debug)]
pub struct GraphStepDecl {
    /// Kernel specs in launch order (== capture node order).
    specs: Vec<KernelSpec>,
    /// `CUfunction` launch handles, aligned with `specs` (the decl-driven
    /// launch's handles: the specs carry the `CUkernel` form the capture
    /// records, but launching requires the `cuKernelGetFunction`-converted
    /// `CUfunction` — direct CUkernel cast segfaults; converted once at
    /// build so the per-step decl-driven launch is one launch per node).
    /// Null for `CublasGemm` specs (the decl-driven path never runs with
    /// cublas nodes — the REINFER_JGEMM=off boundary).
    launch_fns: Vec<dsys::CUfunction>,
    /// Per-node offset into `cells` (node-major layout).
    cell_off: Vec<usize>,
    /// 8-byte argument cells: `cells[cell_off[node] + slot]` is the
    /// argument value of the node's `slot`-th declared slot.
    cells: Vec<u64>,
    /// Stable per-node kernelParams arrays (flat, aligned with `cells`:
    /// `&arg_slots[cell_off[node]]` is the node's args array, `slots`
    /// entries, entry i = the address of `cells[cell_off[node] + i]`).
    /// The driver records the kernelParams array *pointer* (no copy — the
    /// measured 595.84/CUDA 13.2 capture keeps the caller's array live), so
    /// the arrays must outlive every replay; these are owned by the decl
    /// and never move (built after `cells` is final).
    arg_slots: Vec<*mut c_void>,
    /// Constant replay update list (one `PtrUpdate` per declared slot,
    /// pointing at the cell addresses — built last, so the `cells` Vec
    /// never moves afterwards).
    updates: Vec<PtrUpdate>,
    /// Cell index of the gather `row` argument (per-step token write).
    cell_token: usize,
    /// Cell indices of the rope q/k `pos` arguments (per layer).
    cell_rope_q_pos: Vec<usize>,
    cell_rope_k_pos: Vec<usize>,
    /// Cell indices of the kv_write `phys`/`off` arguments (per layer).
    cell_kv_phys: Vec<usize>,
    cell_kv_off: Vec<usize>,
    /// Per-step content-refresh slots (Graph V2): (node, slot) pairs whose
    /// cells `write_step` updates — gather token, rope q/k pos, kv_write
    /// phys/off (85 nodes for Qwen3-0.6B). The 13.2 driver bakes kernel
    /// node params at capture (scalar args frozen; only pointer targets and
    /// captured memcpy nodes are re-read at replay), so every replay must
    /// re-refresh these nodes via the V2 SetParams path — the baked values
    /// are re-read from the cells at set time.
    refresh: Vec<(usize, usize)>,
}

// The declaration is plain data: raw handles are used only as values by
// the graph refresh, the cells are only ever written by `write_step`, and
// the update list's cell addresses stay valid (the `cells` Vec never
// moves after build). Moving it between threads is sound.
unsafe impl Send for GraphStepDecl {}

/// Spec/cell accumulator for [`GraphStepDecl::build`] — node-major cells.
#[derive(Debug, Default)]
struct SpecAcc {
    specs: Vec<KernelSpec>,
    cell_off: Vec<usize>,
    cells: Vec<u64>,
}

impl SpecAcc {
    /// Custom-kernel node with explicit dynamic shared memory: `vals`
    /// covers every slot of the layout (full pointer-slot coverage — see
    /// `GraphStepDecl`). The flash decode kernel launches with
    /// `(d + max_kv) * 4` bytes of dynamic smem (Graph V2: the declared
    /// geometry must mirror the eager launch exactly, because refresh
    /// rewrites grid/block/shared from the declaration).
    fn custom_with_shared(
        &mut self,
        handle: *mut c_void,
        slots: usize,
        grid: cudarc::runtime::sys::dim3,
        block: cudarc::runtime::sys::dim3,
        shared: u32,
        vals: &[u64],
    ) {
        debug_assert_eq!(slots, vals.len());
        self.cell_off.push(self.cells.len());
        self.cells.extend_from_slice(vals);
        self.specs.push(KernelSpec {
            role: NodeRole::CustomKernel,
            layout: ParamLayout::Fixed { slots },
            ptr_slots: (0..slots).map(|i| (i, PtrRole::Pointer)).collect(),
            handle,
            grid,
            block,
            shared,
        });
    }

    /// Custom-kernel node without dynamic shared memory (the common
    /// case — every small fused kernel and the jgemm phase pair).
    fn custom(
        &mut self,
        handle: *mut c_void,
        slots: usize,
        grid: cudarc::runtime::sys::dim3,
        block: cudarc::runtime::sys::dim3,
        vals: &[u64],
    ) {
        self.custom_with_shared(handle, slots, grid, block, 0, vals);
    }

    /// Cublas GEMM node: m/n/k value cells at the layout's geometry slots
    /// 0/1/2. grid/block/handle are recovered by the post-capture
    /// read-back; the declaration is the count/order anchor.
    fn gemm(&mut self, plan: &GemmPlan) {
        self.cell_off.push(self.cells.len());
        self.cells.extend_from_slice(&[
            (plan.m as u32) as u64,
            (plan.n as u32) as u64,
            (plan.k as u32) as u64,
        ]);
        self.specs.push(plan.kernel_spec(vec![
            (0, PtrRole::Pointer),
            (1, PtrRole::Pointer),
            (2, PtrRole::Pointer),
        ]));
    }

    /// Absolute cell index of the current (last-pushed) node's `slot`.
    fn cell_of(&self, slot: usize) -> usize {
        self.cell_off[self.specs.len() - 1] + slot
    }
}

impl GraphStepDecl {
    /// Kernel specs (launch order).
    #[must_use]
    pub fn specs(&self) -> &[KernelSpec] {
        &self.specs
    }

    /// Constant replay update list.
    #[must_use]
    pub fn updates(&self) -> &[PtrUpdate] {
        &self.updates
    }

    /// Per-step content-refresh slots (node, slot) — see the struct docs.
    /// The engine passes this to the capture so every replay re-refreshes
    /// exactly these nodes (the 13.2 driver bakes kernel params at
    /// capture; scalars must be re-baked per step).
    #[must_use]
    pub fn refresh(&self) -> &[(usize, usize)] {
        &self.refresh
    }

    /// Declared m/n/k values of a `CublasGemm` node (`None` for custom
    /// nodes) — the declared side of the post-capture alignment check.
    fn gemm_cells(&self, node: usize) -> Option<(u64, u64, u64)> {
        let spec = self.specs.get(node)?;
        let ParamLayout::Gemm { m, n, k, .. } = spec.layout else {
            return None;
        };
        let base = self.cell_off[node];
        Some((self.cells[base + m], self.cells[base + n], self.cells[base + k]))
    }

    /// Write the per-step variables into the stable cells (token/pos/
    /// phys/off). Called before every replay — the graph nodes read the
    /// cell values at launch (C3 discipline), so no SetParams is involved.
    fn write_step(&mut self, token: u32, pos: usize, _n_layers: usize, pp: usize) {
        let lp = pos / BLOCK_LEN;
        let off = pos % BLOCK_LEN;
        self.cells[self.cell_token] = token as u64;
        for li in 0..self.cell_rope_q_pos.len() {
            self.cells[self.cell_rope_q_pos[li]] = pos as u64;
            self.cells[self.cell_rope_k_pos[li]] = pos as u64;
        }
        // kv_write phys/off cells exist only on the split declaration —
        // the fused declaration folds the kv write into the fused flash
        // node, whose slot is derived from kv_len inside the kernel (no
        // per-step cells; the zip loop is a no-op when the vecs are
        // empty).
        for (li, (&c_phys, &c_off)) in
            self.cell_kv_phys.iter().zip(&self.cell_kv_off).enumerate()
        {
            let phys = (li * pp + lp) as u32;
            self.cells[c_phys] = phys as u64;
            self.cells[c_off] = off as u64;
        }
    }

    /// Build the declaration for `eng`'s decode step (f16 channel).
    ///
    /// Exact launch-order mirror of `step_decode_launches` (f16 path):
    /// gather, then per layer [rms_attn, gemm q, cast q, gemm k, cast k,
    /// gemm v, cast v, rms_heads q/k (when `head_norm`), rope_heads q/k,
    /// kv_write, decode_step, gemm o, fused_add_rms (o residual + ffn
    /// norm, S1-4), gemm gate, gemm up, fused_cast_swiglu (S1-4), gemm
    /// down, add_cast], final rms_norm, gemm lm_head. The caller must
    /// never reorder: capture
    /// validates the node count only, and the cublas read-back alignment
    /// assumes declaration order.
    ///
    /// Graph V2 (JitGemm loaded): every m=1 projection GEMM is declared
    /// as the two jgemm phase kernels — the mirror of `Jgemm::launch`,
    /// including the shared slab-partials scratch (pre-ensured for every
    /// plan here, so no allocation can happen inside a capture window,
    /// and the post-growth pointer is stable) and the per-plan nslabs
    /// geometry. The flash decode node declares FLASH_TPB = 512 threads
    /// with `(d + max_kv) * 4` bytes dynamic smem — the actual launch
    /// geometry (refresh rewrites the geometry from the declaration).
    fn build(eng: &Engine) -> Result<Self, LaunchError> {
        // S1-9: the fused kernels loaded -> the declaration mirrors the
        // fused 7-node sequence (the exact mirror of
        // `step_decode_launches_fused`).
        if eng.fused.is_some() {
            return Self::build_fused(eng);
        }
        let cfg = &eng.cfg;
        let h = cfg.hidden_size;
        let nqk = cfg.q_heads * cfg.head_dim;
        let kvk = cfg.kv_heads * cfg.head_dim;
        let ffn = cfg.ffn_hidden;
        let d = cfg.head_dim;
        let kv_heads = cfg.kv_heads;
        let q_heads = cfg.q_heads;
        let ratio = q_heads / kv_heads;
        let total_pages = cfg.n_layer * eng.pp;
        let eps = cfg.rms_eps;
        let eta = cfg.rope_theta;
        let n_layers = eng.layers.len();

        // CUkernel handles — the handle form the capture records
        // (`cudaKernelFunctionTypeKernel`; see graph.rs `FN_TYPE`).
        let cu = |lib: cudarc::driver::sys::CUlibrary,
                  name: &str|
         -> Result<*mut c_void, LaunchError> { cu_kernel_of(lib, name) };
        let dense = eng.kernels.raw_lib();
        let diff = eng.diff.raw_lib();
        // 006-2 T2: the eager step launches the flash decode kernel — the
        // declaration must mirror it (capture records the launched kernel).
        let flash_decode = eng.decode.flash_raw_lib();
        let h_gather = cu(dense, "gather_row")?;
        let h_rms = cu(dense, "rms_norm_row_f16")?;
        let h_rms_heads = cu(dense, "rms_norm_heads_f16")?;
        let h_rope_heads = cu(dense, "rope_heads_f16")?;
        let h_add_cast = cu(dense, "add_cast_f16")?;
        let h_add_rms = cu(dense, "fused_add_rms_f16")?;
        let h_cast_swiglu = cu(dense, "fused_cast_swiglu_f16")?;
        let h_kv_write = cu(dense, "kv_write_row")?;
        let h_cast = cu(diff, "cast_f32_to_f16")?;
        let h_decode = cu(flash_decode, "decode_step_gqa_flash")?;

        let b256 = cudarc::runtime::sys::dim3 { x: 256, y: 1, z: 1 };
        // Flash decode launches with FLASH_TPB = 512 threads (decode.rs).
        let b512 = cudarc::runtime::sys::dim3 { x: 512, y: 1, z: 1 };
        let g1 = cudarc::runtime::sys::dim3 { x: 1, y: 1, z: 1 };
        let g = |x: u32| cudarc::runtime::sys::dim3 { x, y: 1, z: 1 };

        let mut acc = SpecAcc::default();
        // gather: (src=embed, dst=x, row=token, n=h) — token is per-step.
        acc.custom(
            h_gather,
            4,
            g(h.div_ceil(256) as u32),
            b256,
            &[cell_ptr(eng.embed.as_ptr()), cell_ptr(eng.x.as_ptr()), 0, h as u64],
        );
        // Graph V2 per-step refresh list: (node, slot) pairs whose cells
        // `write_step` updates (gather token, rope q/k pos, kv_write
        // phys/off). The 13.2 driver bakes kernel-node params at capture —
        // scalar args are frozen, only pointer targets and captured memcpy
        // nodes are re-read at replay — so every replay re-refreshes these
        // nodes' params from the cells (V2 SetParams; see
        // `GraphExec::refresh` in graph.rs).
        let mut refresh: Vec<(usize, usize)> = vec![(acc.specs.len() - 1, 2)];
        let cell_token = acc.cell_of(2);
        let mut cell_rope_q_pos = Vec::with_capacity(n_layers);
        let mut cell_rope_k_pos = Vec::with_capacity(n_layers);
        let mut cell_kv_phys = Vec::with_capacity(n_layers);
        let mut cell_kv_off = Vec::with_capacity(n_layers);

        let eps_bits = eps.to_bits() as u64;
        let eta_bits = eta.to_bits() as u64;
        let qs_bits = (1.0 / (d as f32).sqrt()).to_bits() as u64;
        let one_bits = 1.0f32.to_bits() as u64;
        let x_ptr = eng.x.as_ptr();
        let xn_ptr = eng.xn.as_ptr();
        let q_ptr = eng.q.as_ptr();
        let k_ptr = eng.k.as_ptr();
        let v_ptr = eng.v.as_ptr();
        let down_ptr = eng.down.as_ptr();
        let attn_ptr = eng.attn.as_ptr();
        let kv_ptr = eng.kv.data.as_ptr();
        let lens_ptr = eng.lens_dev.as_ptr();
        let _scores_ptr = eng.scores.as_ptr(); // naive-fallback scratch (flash uses smem)
        let pages_ptr = eng.pages_dev.as_ptr() as *const u32;
        // Token-capacity bound for the flash decode kernel (smem sizing +
        // in-kernel guard): pp * BLOCK_LEN >= config max_kv by construction.
        let max_kv = (eng.pp * BLOCK_LEN) as u32;

        // Graph V2: the m=1 decode GEMMs launch as the jgemm phase pair
        // when the JitGemm path is loaded. Pre-ensure the shared
        // slab-partials scratch for EVERY plan BEFORE capture (cudaMalloc
        // inside a capture window is illegal — the eager path's lazy
        // ensure would otherwise allocate mid-capture), then read the
        // stable pointer AFTER all ensures (the buffer grows
        // monotonically; the post-growth pointer is the one every captured
        // launch records). `(partials cell, phase1 CUkernel, phase2
        // CUkernel)`.
        let jgemm: Option<(u64, *mut c_void, *mut c_void)> = match eng.jgemm.as_ref() {
            Some(jg) => {
                let mut plans: Vec<&GemmPlan> = Vec::with_capacity(7 * n_layers + 1);
                for l in &eng.plans.layers {
                    for p in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
                        if jg.matches(p) {
                            plans.push(p);
                        }
                    }
                }
                if jg.matches(&eng.plans.lm_head) {
                    plans.push(&eng.plans.lm_head);
                }
                for p in &plans {
                    let (_, nslabs) = jg.shape(p.n, p.k);
                    jg.ensure_partials(p.n as usize, nslabs)?;
                }
                // No-op ensure for the largest need: returns the final
                // (post-growth) buffer pointer, stable for all plans.
                let (max_n, max_nslabs) = plans
                    .iter()
                    .map(|p| {
                        let (_, nslabs) = jg.shape(p.n, p.k);
                        (p.n as usize, nslabs)
                    })
                    .max_by_key(|(n, s)| n * *s as usize)
                    .expect("jgemm plans non-empty (all decode plans are m=1 f16)");
                let partials = jg.ensure_partials(max_n, max_nslabs)?;
                let (phase1, phase2) = jg.kernel_handles()?;
                Some((cell_ptr(partials), phase1, phase2))
            }
            None => None,
        };
        // One projection GEMM's declaration: the jgemm phase pair when the
        // JitGemm path covers the plan (phase 1 `gemv_m1_f16f32` — 6 slots
        // (a, b, partials, n, k, nslabs), grid = ncols*nslabs; phase 2
        // `gemv_m1_f16f32_reduce` — 4 slots (partials, c, n, nslabs),
        // grid = ncols — the exact mirror of `Jgemm::launch`), else the
        // cublas node (replay fails closed on the V2 read-back — the
        // REINFER_JGEMM=off boundary).
        let declare_gemm = |acc: &mut SpecAcc, pl: &GemmPlan| {
            match (jgemm, eng.jgemm.as_ref()) {
                (Some((partials, phase1, phase2)), Some(jg)) if jg.matches(pl) => {
                    let (ncols, nslabs) = jg.shape(pl.n, pl.k);
                    acc.custom(
                        phase1,
                        6,
                        g(ncols * nslabs),
                        b256,
                        &[
                            cell_ptr(pl.a),
                            cell_ptr(pl.b),
                            partials,
                            pl.n as u64,
                            pl.k as u64,
                            nslabs as u64,
                        ],
                    );
                    acc.custom(
                        phase2,
                        4,
                        g(ncols),
                        b256,
                        &[partials, cell_ptr(pl.c), pl.n as u64, nslabs as u64],
                    );
                }
                _ => acc.gemm(pl),
            }
        };

        for li in 0..n_layers {
            let w = &eng.layers[li];
            let pl = &eng.plans.layers[li];
            // attn rms_norm (single block 256)
            acc.custom(
                h_rms,
                5,
                g1,
                b256,
                &[
                    cell_ptr(x_ptr),
                    cell_ptr(xn_ptr),
                    cell_ptr(w.attn_norm.as_ptr()),
                    h as u64,
                    eps_bits,
                ],
            );
            // q/k/v projections + f32->f16 casts
            declare_gemm(&mut acc, &pl.q);
            acc.custom(
                h_cast,
                3,
                g(nqk.div_ceil(256) as u32),
                b256,
                &[cell_ptr(eng.c_q.as_ptr()), cell_ptr(q_ptr), nqk as u64],
            );
            declare_gemm(&mut acc, &pl.k);
            acc.custom(
                h_cast,
                3,
                g(kvk.div_ceil(256) as u32),
                b256,
                &[cell_ptr(eng.c_k.as_ptr()), cell_ptr(k_ptr), kvk as u64],
            );
            declare_gemm(&mut acc, &pl.v);
            acc.custom(
                h_cast,
                3,
                g(kvk.div_ceil(256) as u32),
                b256,
                &[cell_ptr(eng.c_v.as_ptr()), cell_ptr(v_ptr), kvk as u64],
            );
            // q/k head norms (grid = heads; in-place x == out)
            if let (Some(qn), Some(kn)) = (&w.q_norm, &w.k_norm) {
                acc.custom(
                    h_rms_heads,
                    6,
                    g(q_heads as u32),
                    b256,
                    &[
                        cell_ptr(q_ptr),
                        cell_ptr(q_ptr),
                        cell_ptr(qn.as_ptr()),
                        q_heads as u64,
                        d as u64,
                        eps_bits,
                    ],
                );
                acc.custom(
                    h_rms_heads,
                    6,
                    g(kv_heads as u32),
                    b256,
                    &[
                        cell_ptr(k_ptr),
                        cell_ptr(k_ptr),
                        cell_ptr(kn.as_ptr()),
                        kv_heads as u64,
                        d as u64,
                        eps_bits,
                    ],
                );
            }
            // rope q/k (pos is per-step; scale folded into the q pass)
            acc.custom(
                h_rope_heads,
                6,
                g(q_heads as u32),
                b256,
                &[cell_ptr(q_ptr), q_heads as u64, (d / 2) as u64, 0, eta_bits, qs_bits],
            );
            refresh.push((acc.specs.len() - 1, 3)); // rope q pos (per-step)
            cell_rope_q_pos.push(acc.cell_of(3));
            acc.custom(
                h_rope_heads,
                6,
                g(kv_heads as u32),
                b256,
                &[cell_ptr(k_ptr), kv_heads as u64, (d / 2) as u64, 0, eta_bits, one_bits],
            );
            refresh.push((acc.specs.len() - 1, 3)); // rope k pos (per-step)
            cell_rope_k_pos.push(acc.cell_of(3));
            // kv write (phys/off are per-step)
            acc.custom(
                h_kv_write,
                9,
                g((kv_heads * d).div_ceil(256) as u32),
                b256,
                &[
                    cell_ptr(k_ptr),
                    cell_ptr(v_ptr),
                    cell_ptr(kv_ptr),
                    0,
                    0,
                    BLOCK_LEN as u64,
                    kv_heads as u64,
                    d as u64,
                    total_pages as u64,
                ],
            );
            // kv_write phys/off are per-step (both slots of this node).
            let kvw = acc.specs.len() - 1;
            refresh.push((kvw, 3));
            refresh.push((kvw, 4));
            cell_kv_phys.push(acc.cell_of(3));
            cell_kv_off.push(acc.cell_of(4));
            // decode attention (006-2 T2 flash kernel; page table = identity,
            // li*pp offset — args mirror launch_decode_step_gqa_flash).
            // The launch runs FLASH_TPB = 512 threads with dynamic smem
            // (d + max_kv) * 4 (decode.rs `launch_fmha`) — the declaration
            // mirrors the geometry exactly (Graph V2 refresh rewrites
            // grid/block/shared from the declaration; a mismatch would
            // launch the kernel with wrong smem).
            let page_li = unsafe { pages_ptr.add(li * eng.pp) };
            let flash_shared = ((d as usize) + max_kv as usize) * 4;
            acc.custom_with_shared(
                h_decode,
                14,
                g(q_heads as u32),
                b512,
                flash_shared as u32,
                &[
                    cell_ptr(q_ptr),
                    cell_ptr(page_li),
                    cell_ptr(kv_ptr),
                    cell_ptr(lens_ptr),
                    cell_ptr(attn_ptr),
                    1,
                    q_heads as u64,
                    d as u64,
                    BLOCK_LEN as u64,
                    ratio as u64,
                    kv_heads as u64,
                    max_kv as u64,
                    total_pages as u64,
                    1, // identity page table (S1-2 static identity mapping)
                ],
            );
            // o projection, then fused residual add + ffn rms (S1-4: the
            // add_cast + rms_ffn nodes fused into one — bit-identical)
            declare_gemm(&mut acc, &pl.o);
            acc.custom(
                h_add_rms,
                6,
                g1,
                b256,
                &[
                    cell_ptr(x_ptr),
                    cell_ptr(eng.c_o.as_ptr()),
                    cell_ptr(xn_ptr),
                    cell_ptr(w.ffn_norm.as_ptr()),
                    h as u64,
                    eps_bits,
                ],
            );
            // gate/up GEMMs + fused cast-swiglu (S1-4: the cast gate + cast
            // up + swiglu nodes fused into one — bit-identical)
            declare_gemm(&mut acc, &pl.gate);
            declare_gemm(&mut acc, &pl.up);
            acc.custom(
                h_cast_swiglu,
                4,
                g(ffn.div_ceil(256) as u32),
                b256,
                &[
                    cell_ptr(eng.c_g.as_ptr()),
                    cell_ptr(eng.c_u.as_ptr()),
                    cell_ptr(down_ptr),
                    ffn as u64,
                ],
            );
            declare_gemm(&mut acc, &pl.down);
            acc.custom(
                h_add_cast,
                3,
                g(h.div_ceil(256) as u32),
                b256,
                &[cell_ptr(x_ptr), cell_ptr(eng.c_d.as_ptr()), h as u64],
            );
        }
        // final rms_norm + lm_head
        acc.custom(
            h_rms,
            5,
            g1,
            b256,
            &[
                cell_ptr(x_ptr),
                cell_ptr(xn_ptr),
                cell_ptr(eng.final_norm.as_ptr()),
                h as u64,
                eps_bits,
            ],
        );
        declare_gemm(&mut acc, &eng.plans.lm_head);

        // Launch-order self-check: a divergence from the eager launch
        // count would fail every capture closed (diagnostic only).
        let expected = expected_node_count(n_layers, cfg.head_norm, jgemm.is_some(), false);
        if acc.specs.len() != expected {
            eprintln!(
                "reinfer-cuda graph: spec declaration count {} != expected {expected} — \
                 declaration out of sync with step_decode_launches; captures will fail closed",
                acc.specs.len()
            );
        }

        // Constant update list — one PtrUpdate per declared slot, pointing
        // at the cell addresses (built last: `cells` must never move).
        let mut updates = Vec::with_capacity(acc.specs.len());
        for (node, spec) in acc.specs.iter().enumerate() {
            let base = acc.cell_off[node];
            for (slot, _) in &spec.ptr_slots {
                let cell_ptr = (&mut acc.cells[base + *slot] as *mut u64) as *mut c_void;
                updates.push(PtrUpdate { node, slot: *slot, ptr: cell_ptr });
            }
        }
        // Stable per-node kernelParams arrays (Graph V2 — see `arg_slots`):
        // the driver records the args array *pointer* at capture (measured:
        // no copy — transient arrays are read as reused garbage at replay),
        // so the arrays must live as long as the decl. Built here, after
        // `cells` is final — the arrays' entries are the cell addresses and
        // the per-node array for node n is `&arg_slots[cell_off[n]]`.
        let arg_slots: Vec<*mut c_void> = acc
            .cells
            .iter()
            .map(|c| (c as *const u64) as *mut c_void)
            .collect();

        // CUfunction launch handles for the decl-driven launch (Graph V2 —
        // the same kernels the eager launch fns use; converted once here so
        // the per-step decl-driven path is one launch per node with no
        // per-step driver conversion calls).
        let mut launch_fns = Vec::with_capacity(acc.specs.len());
        for spec in &acc.specs {
            if spec.role == NodeRole::CustomKernel {
                let mut f: dsys::CUfunction = std::ptr::null_mut();
                // SAFETY: `spec.handle` is a live CUkernel from
                // cuLibraryGetKernel (engine-owned library); output slot
                // valid; null handle would fail the call (fails closed).
                let r = unsafe { dsys::cuKernelGetFunction(&mut f, spec.handle as dsys::CUkernel) };
                if r != dsys::CUresult::CUDA_SUCCESS {
                    return Err(LaunchError::Fatal);
                }
                launch_fns.push(f);
            } else {
                launch_fns.push(std::ptr::null_mut());
            }
        }

        Ok(Self {
            specs: acc.specs,
            launch_fns,
            cell_off: acc.cell_off,
            cells: acc.cells,
            arg_slots,
            updates,
            cell_token,
            cell_rope_q_pos,
            cell_rope_k_pos,
            cell_kv_phys,
            cell_kv_off,
            refresh,
        })
    }

    /// S1-9: fused-decode declaration — the exact launch-order mirror of
    /// `step_decode_launches_fused`: gather, rms_attn(0), per layer
    /// [p1_qkv (phase-1 of q/k/v), p2_qkv (reductions + casts + head-norm
    /// + rope), flash_fused (kv write + flash decode), p1_o (o phase-1 —
    /// its own node: it reads the full attention row, which the flash
    /// blocks write only per head), p2_o (o reductions + residual + ffn
    /// norm), p1_gu (phase-1 of gate/up — reads p2_o's ffn-normed xn),
    /// p2_gu_d (gate/up phase-2 + cast-SiLU-GLU + down phase-1 in one
    /// kernel — block-local merge, see `gemv_p2_gu_p1d_swiglu`), p2_down
    /// (down reductions + residual + next attn norm)], lm_head phase pair
    /// (multi p1 + the shared jgemm reduce). 4 + 8*n_layers nodes, every
    /// one a `CustomKernel` with full pointer-slot coverage (Graph V2
    /// refresh-safe). Per-step refresh slots: gather token and the p2_qkv
    /// `pos` (one cell shared by the q and k rope passes) — the 13.2
    /// driver bakes kernel params at capture, so every replay
    /// re-refreshes these nodes' params from the cells (V2 SetParams).
    /// The fused flash node derives its kv slot from kv_len inside the
    /// kernel, so no phys/off cells are declared.
    fn build_fused(eng: &Engine) -> Result<Self, LaunchError> {
        let cfg = &eng.cfg;
        let h = cfg.hidden_size;
        let nqk = cfg.q_heads * cfg.head_dim;
        let kvk = cfg.kv_heads * cfg.head_dim;
        let d = cfg.head_dim;
        let kv_heads = cfg.kv_heads;
        let q_heads = cfg.q_heads;
        let ratio = q_heads / kv_heads;
        let total_pages = cfg.n_layer * eng.pp;
        let eps = cfg.rms_eps;
        let eta = cfg.rope_theta;
        let n_layers = eng.layers.len();
        let fused = eng.fused.as_ref().expect("fused loaded");
        let g = fused.geom();

        // CUkernel handles — the handle form the capture records.
        let cu = |lib: cudarc::driver::sys::CUlibrary,
                  name: &str|
         -> Result<*mut c_void, LaunchError> { cu_kernel_of(lib, name) };
        let dense = eng.kernels.raw_lib();
        let flib = fused.raw_lib();
        let flash_decode = eng.decode.flash_raw_lib();
        let h_gather = cu(dense, "gather_row")?;
        let h_rms = cu(dense, "rms_norm_row_f16")?;
        let h_p1 = cu(flib, "gemv_m1_f16f32_multi")?;
        let h_p2qkv = cu(flib, "gemv_p2_qkv_cast_hn_rope")?;
        let h_p2gud = cu(flib, "gemv_p2_gu_p1d_swiglu")?;
        let h_p2rms = cu(flib, "gemv_p2_add_rms")?;
        let h_flash_fused = cu(flash_decode, "decode_step_gqa_flash_fused")?;
        // lm_head phase-2 reuses the loaded Jgemm's reduce kernel.
        let (_, h_reduce) = eng.jgemm.as_ref().expect("jgemm loaded").kernel_handles()?;

        let b256 = cudarc::runtime::sys::dim3 { x: 256, y: 1, z: 1 };
        // Flash decode launches with FLASH_TPB = 512 threads (decode.rs).
        let b512 = cudarc::runtime::sys::dim3 { x: 512, y: 1, z: 1 };
        let g1 = cudarc::runtime::sys::dim3 { x: 1, y: 1, z: 1 };
        let gd = |x: u32| cudarc::runtime::sys::dim3 { x, y: 1, z: 1 };

        let eps_bits = eps.to_bits() as u64;
        let eta_bits = eta.to_bits() as u64;
        let qs_bits = (1.0 / (d as f32).sqrt()).to_bits() as u64;
        let one_bits = 1.0f32.to_bits() as u64;
        let d64 = d as u64;
        let half64 = (d / 2) as u64;
        let hn = if cfg.head_norm { 1u64 } else { 0u64 };
        let x_ptr = eng.x.as_ptr();
        let xn_ptr = eng.xn.as_ptr();
        let q_ptr = eng.q.as_ptr();
        let k_ptr = eng.k.as_ptr();
        let v_ptr = eng.v.as_ptr();
        let attn_ptr = eng.attn.as_ptr();
        let kv_ptr = eng.kv.data.as_ptr();
        let lens_ptr = eng.lens_dev.as_ptr();
        let pages_ptr = eng.pages_dev.as_ptr() as *const u32;
        // Token-capacity bound for the flash decode kernel (smem sizing +
        // in-kernel guard): pp * BLOCK_LEN >= config max_kv by construction.
        let max_kv = (eng.pp * BLOCK_LEN) as u32;

        let mut acc = SpecAcc::default();
        // gather: (src=embed, dst=x, row=token, n=h) — token is per-step.
        acc.custom(
            h_gather,
            4,
            gd(h.div_ceil(256) as u32),
            b256,
            &[cell_ptr(eng.embed.as_ptr()), cell_ptr(x_ptr), 0, h as u64],
        );
        let mut refresh: Vec<(usize, usize)> = vec![(acc.specs.len() - 1, 2)];
        let cell_token = acc.cell_of(2);
        let mut cell_rope_q_pos = Vec::with_capacity(n_layers);
        let mut cell_rope_k_pos = Vec::with_capacity(n_layers);
        // attn rms(0) — the layer-0 qkv plans read xn after this.
        acc.custom(
            h_rms,
            5,
            g1,
            b256,
            &[
                cell_ptr(x_ptr),
                cell_ptr(xn_ptr),
                cell_ptr(eng.layers[0].attn_norm.as_ptr()),
                h as u64,
                eps_bits,
            ],
        );

        for li in 0..n_layers {
            let w = &eng.layers[li];
            // 1. phase-1 of the q/k/v plans (a = xn, the layer input)
            let table = unsafe { (g.tables as *const PlanRow).add(li * 7) };
            acc.custom(
                h_p1,
                2,
                gd(g.grid_qkv[li]),
                b256,
                &[cell_ptr(table), 3],
            );
            // 2. q/k/v phase-2 + casts + head-norm + rope (22 slots; pos
            // at slot 16 is per-step — one shared cell for q and k)
            let (wq, wk) = match (&w.q_norm, &w.k_norm) {
                (Some(qn), Some(kn)) => (cell_ptr(qn.as_ptr()), cell_ptr(kn.as_ptr())),
                _ => (0, 0),
            };
            acc.custom(
                h_p2qkv,
                22,
                gd(g.grid_qkv_p2),
                b256,
                &[
                    cell_ptr(g.pq),
                    cell_ptr(g.pk),
                    cell_ptr(g.pv),
                    cell_ptr(q_ptr),
                    cell_ptr(k_ptr),
                    cell_ptr(v_ptr),
                    wq,
                    wk,
                    nqk as u64,
                    kvk as u64,
                    kvk as u64,
                    g.nslabs_q as u64,
                    g.nslabs_k as u64,
                    g.nslabs_v as u64,
                    d64,
                    half64,
                    0, // pos (per-step)
                    eta_bits,
                    qs_bits,
                    one_bits,
                    eps_bits,
                    hn,
                ],
            );
            refresh.push((acc.specs.len() - 1, 16)); // p2_qkv pos (per-step)
            cell_rope_q_pos.push(acc.cell_of(16));
            cell_rope_k_pos.push(acc.cell_of(16));
            // 3. fused decode (kv write + flash attention in one kernel;
            // page table = identity, li*pp offset; grid = B*QH; the kv
            // slot is derived from kv_len inside the kernel — args mirror
            // launch_decode_step_gqa_flash_fused).
            let page_li = unsafe { pages_ptr.add(li * eng.pp) };
            let flash_shared = ((d as usize) + max_kv as usize) * 4;
            acc.custom_with_shared(
                h_flash_fused,
                16,
                gd(q_heads as u32),
                b512,
                flash_shared as u32,
                &[
                    cell_ptr(q_ptr),
                    cell_ptr(page_li),
                    cell_ptr(kv_ptr),
                    cell_ptr(lens_ptr),
                    cell_ptr(attn_ptr),
                    1,
                    q_heads as u64,
                    d64,
                    BLOCK_LEN as u64,
                    ratio as u64,
                    kv_heads as u64,
                    max_kv as u64,
                    total_pages as u64,
                    1, // identity page table (S1-2 static identity mapping)
                    cell_ptr(k_ptr),
                    cell_ptr(v_ptr),
                ],
            );
            // 4. o phase-1 — its own node: it reads the FULL attention row
            // (all B*QH heads), which the flash blocks write only for
            // their own heads — folding it in would race across blocks.
            // Stream-ordered after the flash node, it is race-free.
            acc.custom(
                h_p1,
                2,
                gd(g.grid_o[li]),
                b256,
                &[cell_ptr(unsafe { table.add(3) }), 1],
            );
            // 5. o phase-2 + residual add + ffn rms (consumes the o
            // phase-1 node's partials)
            acc.custom(
                h_p2rms,
                7,
                g1,
                b256,
                &[
                    cell_ptr(g.po),
                    cell_ptr(x_ptr),
                    cell_ptr(xn_ptr),
                    cell_ptr(w.ffn_norm.as_ptr()),
                    h as u64,
                    g.nslabs_o as u64,
                    eps_bits,
                ],
            );
            // 6. gate/up phase-1 — reads p2_o's ffn-normed xn (the split
            // path's read point)
            acc.custom(
                h_p1,
                2,
                gd(g.grid_gu[li]),
                b256,
                &[cell_ptr(unsafe { table.add(4) }), 2],
            );
            // 7. gate/up phase-2 + cast-SiLU-GLU + down phase-1 in one
            // kernel (`gemv_p2_gu_p1d_swiglu`, grid = ncols_d*nslabs_d,
            // the split's full down phase-1 tile grid): each block
            // redundantly writes the 256-col phase-1 stripe its phase-2
            // k-range lies in, then computes the down tile
            // (bx/nslabs_d, bx%nslabs_d) (valid iff every slab's k-range
            // fits in its block's stripe — checked in build_plans).
            // Block-local, same arithmetic as the split p2_gu + p1_d pair.
            acc.custom(
                h_p2gud,
                2,
                gd(g.grid_gu_p2),
                b256,
                &[cell_ptr(unsafe { table.add(4) }), 3],
            );
            // 8. down phase-2 + residual add + next layer's attn rms (the
            // last layer's rms target is final_norm — the fused step's
            // final norm lives here, no separate final-rms node)
            let wnext = if li + 1 < n_layers {
                cell_ptr(eng.layers[li + 1].attn_norm.as_ptr())
            } else {
                cell_ptr(eng.final_norm.as_ptr())
            };
            acc.custom(
                h_p2rms,
                7,
                g1,
                b256,
                &[
                    cell_ptr(g.pd),
                    cell_ptr(x_ptr),
                    cell_ptr(xn_ptr),
                    wnext,
                    h as u64,
                    g.nslabs_d as u64,
                    eps_bits,
                ],
            );
        }
        // lm_head phase pair: multi p1 (a = xn) + shared reduce p2
        acc.custom(h_p1, 2, gd(g.grid_lm), b256, &[cell_ptr(g.lm_table), 1]);
        acc.custom(
            h_reduce,
            4,
            gd(g.grid_lm),
            b256,
            &[
                cell_ptr(g.plm),
                cell_ptr(eng.logits.as_ptr()),
                eng.cfg.vocab_size as u64,
                1,
            ],
        );

        // Launch-order self-check (diagnostic only).
        let expected = expected_node_count(n_layers, cfg.head_norm, true, true);
        if acc.specs.len() != expected {
            eprintln!(
                "reinfer-cuda graph: fused spec declaration count {} != expected {expected} — \
                 declaration out of sync with step_decode_launches_fused; captures will fail \
                 closed",
                acc.specs.len()
            );
        }

        // Constant update list — one PtrUpdate per declared slot, pointing
        // at the cell addresses (built last: `cells` must never move).
        let mut updates = Vec::with_capacity(acc.specs.len());
        for (node, spec) in acc.specs.iter().enumerate() {
            let base = acc.cell_off[node];
            for (slot, _) in &spec.ptr_slots {
                let cell_ptr = (&mut acc.cells[base + *slot] as *mut u64) as *mut c_void;
                updates.push(PtrUpdate { node, slot: *slot, ptr: cell_ptr });
            }
        }
        // Stable per-node kernelParams arrays (Graph V2 — see `arg_slots`).
        let arg_slots: Vec<*mut c_void> = acc
            .cells
            .iter()
            .map(|c| (c as *const u64) as *mut c_void)
            .collect();

        // CUfunction launch handles for the decl-driven launch (Graph V2).
        let mut launch_fns = Vec::with_capacity(acc.specs.len());
        for spec in &acc.specs {
            if spec.role == NodeRole::CustomKernel {
                let mut f: dsys::CUfunction = std::ptr::null_mut();
                // SAFETY: `spec.handle` is a live CUkernel from
                // cuLibraryGetKernel (engine-owned library); output slot
                // valid; null handle would fail the call (fails closed).
                let r = unsafe { dsys::cuKernelGetFunction(&mut f, spec.handle as dsys::CUkernel) };
                if r != dsys::CUresult::CUDA_SUCCESS {
                    return Err(LaunchError::Fatal);
                }
                launch_fns.push(f);
            } else {
                launch_fns.push(std::ptr::null_mut());
            }
        }

        Ok(Self {
            specs: acc.specs,
            launch_fns,
            cell_off: acc.cell_off,
            cells: acc.cells,
            arg_slots,
            updates,
            cell_token,
            cell_rope_q_pos,
            cell_rope_k_pos,
            // The fused declaration folds the kv write into the fused
            // flash node (slot derived from kv_len in-kernel) — no
            // phys/off cells; write_step's zip loop is a no-op.
            cell_kv_phys: Vec::new(),
            cell_kv_off: Vec::new(),
            refresh,
        })
    }

    /// Whether every node is a declared custom kernel (the Graph V2
    /// decl-driven precondition: only then can the capture record the C3
    /// cell addresses as the node params — the cublas declaration boundary,
    /// REINFER_JGEMM=off, stays on the fail-closed refresh path).
    #[must_use]
    pub fn is_all_custom(&self) -> bool {
        self.specs.iter().all(|s| s.role == NodeRole::CustomKernel)
    }
}

/// Send wrapper for the captured-exec map (S1-3). `GraphExec` holds raw
/// CUDA graph handles (!Send), but serve.rs moves the Engine across
/// threads (`Mutex<Engine>` in AppState, spawn_blocking workers). The
/// handles are context-bound: Drop binds the device context before
/// destroying the execs (best-effort — a missing context at process
/// teardown is cleaned up by the driver anyway).
#[derive(Debug)]
pub struct GraphExecStore {
    execs: HashMap<usize, GraphExec>,
    dev: u32,
}

// Ownership transfer of driver handles is sound as long as every destroy
// path runs with a current context — the wrapper's Drop provides it.
unsafe impl Send for GraphExecStore {}

impl GraphExecStore {
    fn new(dev: u32) -> Self {
        Self { execs: HashMap::new(), dev }
    }

    fn contains_key(&self, bucket: usize) -> bool {
        self.execs.contains_key(&bucket)
    }

    fn get_mut(&mut self, bucket: usize) -> Option<&mut GraphExec> {
        self.execs.get_mut(&bucket)
    }

    fn insert(&mut self, bucket: usize, exec: GraphExec) {
        self.execs.insert(bucket, exec);
    }

    fn remove(&mut self, bucket: usize) {
        self.execs.remove(&bucket);
    }
}

impl Drop for GraphExecStore {
    fn drop(&mut self) {
        let _ = CtxGuard::set_current(self.dev);
        self.execs.clear();
    }
}

impl Engine {
    /// 装载：读 config.json + model.safetensors（模型目录；权重 f16 化上传）。
    /// 006 T3E: 图池按 REINFER_GRAPH（默认 on）决定——on → 真池，off → 恒 eager。
    pub fn load(
        dev: DeviceId,
        arch: &str,
        cache_dir: Option<PathBuf>,
        model_dir: &Path,
        max_kv: usize,
    ) -> Result<Self, EngineError> {
        let graph = if graph_enabled_from_env(std::env::var(GRAPH_ENV).ok().as_deref()) {
            GraphPool::new(dev)
        } else {
            GraphPool::disabled()
        };
        Self::load_with_graph(dev, arch, cache_dir, model_dir, max_kv, graph)
    }

    /// 装载（图池注入——测试/诊断用；`load` 按环境委托本构造器）。
    pub fn load_with_graph(
        dev: DeviceId,
        arch: &str,
        cache_dir: Option<PathBuf>,
        model_dir: &Path,
        max_kv: usize,
        graph: GraphPool,
    ) -> Result<Self, EngineError> {
        // 014 S0-3b: parity-f32 criterion tier — read once at load (default
        // off); when on, every weight tensor is loaded as the f32 expansion
        // of its f16 value (bit-identical values, gemm B layout unchanged).
        let parity_f32 = parity_f32_enabled_from_env(std::env::var(PARITY_F32_ENV).ok().as_deref());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
                .map_err(|e| EngineError::Sts(format!("config.json: {e}")))?;
        let cfg = from_hf_config(&config)?;
        let safe = SafeFile::open(&model_dir.join("model.safetensors"))
            .map_err(|e| EngineError::Sts(e.to_string()))?;

        let h = cfg.hidden_size;
        let nqk = cfg.q_heads * cfg.head_dim;
        let kvk = cfg.kv_heads * cfg.head_dim;
        let ffn = cfg.ffn_hidden;
        let vocab = cfg.vocab_size;
        let n_layers = cfg.n_layer;

        let expand =
            |w16: Vec<u8>| -> Vec<u8> { if parity_f32 { expand_f16_to_f32(&w16) } else { w16 } };
        let t =
            |name: &str, out_rows: usize, in_cols: usize| -> Result<DeviceBuffer, EngineError> {
                let view = safe.tensor(name).map_err(|e| EngineError::Sts(e.to_string()))?;
                if view.shape.len() != 2
                    || view.shape[0] != out_rows as u64
                    || view.shape[1] != in_cols as u64
                {
                    return Err(EngineError::WeightShape(format!(
                        "{name}: expected [{out_rows},{in_cols}] got {:?}",
                        view.shape
                    )));
                }
                let w = expand(to_f16_rm(&view, out_rows, in_cols)?);
                upl(dev, &w).map_err(EngineError::Launch)
            };
        let tv = |name: &str, d: usize| -> Result<DeviceBuffer, EngineError> {
            let view = safe.tensor(name).map_err(|e| EngineError::Sts(e.to_string()))?;
            if view.shape.len() != 1 || view.shape[0] != d as u64 {
                return Err(EngineError::WeightShape(format!(
                    "{name}: expected [{d}] got {:?}",
                    view.shape
                )));
            }
            let w = expand(to_f16_vec(&view)?);
            upl(dev, &w).map_err(EngineError::Launch)
        };

        // embed 行主序取（gather 行长=token——**不做** gemm 转置；lm_head 走 gemm B 转置）。
        let embed = {
            let view = safe
                .tensor("model.embed_tokens.weight")
                .map_err(|e| EngineError::Sts(e.to_string()))?;
            if view.shape.len() != 2 || view.shape[0] != vocab as u64 || view.shape[1] != h as u64 {
                return Err(EngineError::WeightShape(format!(
                    "model.embed_tokens.weight: expected [{vocab},{h}] got {:?}",
                    view.shape
                )));
            }
            let w = expand(to_f16_rows(&view)?);
            upl(dev, &w).map_err(EngineError::Launch)?
        };
        let lm_head = t("lm_head.weight", vocab, h)?;
        let final_norm = tv("model.norm.weight", h)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            // S1-7: fused QKV weight (f16 channel only — the parity-f32 tier
            // routes prefill to the per-token f32 path, so no fused copy is
            // built there). All three projections share k = h, so the fused
            // [h x (nqk+2kvk)] matrix is the per-row column join of the
            // three [k x n] to_f16_rm buffers (fused row r = q row r ++ k
            // row r ++ v row r) — the byte layout gemm1r's column-major A
            // (ld N) must see to reproduce the three separated GEMMs
            // (verified bit-exact by the fused_qkv_gemm_layout_probe test).
            let qkv_proj = if parity_f32 {
                None
            } else {
                let q16 = to_f16_rm(
                    &safe.tensor(&p("self_attn.q_proj.weight")).map_err(|e| {
                        EngineError::Sts(format!("self_attn.q_proj.weight: {e}"))
                    })?,
                    nqk,
                    h,
                )?;
                let k16 = to_f16_rm(
                    &safe.tensor(&p("self_attn.k_proj.weight")).map_err(|e| {
                        EngineError::Sts(format!("self_attn.k_proj.weight: {e}"))
                    })?,
                    kvk,
                    h,
                )?;
                let v16 = to_f16_rm(
                    &safe.tensor(&p("self_attn.v_proj.weight")).map_err(|e| {
                        EngineError::Sts(format!("self_attn.v_proj.weight: {e}"))
                    })?,
                    kvk,
                    h,
                )?;
                // Fused [h x N] built as a per-row column join: fused row r
                // = q16 row r ++ k16 row r ++ v16 row r. gemm1r's cublas A
                // reads the buffer as column-major [N x h] (ld N), so fused
                // element (i, k) must equal the q/k/v weight element [i, k]
                // of the corresponding separated buffer — exactly the
                // per-row join (verified bit-exact vs the three separated
                // GEMMs in fmha_heuristics_bench fused_qkv_gemm_layout_probe;
                // a flat q16++k16++v16 byte stack is NOT a column join and
                // diverges at O(1e2) — the S1-7 D7 bug).
                let n2 = (nqk + 2 * kvk) * 2;
                let nk2 = kvk * 2;
                let nq2 = nqk * 2;
                let mut fused = vec![0u8; h * n2];
                for r in 0..h {
                    let row = r * n2;
                    fused[row..row + nq2].copy_from_slice(&q16[r * nq2..(r + 1) * nq2]);
                    fused[row + nq2..row + nq2 + nk2]
                        .copy_from_slice(&k16[r * nk2..(r + 1) * nk2]);
                    fused[row + nq2 + nk2..(r + 1) * n2]
                        .copy_from_slice(&v16[r * nk2..(r + 1) * nk2]);
                }
                Some(upl(dev, &fused).map_err(EngineError::Launch)?)
            };
            layers.push(LayerWeights {
                attn_norm: tv(&p("input_layernorm.weight"), h)?,
                q_proj: t(&p("self_attn.q_proj.weight"), nqk, h)?,
                k_proj: t(&p("self_attn.k_proj.weight"), kvk, h)?,
                v_proj: t(&p("self_attn.v_proj.weight"), kvk, h)?,
                qkv_proj,
                o_proj: t(&p("self_attn.o_proj.weight"), h, nqk)?,
                q_norm: cfg
                    .head_norm
                    .then(|| tv(&p("self_attn.q_norm.weight"), cfg.head_dim))
                    .transpose()?,
                k_norm: cfg
                    .head_norm
                    .then(|| tv(&p("self_attn.k_norm.weight"), cfg.head_dim))
                    .transpose()?,
                ffn_norm: tv(&p("post_attention_layernorm.weight"), h)?,
                gate_proj: t(&p("mlp.gate_proj.weight"), ffn, h)?,
                up_proj: t(&p("mlp.up_proj.weight"), ffn, h)?,
                down_proj: t(&p("mlp.down_proj.weight"), h, ffn)?,
            });
        }

        let stream = CudaStream::new(dev)?;
        let kernels = DenseKernels::new(arch, cache_dir.clone())?;
        let diff = DiffKernels::new(arch, None, stream.clone())?;
        let decode = DecodeKernels::new(arch, None, stream.clone())?;
        let gemm = Gemm::new(dev.index())?;
        // JitGemm (m=1 decode projections): default on. The decode-step
        // graph declaration (Graph V2) mirrors the jgemm launch path —
        // each m=1 GEMM is declared as the two custom phase kernels
        // (CustomKernel nodes with full pointer-slot coverage), which is
        // exactly what makes replay refreshable on the 13.x runtime. Load
        // failure fails open to cublas with a note.
        let jgemm = if jgemm_disabled_from_env(std::env::var(JGEMM_ENV).ok().as_deref()) {
            None
        } else {
            match Jgemm::new(dev.index(), arch, cache_dir.clone()) {
                Ok(jg) => Some(jg),
                Err(e) => {
                    eprintln!("reinfer-cuda: jgemm load failed — cublas path (fail-open): {e}");
                    None
                }
            }
        };

        let pp = max_kv.div_ceil(BLOCK_LEN);
        let kv = crate::decode::KvStore::alloc(
            dev,
            cfg.n_layer * pp,
            BLOCK_LEN,
            cfg.kv_heads,
            cfg.head_dim,
        )?;

        let a16 = |n: usize| DeviceBuffer::alloc(dev, n * 2);
        let c32 = |n: usize| DeviceBuffer::alloc(dev, n * 4);
        // S1-2: static identity page table (see `pages_dev` field comment) —
        // uploaded once here, never touched per step. kv_write computes
        // phys = li*pp + lp and decode reads page[j] = li*pp + j, which is
        // exactly this table's identity content.
        let pages_dev = {
            let ids: Vec<u8> = (0..(n_layers * pp) as u32).flat_map(|i| i.to_le_bytes()).collect();
            upl(dev, &ids).map_err(EngineError::Launch)?
        };
        let mut this = Self {
            dev: dev.index(),
            cfg,
            gemm,
            jgemm,
            jgemm_fallbacks: AtomicU64::new(0),
            // S1-9: placeholder — built below from the fully-constructed
            // `this` (needs the plans and the loaded JitGemm).
            fused: None,
            kernels,
            diff,
            decode,
            fmha: None,
            prefill: None,
            arch: arch.to_string(),
            stream,
            embed,
            lm_head,
            final_norm,
            layers,
            kv,
            pp,
            pages_dev,
            lens_dev: DeviceBuffer::alloc(dev, 4)?,
            x: a16(h)?,
            xn: a16(h)?,
            q: a16(nqk)?,
            k: a16(kvk)?,
            v: a16(kvk)?,
            attn: a16(nqk)?,
            down: a16(ffn)?,
            x32: c32(h)?,
            xn32: c32(h)?,
            attn32: c32(nqk)?,
            c_q: c32(nqk)?,
            c_k: c32(kvk)?,
            c_v: c32(kvk)?,
            c_o: c32(h)?,
            c_g: c32(ffn)?,
            c_u: c32(ffn)?,
            c_d: c32(h)?,
            logits: c32(vocab)?,
            logits_host: HostBuffer::alloc(vocab * 4)?,
            scores: DeviceBuffer::alloc(dev, nqk.max(1) * max_kv * 4)?,
            db: TuneDb::open(),
            sel: SelectionCache::new(),
            graph,
            graph_failed: HashSet::new(),
            graph_eager_fallbacks: 0,
            decode_flash_fallbacks: 0,
            lens_hb: HostBuffer::alloc(4)?,
            // S1-3: placeholder — built below from the fully-constructed
            // `this` (needs the plans and the loaded kernels).
            graph_decl: None,
            graph_execs: GraphExecStore::new(dev.index()),
            graph_captures: 0,
            graph_replays: 0,
            parity_f32,
            prof: DecodeProfiler::new(dev),
            prefill_prof: PrefillProfiler::new(dev),
            // Placeholder — built below from the fully-constructed `this`.
            plans: DecodeGemmPlans {
                layers: Vec::new(),
                lm_head: GemmPlan::row_major_f16(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    1,
                    1,
                    1,
                ),
            },
        };
        // S1-2: stable GEMM parameter grid — cells address the engine's
        // buffers directly (they never move after load), so the decode step
        // only executes prebuilt plans.
        this.plans = DecodeGemmPlans::build(&this);
        // S1-9: fused decode-step kernels (default on). The fused step
        // replaces the split launch sequence with 8 nodes/layer (228 total
        // for Qwen3-0.6B vs 760) — bit-identical by construction (fused.rs;
        // REINFER_FUSED=off is the split A/B reference arm). Off when the
        // jgemm path is off (REINFER_JGEMM=off keeps the cublas reference),
        // on the parity-f32 tier, when the naive decode attention is forced
        // (REINFER_DECODE_FLASH=off — the fused sequence is declared around
        // the flash kernel), or when the fused geometry is unsupported
        // (head_dim not dividing the 256-thread block, head_dim < 32, or
        // hidden > 1024 — the fused rms row contract, same bound as the
        // split fused_add_rms). Load failure fails open to the split path
        // with a note.
        this.fused = if this.parity_f32
            || fused_decode_disabled_from_env(std::env::var(FUSED_ENV).ok().as_deref())
            || jgemm_disabled_from_env(std::env::var(JGEMM_ENV).ok().as_deref())
            || decode_flash_disabled_from_env(std::env::var(DECODE_FLASH_ENV).ok().as_deref())
            || this.cfg.head_dim < 32
            || 256 % this.cfg.head_dim != 0
            || this.cfg.hidden_size > 1024
        {
            None
        } else {
            let jg = this.jgemm.as_ref().expect("jgemm loaded (gate above)");
            match FusedDecodeKernels::new(
                dev.index(),
                arch,
                cache_dir.clone(),
                jg.kernel_reduce_fn(),
            ) {
                Ok(mut f) => match f.build_plans(dev, jg, &this.plans) {
                    Ok(()) => Some(f),
                    Err(e) => {
                        eprintln!(
                            "reinfer-cuda: fused decode plans build failed — split path \
                             (fail-open): {e}"
                        );
                        None
                    }
                },
                Err(e) => {
                    eprintln!(
                        "reinfer-cuda: fused decode kernels load failed — split path \
                         (fail-open): {e}"
                    );
                    None
                }
            }
        };
        // S1-3: decode-step graph declaration — built once for the f16
        // channel (the parity-f32 tier stays eager; its launch sequence
        // differs). When the fused kernels are loaded, the declaration
        // mirrors the fused 8-node sequence (`build_fused`); otherwise it
        // mirrors the split sequence. Build failure is fail-open: the
        // engine runs eager and logs (a wrong declaration would fail every
        // capture closed).
        if this.graph.enabled() && !this.parity_f32 {
            match GraphStepDecl::build(&this) {
                Ok(decl) => this.graph_decl = Some(decl),
                Err(e) => {
                    eprintln!(
                        "reinfer-cuda graph: spec declaration build failed — graph \
                               path disabled (eager): {e}"
                    );
                }
            }
        }
        // S1-2: profiler events are illegal inside a graph capture window —
        // keep the probe inert on the graph path (graph default off; the
        // S1-1/S1-2 attribution runs happen on the eager path).
        if this.graph.enabled() {
            this.prof.active = false;
        }
        // 006 T3E graph: warm up cublas on a real-shaped gemm BEFORE any
        // capture window. cublas lazily allocates its workspace on first
        // gemm use; inside a capture that allocation becomes a graph memalloc
        // node with deferred memory, and the handle is left holding a stale
        // workspace pointer once the graph is destroyed -> flaky eager
        // corruption on later steps (observed: all-logits-garbage at step 1
        // in some runs). Forcing the allocation here keeps every capture
        // window free of memalloc nodes. The lm_head plan is the production
        // geometry (m=1 k=h); logits is fully overwritten by the first real
        // step.
        if this.graph.enabled() {
            this.gemm_exec_plan(&this.stream, &this.plans.lm_head)?;
        }
        Ok(this)
    }

    /// 配置。
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// 设备索引。
    pub fn device(&self) -> u32 {
        self.dev
    }

    /// The engine's decode stream — every eager launch and graph replay is
    /// enqueued on it. Benchmark/test access for GPU-side timing
    /// (cudaEvent pairs around a decode loop measure the GPU-busy mean
    /// vs the host-wall mean).
    pub fn decode_stream(&self) -> &CudaStream {
        &self.stream
    }

    /// 006-2 T3E wiring: wrap the engine's per-step device logits in a
    /// `LogitsView` (GPU sampler chain input). The view borrows the reused
    /// `logits` device buffer — it must be dropped before the next `step`
    /// (LogitsView per-step buffer reuse contract). `to_host()` on the view
    /// performs a fresh D2H readback with the 014 `Backend::logits()` element
    /// order (row-major `[vocab]`); device copy failures are fatal at call
    /// time (LogitsView contract: `to_host` is infallible).
    pub fn logits_view(&self) -> reinfer_kernels::LogitsView {
        let dev = DeviceId::new(self.dev);
        let dev_idx = self.dev;
        let ptr = self.logits.as_ptr() as usize;
        let bytes = self.cfg.vocab_size * 4;
        let vocab = self.cfg.vocab_size;
        reinfer_kernels::LogitsView::new(
            dev,
            reinfer_kernels::DeviceBuffer::new(ptr, bytes),
            vocab,
            move || {
                // Context-bound device pointer: bind the device on this thread
                // (CtxGuard, same discipline as every kernel launch), then a
                // synchronous D2H readback of the logits buffer.
                let _guard =
                    CtxGuard::set_current(dev_idx).expect("reinfer-cuda: set current device");
                let hb = HostBuffer::alloc(bytes).expect("reinfer-cuda: host logits buffer");
                let rc = unsafe {
                    cudarc::runtime::sys::cudaMemcpy(
                        hb.as_ptr() as *mut c_void,
                        ptr as *const c_void,
                        bytes,
                        cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                rc.result().map_err(from_runtime_error).expect("reinfer-cuda: logits D2H copy");
                unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, vocab).to_vec() }
            },
        )
    }

    /// 单 token 步进：写 KV（pos）并返回 logits（f32 行主序 [vocab]）。
    pub fn step(&mut self, token: u32, pos: usize, kv_len: usize) -> Result<Vec<f32>, EngineError> {
        self.step_impl(token, pos, kv_len, &mut None, &mut None)
    }

    /// 步进 + 每层 hidden 轨迹（调试锚——逐层 vs CPU 参考定位）。
    pub fn step_trace(
        &mut self,
        token: u32,
        pos: usize,
        kv_len: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), EngineError> {
        let mut trace = Some(Vec::new());
        let logits = self.step_impl(token, pos, kv_len, &mut trace, &mut None)?;
        Ok((logits, trace.unwrap_or_default()))
    }

    /// 步进 + 细粒度轨迹（debug 锚：attn_norm 后、q（rope 后）、attn 后）。
    pub fn step_trace_detail(
        &mut self,
        token: u32,
        pos: usize,
        kv_len: usize,
    ) -> Result<(Vec<f32>, Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>), EngineError> {
        let mut det = Some(Vec::new());
        let logits = self.step_impl(token, pos, kv_len, &mut None, &mut det)?;
        Ok((logits, det.unwrap_or_default()))
    }

    /// 步进实现（trace 每层末回读 x；detail 记录 xn/q_rope/attn 三项）。
    ///
    /// 006 T3E / S1-3: 首个进入桶[kv_len] 的 plain 步尝试图捕获
    /// （trace/detail 锚恒 eager）；捕获成功 → 绑定 exec，后续同桶步走
    /// 图重放（`replay_step`）；重放失败 → 丢弃 exec + 桶标记失败 +
    /// eager 回退 + 计数（本进程不重试该桶）。捕获/重放失败时
    /// `step_decode_launches` 恒执行——数值基线不变，图接入不得改变单步
    /// 数值语义。
    #[allow(clippy::too_many_arguments)]
    fn step_impl(
        &mut self,
        token: u32,
        pos: usize,
        kv_len: usize,
        trace: &mut Option<Vec<Vec<f32>>>,
        detail: &mut Option<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>>,
    ) -> Result<Vec<f32>, EngineError> {
        if token as usize >= self.cfg.vocab_size {
            return Err(EngineError::EmbeddingOov(token));
        }
        let stream = self.stream.clone();

        // Graph V2: the C3 cells hold this step's values during capture AND
        // replay (the captured nodes permanently reference the cells and
        // read their contents at every launch — no refresh is involved), so
        // write them before the graph decision. The decl-driven launch
        // path, the capture closure and the replay all read the same cells.
        if let Some(decl) = self.graph_decl.as_mut() {
            decl.write_step(token, pos, self.layers.len(), self.pp);
            let lens: u32 = kv_len as u32;
            // SAFETY: lens_hb is 4 pinned bytes; the captured H2D memcpy
            // node re-reads this cell at every replay.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &lens as *const u32 as *const u8,
                    self.lens_hb.as_ptr() as *mut u8,
                    4,
                )
            }
        }

        let mut replayed = false;
        if trace.is_none() && detail.is_none() {
            let graph_on = self.graph.enabled() && self.graph_decl.is_some();
            if graph_on {
                let b = bucket_index(kv_len as u32);
                if self.graph_failed.contains(&b) {
                    // Known-failed bucket — stay eager (no in-process retry).
                } else if self.graph_execs.contains_key(b) {
                    match self.replay_step(&stream, b) {
                        Ok(()) => replayed = true,
                        Err(e) => {
                            // Fail-closed: the exec is the only way through
                            // this bucket — drop it, mark the bucket failed
                            // and serve the step eager.
                            self.graph_execs.remove(b);
                            self.graph_failed.insert(b);
                            self.graph_eager_fallbacks += 1;
                            eprintln!(
                                "reinfer-cuda graph: bucket {b} (seq_len {kv_len}) replay \
                                 failed — eager fallback: {e}"
                            );
                        }
                    }
                } else {
                    self.try_graph_capture(token, pos, kv_len);
                }
            }
        }

        // eager 序列——与捕获闭包共用 step_decode_launches（同发射序列；
        // 捕获成功后本步仍走 eager：捕获消费了发射，eager 出真值）。gather
        // 在闭包内（embed 行读是步序列第一发射——若在 begin_capture 前发射，
        // 失败捕获会把在途 gather 吸收进图并随图丢弃，x 变垃圾）。
        if !replayed {
            self.step_decode_launches(&stream, token, pos, kv_len, trace, detail)?;
        }
        stream.synchronize()?;
        copy(
            &mut MemRef::Host(&self.logits_host),
            &MemRef::Device(&self.logits),
            self.cfg.vocab_size * 4,
            None,
        )?;
        Ok(unsafe {
            std::slice::from_raw_parts(self.logits_host.as_ptr() as *const f32, self.cfg.vocab_size)
                .to_vec()
        })
    }

    /// Graph V2 decl-driven launch: every declared custom node is launched
    /// with `kernelParams[i] = &cell_i` — the stable C3 argument cells — so
    /// a capture records the cell addresses permanently (the driver's
    /// capture-time param storage is durable; replay then needs no refresh:
    /// `write_step` updates the cell contents per step and the nodes read
    /// them at every launch). The launch order mirrors the declaration
    /// (== capture node order) and the eager launch-fn sequence exactly:
    /// same kernels (the specs' `CUkernel` handles, converted to
    /// `CUfunction` once at build), same geometry (declared grid/block/
    /// shared — the flash node launches its real 512 threads / dynamic
    /// smem), same argument values (the cells hold the eager arguments).
    /// `upload_lens` runs first like the eager path (the captured H2D
    /// memcpy node re-reads lens_hb at every replay).
    ///
    /// Any launch error is propagated (the caller falls back to the
    /// launch-fn sequence — the fail-closed reference).
    fn launch_decode_decl(&self, stream: &CudaStream, kv_len: usize) -> Result<(), EngineError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let decl = self.graph_decl.as_ref().expect("graph decl present");
        let dev = self.dev;
        self.upload_lens(stream, kv_len as u32)?;
        for (node, spec) in decl.specs().iter().enumerate() {
            let base = decl.cell_off[node];
            let slots = layout_slots(&spec.layout);
            // The node's stable args array — `&arg_slots[base..]` lives as
            // long as the decl (the driver records the array *pointer* at
            // capture, so transient arrays would be read as reused garbage
            // at replay — see `GraphStepDecl::arg_slots`).
            let args = &decl.arg_slots[base..base + slots];
            let kernel = KernelFn::from_raw(decl.launch_fns[node]);
            // SAFETY: decl-owned kernels/cells/args arrays outlive the
            // launches (cells are stable — the updates and arg_slots are
            // built after the last push); stream and context valid; args
            // length == the kernel's arity (the layout mirrors the launch
            // exactly).
            unsafe {
                launch_fmha(
                    kernel,
                    stream,
                    dev,
                    spec.grid.x,
                    spec.grid.y,
                    spec.grid.z,
                    spec.block.x,
                    spec.shared,
                    args.as_ptr() as *mut *mut c_void,
                )?;
            }
        }
        Ok(())
    }

    /// 006 T3E: decode 步内核发射序列（embed gather + 28 层 + final norm +
    /// lm_head）——eager 与图捕获共用同一执行体。gather 必须是步序列第一
    /// 发射且在闭包内：若在 begin_capture 前发射，失败捕获会把在途 gather
    /// 吸收进图并随图丢弃（eager 路径不重发 gather → x 变垃圾）。capture
    /// 期合法面：仅内核 launch + 流内异步 H2D（预分配 pinned 缓冲；capture
    /// 期禁止 cudaMallocHost 与同步点——流内排序取代旧默认流 + 同步，数值
    /// 语义逐位不变）。
    #[allow(clippy::too_many_arguments)]
    fn step_decode_launches(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: usize,
        kv_len: usize,
        trace: &mut Option<Vec<Vec<f32>>>,
        detail: &mut Option<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>>,
    ) -> Result<(), EngineError> {
        if self.parity_f32 {
            // 014 S0-3b: parity-f32 criterion tier — the trace/detail anchors
            // are f16-mode only (they read back the f16 buffers); the f32
            // channel returns plain logits.
            return self.step_decode_launches_f32(stream, token, pos, kv_len);
        }
        // Graph V2: with an all-custom declaration (and the flash decode
        // path active) the step launches through the declaration — every
        // node's kernelParams entries are the stable C3 cell addresses, so
        // a capture records the cells (replay then needs no refresh) and
        // the eager fallback runs the same cell-args launches (bit-identical
        // numerics — the cells mirror the launch-fn arguments exactly). Any
        // launch error falls through to the launch-fn sequence below (the
        // fail-closed reference, with its flash->naive fallback).
        let use_decl = self.graph.enabled()
            && self.graph_decl.as_ref().is_some_and(|d| d.is_all_custom())
            && !decode_flash_disabled_from_env(std::env::var(DECODE_FLASH_ENV).ok().as_deref());
        if use_decl && self.launch_decode_decl(stream, kv_len).is_ok() {
            return Ok(());
        }
        // S1-9: fused decode step (REINFER_FUSED, default on) — the whole
        // layer runs 8 nodes instead of 27 (bit-identical to the split
        // sequence; see fused.rs). The graph declaration mirrors this
        // sequence when fused is loaded (the decl-driven path above). A
        // fused launch failure (e.g. smem/launch-capability limits) falls
        // through to the split sequence below — profiler begin_step
        // resets the counters, so the fallback run is clean.
        if self.fused.is_some() {
            match self.step_decode_launches_fused(stream, token, pos, kv_len, trace, detail) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!(
                        "reinfer-cuda: fused decode step failed ({e}); running the split sequence"
                    );
                }
            }
        }
        let dev = self.dev;
        self.prof.begin_step(stream)?;
        // embed 行 → x（步序列第一发射——必须在捕获闭包内；见 step_impl）。
        self.kernels.launch_gather(
            dev,
            stream,
            self.embed.as_ptr() as *const u16,
            self.x.as_ptr() as *mut u16,
            token,
            self.cfg.hidden_size as u32,
        )?;
        self.prof.count(SEG_SMALL);
        let h = self.cfg.hidden_size;
        let nqk = self.cfg.q_heads * self.cfg.head_dim;
        let kvk = self.cfg.kv_heads * self.cfg.head_dim;
        let ffn = self.cfg.ffn_hidden;
        let d = self.cfg.head_dim;
        let kv_heads = self.cfg.kv_heads;
        let q_heads = self.cfg.q_heads;
        let ratio = q_heads / kv_heads;
        let total_pages = self.cfg.n_layer * self.pp;
        let eps = self.cfg.rms_eps;
        let eta = self.cfg.rope_theta;

        let lp = pos / BLOCK_LEN;
        let off = pos % BLOCK_LEN;

        // 步内恒量（kv_len 不变）：lens 每步只上传一次（此前每层一次 = 28 次
        // 重复 H2D——S1-2 发射数削减）。
        self.upload_lens(stream, kv_len as u32)?;
        self.prof.count(SEG_SMALL);

        // 层循环经裸指针索引：权重引用不持 borrow，profile 段标记与层内核
        // 发射可以交错（borrow checker 不允许 `w` 与 `&mut self.prof` 共存）。
        let layers = self.layers.as_ptr();
        let n_layers = self.layers.len();
        for li in 0..n_layers {
            let w = unsafe { &*layers.add(li) };
            let pl = &self.plans.layers[li];
            // attn norm
            self.kernels.launch_rms_norm(
                dev,
                stream,
                self.x.as_ptr() as *const u16,
                self.xn.as_ptr() as *mut u16,
                w.attn_norm.as_ptr() as *const u16,
                h as u32,
                eps,
            )?;
            self.prof.count(SEG_SMALL);
            self.prof.end_segment(SEG_SMALL, stream)?;
            let x0_snap = if detail.is_some() { self.readback_f16(&self.x) } else { None };
            let xn_snap = if detail.is_some() { self.readback_f16(&self.xn) } else { None };
            // q/k/v 投影（32F compute → cast f16；稳定参数格执行——S1-2）
            self.prof.start_segment(SEG_QKV, stream)?;
            self.gemm_exec_plan(stream, &pl.q)?;
            self.prof.count(SEG_QKV);
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                self.c_q.as_ptr() as *const f32,
                self.q.as_ptr() as *mut u16,
                nqk as u32,
            )?;
            self.prof.count(SEG_QKV);
            self.gemm_exec_plan(stream, &pl.k)?;
            self.prof.count(SEG_QKV);
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                self.c_k.as_ptr() as *const f32,
                self.k.as_ptr() as *mut u16,
                kvk as u32,
            )?;
            self.prof.count(SEG_QKV);
            self.gemm_exec_plan(stream, &pl.v)?;
            self.prof.count(SEG_QKV);
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                self.c_v.as_ptr() as *const f32,
                self.v.as_ptr() as *mut u16,
                kvk as u32,
            )?;
            self.prof.count(SEG_QKV);
            self.prof.end_segment(SEG_QKV, stream)?;

            // q/k head norm（Qwen3 系：RoPE 前）
            self.prof.start_segment(SEG_ATTN, stream)?;
            if let (Some(qn), Some(kn)) = (&w.q_norm, &w.k_norm) {
                self.kernels.launch_rms_heads(
                    dev,
                    stream,
                    self.q.as_ptr() as *const u16,
                    self.q.as_ptr() as *mut u16,
                    qn.as_ptr() as *const u16,
                    q_heads as u32,
                    d as u32,
                    eps,
                )?;
                self.prof.count(SEG_ATTN);
                self.kernels.launch_rms_heads(
                    dev,
                    stream,
                    self.k.as_ptr() as *const u16,
                    self.k.as_ptr() as *mut u16,
                    kn.as_ptr() as *const u16,
                    kv_heads as u32,
                    d as u32,
                    eps,
                )?;
                self.prof.count(SEG_ATTN);
            }
            // RoPE + scale（S1-2 融合：q/k 各一次整批发射，替代 32 次逐头
            // rope + 1 次 scale；scale 折入 q 发射——见 rope_heads_f16 注）。
            let qs = 1.0 / (d as f32).sqrt();
            self.kernels.launch_rope_heads(
                dev,
                stream,
                self.q.as_ptr() as *mut u16,
                q_heads as u32,
                (d / 2) as u32,
                pos as u32,
                eta,
                qs,
            )?;
            self.prof.count(SEG_ATTN);
            self.kernels.launch_rope_heads(
                dev,
                stream,
                self.k.as_ptr() as *mut u16,
                kv_heads as u32,
                (d / 2) as u32,
                pos as u32,
                eta,
                1.0,
            )?;
            self.prof.count(SEG_ATTN);
            let q_snap = if detail.is_some() { self.readback_f16(&self.q) } else { None };
            // KV 写（物理页 = li*pp + lp；K/V 连续两区）
            let phys = (li * self.pp + lp) as u32;
            self.kernels.launch_kv_write(
                dev,
                stream,
                self.k.as_ptr() as *const u16,
                self.v.as_ptr() as *const u16,
                self.kv.data.as_ptr() as *mut u16,
                phys,
                off as u32,
                BLOCK_LEN as u32,
                kv_heads as u32,
                d as u32,
                total_pages as u32,
            )?;
            self.prof.count(SEG_ATTN);

            // decode —— 006-2 T2 flash 档（静态恒等页表 → identity 连续读
            // 快捷；smem 预算/launch 失败 → naive 核回退 + 计数）。页表传
            // 恒等表的 li*pp 偏移（S1-2：内容 = 恒等映射 li*pp + j），不再
            // 做每层 H2D 上传。REINFER_DECODE_FLASH=off → 直接走 naive 档
            // （A/B 文本/attn 对照面；显式关闭不算回退，计数器不动）。
            let decode_pages = unsafe { (self.pages_dev.as_ptr() as *const u32).add(li * self.pp) };
            if decode_flash_disabled_from_env(std::env::var(DECODE_FLASH_ENV).ok().as_deref()) {
                self.decode.launch_decode_step_gqa(
                    dev,
                    self.q.as_ptr() as *const u16,
                    decode_pages,
                    self.kv.data.as_ptr() as *const u16,
                    self.lens_dev.as_ptr() as *const u32,
                    self.scores.as_ptr() as *mut f32,
                    self.attn.as_ptr() as *mut u16,
                    1,
                    q_heads as u32,
                    d as u32,
                    BLOCK_LEN as u32,
                    ratio as u32,
                    kv_heads as u32,
                    (self.cfg.n_layer * self.pp) as u32,
                    total_pages as u32,
                )?;
            } else {
                let flash = self.decode.launch_decode_step_gqa_flash(
                    dev,
                    self.q.as_ptr() as *const u16,
                    decode_pages,
                    self.kv.data.as_ptr() as *const u16,
                    self.lens_dev.as_ptr() as *const u32,
                    self.attn.as_ptr() as *mut u16,
                    1,
                    q_heads as u32,
                    d as u32,
                    BLOCK_LEN as u32,
                    ratio as u32,
                    kv_heads as u32,
                    (self.pp * BLOCK_LEN) as u32, // token cap (smem/guard)
                    total_pages as u32,
                    1, // identity page table (S1-2 static identity mapping)
                );
                if let Err(e) = flash {
                    self.decode_flash_fallbacks += 1;
                    eprintln!(
                        "reinfer-cuda: decode flash attn fallback (layer {li}): {e} — naive GQA"
                    );
                    self.decode.launch_decode_step_gqa(
                        dev,
                        self.q.as_ptr() as *const u16,
                        decode_pages,
                        self.kv.data.as_ptr() as *const u16,
                        self.lens_dev.as_ptr() as *const u32,
                        self.scores.as_ptr() as *mut f32,
                        self.attn.as_ptr() as *mut u16,
                        1,
                        q_heads as u32,
                        d as u32,
                        BLOCK_LEN as u32,
                        ratio as u32,
                        kv_heads as u32,
                        (self.cfg.n_layer * self.pp) as u32,
                        total_pages as u32,
                    )?;
                }
            }
            self.prof.count(SEG_ATTN);
            self.prof.end_segment(SEG_ATTN, stream)?;

            let attn_snap = if detail.is_some() { self.readback_f16(&self.attn) } else { None };
            if let (Some(det), Some(x0), Some(xn), Some(qq), Some(aa)) =
                (detail.as_mut(), x0_snap, xn_snap, q_snap, attn_snap)
            {
                det.push((x0, xn, qq, aa));
            }
            // o 投影 → FFN（S1-4：o 残差 add + ffn rms 融合为一发射——
            // fused_add_rms_f16，替代 add_cast + rms_norm 两发射）
            self.prof.start_segment(SEG_O, stream)?;
            self.gemm_exec_plan(stream, &pl.o)?;
            self.prof.count(SEG_O);
            self.prof.end_segment(SEG_O, stream)?;
            self.kernels.launch_fused_add_rms(
                dev,
                stream,
                self.x.as_ptr() as *mut u16,
                self.c_o.as_ptr() as *const f32,
                self.xn.as_ptr() as *mut u16,
                w.ffn_norm.as_ptr() as *const u16,
                h as u32,
                eps,
            )?;
            self.prof.count(SEG_FFN_RMS);
            // FFN 组（S1-4：gate/up GEMM + 融合 cast-swiglu——fused_cast_swiglu_f16，
            // 替代 cast_gate + cast_up + swiglu 三发射；本节内三子段：
            // ffn_gu（gate/up GEMM）/ ffn_d（swiglu + down GEMM）/ ffn_rms
            // （down 残差 add_cast））
            self.prof.start_segment(SEG_FFN_GU, stream)?;
            self.gemm_exec_plan(stream, &pl.gate)?;
            self.prof.count(SEG_FFN_GU);
            self.gemm_exec_plan(stream, &pl.up)?;
            self.prof.count(SEG_FFN_GU);
            self.prof.end_segment(SEG_FFN_GU, stream)?;
            self.prof.start_segment(SEG_FFN_D, stream)?;
            self.kernels.launch_fused_cast_swiglu(
                dev,
                stream,
                self.c_g.as_ptr() as *const f32,
                self.c_u.as_ptr() as *const f32,
                self.down.as_ptr() as *mut u16,
                ffn as u32,
            )?;
            self.prof.count(SEG_FFN_D);
            self.gemm_exec_plan(stream, &pl.down)?;
            self.prof.count(SEG_FFN_D);
            self.prof.end_segment(SEG_FFN_D, stream)?;
            self.prof.start_segment(SEG_FFN_RMS, stream)?;
            // down 残差：cast+add 融合（add_cast_f16——x 就地累加）
            self.kernels.launch_add_cast(
                dev,
                stream,
                self.x.as_ptr() as *mut u16,
                self.c_d.as_ptr() as *const f32,
                h as u32,
            )?;
            self.prof.count(SEG_FFN_RMS);
            self.prof.end_segment(SEG_FFN_RMS, stream)?;
            if let Some(tr) = trace {
                stream.synchronize()?;
                let hb = HostBuffer::alloc(h * 2)?;
                copy(&mut MemRef::Host(&hb), &MemRef::Device(&self.x), h * 2, None)?;
                stream.synchronize()?;
                let row: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(hb.as_ptr() as *const u16, h)
                        .iter()
                        .map(|v| {
                            // 粗读位转 f32（engine 内核同构——trace 档）
                            f16_bits_to_f32(*v)
                        })
                        .collect()
                };
                tr.push(row);
            }
        }

        // final norm → lm_head
        self.prof.start_segment(SEG_LM_HEAD, stream)?;
        self.kernels.launch_rms_norm(
            dev,
            stream,
            self.x.as_ptr() as *const u16,
            self.xn.as_ptr() as *mut u16,
            self.final_norm.as_ptr() as *const u16,
            h as u32,
            eps,
        )?;
        self.prof.count(SEG_LM_HEAD);
        self.gemm_exec_plan(stream, &self.plans.lm_head)?;
        self.prof.count(SEG_LM_HEAD);
        self.prof.end_segment(SEG_LM_HEAD, stream)?;
        // 步收口：host 侧 wall 时钟（末发射后）+ 流同步折段（每步一次 sync
        // ——原步末 stream.synchronize 语义不变，finalize 内同步后步骤即可
        // 回读 logits）。
        self.prof.end_wall();
        self.prof.finalize(stream)?;
        Ok(())
    }

    /// S1-9: fused decode step — 8 nodes/layer instead of 27, bit-identical
    /// to the split sequence (fused.rs / decode_fused_kernels.cu). Launch
    /// order is the graph declaration's mirror:
    /// gather, rms_attn(0), per layer [p1_qkv (phase-1 of q/k/v), p2_qkv
    /// (reductions + casts + head-norm + rope), kv_write, flash decode,
    /// p2_add_rms (o residual + ffn norm), p2_gu_swiglu, p1_ogud (phase-1
    /// of o/g/u/d — reads attn post-flash, p2_o's ffn-normed xn and
    /// p2_gu's swiglu output), p2_add_rms (down residual + next attn
    /// norm)], lm_head phase pair. The detail anchors read back
    /// at the same pipeline points as the split path (x0/xn after rms,
    /// q after rope, attn after decode).
    #[allow(clippy::too_many_arguments)]
    fn step_decode_launches_fused(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: usize,
        kv_len: usize,
        trace: &mut Option<Vec<Vec<f32>>>,
        detail: &mut Option<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>>,
    ) -> Result<(), EngineError> {
        let dev = self.dev;
        let h = self.cfg.hidden_size;
        let nqk = self.cfg.q_heads * self.cfg.head_dim;
        let kvk = self.cfg.kv_heads * self.cfg.head_dim;
        let d = self.cfg.head_dim;
        let kv_heads = self.cfg.kv_heads;
        let q_heads = self.cfg.q_heads;
        let ratio = q_heads / kv_heads;
        let total_pages = self.cfg.n_layer * self.pp;
        let eps = self.cfg.rms_eps;
        let eta = self.cfg.rope_theta;
        let qs = 1.0 / (d as f32).sqrt();
        let max_kv = (self.pp * BLOCK_LEN) as u32;
        let hn = if self.cfg.head_norm { 1u32 } else { 0u32 };

        let fused = self.fused.as_ref().expect("fused loaded");
        let g = fused.geom();
        self.prof.begin_step(stream)?;
        // embed 行 → x（步序列第一发射——与 split 路径同序；捕获闭包内）。
        self.kernels.launch_gather(
            dev,
            stream,
            self.embed.as_ptr() as *const u16,
            self.x.as_ptr() as *mut u16,
            token,
            h as u32,
        )?;
        self.prof.count(SEG_SMALL);
        // attn rms(0)（层 0 的 qkv 读 xn —— split 路径在层循环首做同样的事）
        let layers = self.layers.as_ptr();
        let n_layers = self.layers.len();
        let w0 = unsafe { &*layers };
        self.kernels.launch_rms_norm(
            dev,
            stream,
            self.x.as_ptr() as *const u16,
            self.xn.as_ptr() as *mut u16,
            w0.attn_norm.as_ptr() as *const u16,
            h as u32,
            eps,
        )?;
        self.prof.count(SEG_SMALL);
        self.prof.end_segment(SEG_SMALL, stream)?;
        // 步内恒量（kv_len 不变）：lens 每步只上传一次。
        self.upload_lens(stream, kv_len as u32)?;
        self.prof.count(SEG_SMALL);

        for li in 0..n_layers {
            let w = unsafe { &*layers.add(li) };
            let x0_snap = if detail.is_some() { self.readback_f16(&self.x) } else { None };
            let xn_snap = if detail.is_some() { self.readback_f16(&self.xn) } else { None };
            // 1. phase-1 of the q/k/v plans — reads xn (the layer input:
            //    rms(0) or the previous layer's p2_down norm). The g/u/d
            //    partials run later, after their producers (p2_o, p2_gu);
            //    the o phase-1 is folded into the fused flash kernel below
            //    — same input values the split path reads at those points.
            self.prof.start_segment(SEG_QKV, stream)?;
            let table = unsafe { (g.tables as *const PlanRow).add(li * 7) };
            fused.launch_p1(stream, table, 3, g.grid_qkv[li])?;
            self.prof.count(SEG_QKV);
            // 2. q/k/v phase-2 + casts + head-norm + rope (one launch)
            let (wq, wk) = match (&w.q_norm, &w.k_norm) {
                (Some(qn), Some(kn)) => {
                    (qn.as_ptr() as *const u16, kn.as_ptr() as *const u16)
                }
                _ => (std::ptr::null(), std::ptr::null()),
            };
            fused.launch_p2_qkv(
                stream,
                g.pq,
                g.pk,
                g.pv,
                self.q.as_ptr() as *mut u16,
                self.k.as_ptr() as *mut u16,
                self.v.as_ptr() as *mut u16,
                wq,
                wk,
                nqk as u32,
                kvk as u32,
                kvk as u32,
                g.nslabs_q,
                g.nslabs_k,
                g.nslabs_v,
                d as u32,
                (d / 2) as u32,
                pos as u32,
                eta,
                qs,
                1.0,
                eps,
                hn,
                g.grid_qkv_p2,
            )?;
            self.prof.count(SEG_QKV);
            self.prof.end_segment(SEG_QKV, stream)?;
            let q_snap = if detail.is_some() { self.readback_f16(&self.q) } else { None };
            // 3. fused decode — one kernel: idempotent per-block writes of
            //    the current kv slot (phys = li*pp + kv_len/block, derived
            //    internally from kv_len exactly like kv_write_row) + flash
            //    attention + o phase-1. The o phase-1 reads the
            //    just-written attention output and writes per-(col, slab)
            //    partials byte-identically to the split gemv_m1_f16f32
            //    tree (tile assignment differs per block; per-tile values
            //    are identical). grid = B*QH blocks.
            self.prof.start_segment(SEG_ATTN, stream)?;
            let decode_pages = unsafe { (self.pages_dev.as_ptr() as *const u32).add(li * self.pp) };
            self.decode.launch_decode_step_gqa_flash_fused(
                dev,
                self.q.as_ptr() as *const u16,
                decode_pages,
                self.kv.data.as_ptr() as *const u16,
                self.lens_dev.as_ptr() as *const u32,
                self.attn.as_ptr() as *mut u16,
                1,
                q_heads as u32,
                d as u32,
                BLOCK_LEN as u32,
                ratio as u32,
                kv_heads as u32,
                max_kv,
                total_pages as u32,
                1, // identity page table (S1-2 static identity mapping)
                self.k.as_ptr() as *const u16,
                self.v.as_ptr() as *const u16,
            )?;
            self.prof.count(SEG_ATTN);
            self.prof.end_segment(SEG_ATTN, stream)?;
            let attn_snap = if detail.is_some() { self.readback_f16(&self.attn) } else { None };
            // 4. o phase-1 — a separate node (NOT inside the flash kernel):
            //    it reads the FULL attention row (all B*QH heads' outputs),
            //    which each flash block's phase C writes only for its own
            //    head — folding it in would race across blocks. As its own
            //    stream-ordered launch it is race-free and byte-identical.
            self.prof.start_segment(SEG_O, stream)?;
            fused.launch_p1(stream, unsafe { table.add(3) }, 1, g.grid_o[li])?;
            self.prof.count(SEG_O);
            if let (Some(det), Some(x0), Some(xn), Some(qq), Some(aa)) =
                (detail.as_mut(), x0_snap, xn_snap, q_snap, attn_snap)
            {
                det.push((x0, xn, qq, aa));
            }
            // 5. o phase-2 + residual add (add_cast 语义) + ffn rms
            self.prof.start_segment(SEG_O, stream)?;
            fused.launch_p2_add_rms(
                stream,
                g.po,
                self.x.as_ptr() as *mut u16,
                self.xn.as_ptr() as *mut u16,
                w.ffn_norm.as_ptr() as *const u16,
                h as u32,
                g.nslabs_o,
                eps,
            )?;
            self.prof.count(SEG_O);
            self.prof.end_segment(SEG_O, stream)?;
            // 6. gate/up phase-1 — reads xn (p2_o's ffn-normed x), the
            //    exact value the split path reads here. (ffn_gu sub-segment)
            self.prof.start_segment(SEG_FFN_GU, stream)?;
            fused.launch_p1(stream, unsafe { table.add(4) }, 2, g.grid_gu[li])?;
            self.prof.count(SEG_FFN_GU);
            self.prof.end_segment(SEG_FFN_GU, stream)?;
            // 7. gate/up phase-2 + fused cast-SiLU-GLU + down phase-1 —
            //    one merged kernel (`gemv_p2_gu_p1d_swiglu`, grid =
            //    ncols_d*nslabs_d, the split's full down phase-1 tile
            //    grid): each block redundantly writes the 256-col
            //    phase-1 stripe its phase-2 k-range lies in, then
            //    computes the down tile (bx/nslabs_d, bx%nslabs_d)
            //    (valid iff every slab's k-range fits in its block's
            //    stripe — checked in build_plans). Block-local, same
            //    arithmetic as the split p2_gu + p1_d pair.
            self.prof.start_segment(SEG_FFN_D, stream)?;
            fused.launch_p2_gu_d(stream, unsafe { table.add(4) }, g.grid_gu_p2)?;
            self.prof.count(SEG_FFN_D);
            self.prof.end_segment(SEG_FFN_D, stream)?;
            // 8. down phase-2 + residual add + next layer's attn rms
            self.prof.start_segment(SEG_FFN_RMS, stream)?;
            let wnext = if li + 1 < n_layers {
                unsafe { &*layers.add(li + 1) }.attn_norm.as_ptr() as *const u16
            } else {
                self.final_norm.as_ptr() as *const u16
            };
            fused.launch_p2_add_rms(
                stream,
                g.pd,
                self.x.as_ptr() as *mut u16,
                self.xn.as_ptr() as *mut u16,
                wnext,
                h as u32,
                g.nslabs_d,
                eps,
            )?;
            self.prof.count(SEG_FFN_RMS);
            self.prof.end_segment(SEG_FFN_RMS, stream)?;
            if let Some(tr) = trace {
                stream.synchronize()?;
                let hb = HostBuffer::alloc(h * 2)?;
                copy(&mut MemRef::Host(&hb), &MemRef::Device(&self.x), h * 2, None)?;
                stream.synchronize()?;
                let row: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(hb.as_ptr() as *const u16, h)
                        .iter()
                        .map(|v| {
                            // 粗读位转 f32（engine 内核同构——trace 档）
                            f16_bits_to_f32(*v)
                        })
                        .collect()
                };
                tr.push(row);
            }
        }
        // final norm + lm_head：末层 p2_down 的 rms 已用 final_norm 把
        // xn 写好（wnext = final_norm），lm p1 直接读 xn（p1 走 multi 核
        // 单行；p2 复用 reduce 核）。
        self.prof.start_segment(SEG_LM_HEAD, stream)?;
        fused.launch_p1(stream, g.lm_table, 1, g.grid_lm)?;
        self.prof.count(SEG_LM_HEAD);
        fused.launch_reduce(
            stream,
            g.plm,
            self.logits.as_ptr() as *mut f32,
            self.cfg.vocab_size as u32,
            1,
            g.grid_lm,
        )?;
        self.prof.count(SEG_LM_HEAD);
        self.prof.end_segment(SEG_LM_HEAD, stream)?;
        self.prof.end_wall();
        self.prof.finalize(stream)?;
        Ok(())
    }

    /// 014 S0-3b: parity-f32 decode step — the whole layer chain runs with f32
    /// intermediates (the llama.cpp CPU referee's precision profile): q/k/v
    /// stay in the f32 GEMM output buffers (no f16 cast), RoPE/head-norm/
    /// scale/attention-output/residual/FFN all f32. Weights are the f32
    /// expansion of the f16 values (bit-identical); KV stays f16 — rounded
    /// once at the write, like the referee's f16 KV cache. Reused criterion
    /// kernels: DiffKernels f32 rms/rope/add (014 T7), CUBLAS_COMPUTE_32F
    /// gemms, and decode_step_gqa_f32 (f32 q/out, f16 KV in).
    fn step_decode_launches_f32(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: usize,
        kv_len: usize,
    ) -> Result<(), EngineError> {
        let dev = self.dev;
        self.prof.begin_step(stream)?;
        let h = self.cfg.hidden_size;
        let nqk = self.cfg.q_heads * self.cfg.head_dim;
        let kvk = self.cfg.kv_heads * self.cfg.head_dim;
        let ffn = self.cfg.ffn_hidden;
        let d = self.cfg.head_dim;
        let kv_heads = self.cfg.kv_heads;
        let q_heads = self.cfg.q_heads;
        let ratio = q_heads / kv_heads;
        let total_pages = self.cfg.n_layer * self.pp;
        let eps = self.cfg.rms_eps;
        let lp = pos / BLOCK_LEN;
        let off = pos % BLOCK_LEN;

        self.kernels.launch_gather_f32(
            dev,
            stream,
            self.embed.as_ptr() as *const f32,
            self.x32.as_ptr() as *mut f32,
            token,
            h as u32,
        )?;
        self.prof.count(SEG_SMALL);

        // lens 每步一次（同 f16 路径；页表恒等——静态上传见 load）。
        self.upload_lens(stream, kv_len as u32)?;
        self.prof.count(SEG_SMALL);

        let layers = self.layers.as_ptr();
        let n_layers = self.layers.len();
        for li in 0..n_layers {
            let w = unsafe { &*layers.add(li) };
            let pl = &self.plans.layers[li];
            // attn norm (f32)
            self.diff.launch_rms_norm(
                dev,
                self.x32.as_ptr() as *const f32,
                w.attn_norm.as_ptr() as *const f32,
                self.xn32.as_ptr() as *mut f32,
                h as u32,
                eps,
            )?;
            self.prof.count(SEG_SMALL);
            self.prof.end_segment(SEG_SMALL, stream)?;
            // q/k/v projections: f32 in / f32 out (no f16 cast) — stable plans
            self.prof.start_segment(SEG_QKV, stream)?;
            self.gemm_exec_plan(stream, &pl.q)?;
            self.prof.count(SEG_QKV);
            self.gemm_exec_plan(stream, &pl.k)?;
            self.prof.count(SEG_QKV);
            self.gemm_exec_plan(stream, &pl.v)?;
            self.prof.count(SEG_QKV);
            self.prof.end_segment(SEG_QKV, stream)?;
            // q/k head norm + RoPE + scale + KV write + decode — ATTN 段
            self.prof.start_segment(SEG_ATTN, stream)?;
            if let (Some(qn), Some(kn)) = (&w.q_norm, &w.k_norm) {
                for hh in 0..q_heads {
                    let qh = unsafe { (self.c_q.as_ptr() as *mut f32).add(hh * d) };
                    self.diff.launch_rms_norm(
                        dev,
                        qh,
                        qn.as_ptr() as *const f32,
                        qh,
                        d as u32,
                        eps,
                    )?;
                    self.prof.count(SEG_ATTN);
                }
                for hh in 0..kv_heads {
                    let kh = unsafe { (self.c_k.as_ptr() as *mut f32).add(hh * d) };
                    self.diff.launch_rms_norm(
                        dev,
                        kh,
                        kn.as_ptr() as *const f32,
                        kh,
                        d as u32,
                        eps,
                    )?;
                    self.prof.count(SEG_ATTN);
                }
            }
            // RoPE (f32, NEOX half-split; q all heads, k kv heads)
            for hh in 0..q_heads {
                let qh = unsafe { (self.c_q.as_ptr() as *mut f32).add(hh * d) };
                self.diff.launch_rope(
                    dev,
                    qh,
                    qh,
                    (d / 2) as u32,
                    pos as u32,
                    self.cfg.rope_theta,
                )?;
                self.prof.count(SEG_ATTN);
            }
            for hh in 0..kv_heads {
                let kh = unsafe { (self.c_k.as_ptr() as *mut f32).add(hh * d) };
                self.diff.launch_rope(
                    dev,
                    kh,
                    kh,
                    (d / 2) as u32,
                    pos as u32,
                    self.cfg.rope_theta,
                )?;
                self.prof.count(SEG_ATTN);
            }
            // attention score scale 1/sqrt(d) (f32; same scale point as the f16 path)
            let qs = 1.0 / (d as f32).sqrt();
            self.kernels.launch_scale_f32(
                dev,
                stream,
                self.c_q.as_ptr() as *mut f32,
                nqk as u32,
                qs,
            )?;
            self.prof.count(SEG_ATTN);
            // KV write: round k/v to f16 once (referee f16 KV parity), then write
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                self.c_k.as_ptr() as *const f32,
                self.k.as_ptr() as *mut u16,
                kvk as u32,
            )?;
            self.prof.count(SEG_ATTN);
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                self.c_v.as_ptr() as *const f32,
                self.v.as_ptr() as *mut u16,
                kvk as u32,
            )?;
            self.prof.count(SEG_ATTN);
            let phys = (li * self.pp + lp) as u32;
            self.kernels.launch_kv_write(
                dev,
                stream,
                self.k.as_ptr() as *const u16,
                self.v.as_ptr() as *const u16,
                self.kv.data.as_ptr() as *mut u16,
                phys,
                off as u32,
                BLOCK_LEN as u32,
                kv_heads as u32,
                d as u32,
                total_pages as u32,
            )?;
            self.prof.count(SEG_ATTN);
            // decode attention (f32 q/out; f16 KV read in-kernel) — 006-2 T2
            // flash 档 + naive 回退（同 f16 路径的接线与计数；同
            // REINFER_DECODE_FLASH=off 显式关闭面）。
            let decode_pages = unsafe { (self.pages_dev.as_ptr() as *const u32).add(li * self.pp) };
            if decode_flash_disabled_from_env(std::env::var(DECODE_FLASH_ENV).ok().as_deref()) {
                self.decode.launch_decode_step_gqa_f32(
                    dev,
                    self.c_q.as_ptr() as *const f32,
                    decode_pages,
                    self.kv.data.as_ptr() as *const u16,
                    self.lens_dev.as_ptr() as *const u32,
                    self.scores.as_ptr() as *mut f32,
                    self.attn32.as_ptr() as *mut f32,
                    1,
                    q_heads as u32,
                    d as u32,
                    BLOCK_LEN as u32,
                    ratio as u32,
                    kv_heads as u32,
                    (self.cfg.n_layer * self.pp) as u32,
                    total_pages as u32,
                )?;
            } else {
                let flash = self.decode.launch_decode_step_gqa_flash_f32(
                    dev,
                    self.c_q.as_ptr() as *const f32,
                    decode_pages,
                    self.kv.data.as_ptr() as *const u16,
                    self.lens_dev.as_ptr() as *const u32,
                    self.attn32.as_ptr() as *mut f32,
                    1,
                    q_heads as u32,
                    d as u32,
                    BLOCK_LEN as u32,
                    ratio as u32,
                    kv_heads as u32,
                    (self.pp * BLOCK_LEN) as u32, // token cap (smem/guard)
                    total_pages as u32,
                    1, // identity page table (S1-2 static identity mapping)
                );
                if let Err(e) = flash {
                    self.decode_flash_fallbacks += 1;
                    eprintln!(
                        "reinfer-cuda: decode flash attn fallback (layer {li}): {e} — naive GQA"
                    );
                    self.decode.launch_decode_step_gqa_f32(
                        dev,
                        self.c_q.as_ptr() as *const f32,
                        decode_pages,
                        self.kv.data.as_ptr() as *const u16,
                        self.lens_dev.as_ptr() as *const u32,
                        self.scores.as_ptr() as *mut f32,
                        self.attn32.as_ptr() as *mut f32,
                        1,
                        q_heads as u32,
                        d as u32,
                        BLOCK_LEN as u32,
                        ratio as u32,
                        kv_heads as u32,
                        (self.cfg.n_layer * self.pp) as u32,
                        total_pages as u32,
                    )?;
                }
            }
            self.prof.count(SEG_ATTN);
            self.prof.end_segment(SEG_ATTN, stream)?;
            // o projection → residual (f32, in-place add)
            self.prof.start_segment(SEG_O, stream)?;
            self.gemm_exec_plan(stream, &pl.o)?;
            self.prof.count(SEG_O);
            self.diff.launch_add_mask(
                dev,
                stream,
                self.x32.as_ptr() as *mut f32,
                self.c_o.as_ptr() as *const f32,
                h as u32,
            )?;
            self.prof.count(SEG_O);
            self.prof.end_segment(SEG_O, stream)?;
            // FFN (f32)
            self.prof.start_segment(SEG_FFN_GU, stream)?;
            self.prof.start_segment(SEG_SMALL, stream)?;
            self.diff.launch_rms_norm(
                dev,
                self.x32.as_ptr() as *const f32,
                w.ffn_norm.as_ptr() as *const f32,
                self.xn32.as_ptr() as *mut f32,
                h as u32,
                eps,
            )?;
            self.prof.count(SEG_SMALL);
            self.prof.end_segment(SEG_SMALL, stream)?;
            self.gemm_exec_plan(stream, &pl.gate)?;
            self.prof.count(SEG_FFN_GU);
            self.gemm_exec_plan(stream, &pl.up)?;
            self.prof.count(SEG_FFN_GU);
            self.prof.end_segment(SEG_FFN_GU, stream)?;
            self.prof.start_segment(SEG_FFN_D, stream)?;
            self.kernels.launch_swiglu_f32(
                dev,
                stream,
                self.c_g.as_ptr() as *const f32,
                self.c_u.as_ptr() as *const f32,
                self.c_g.as_ptr() as *mut f32, // in-place: gate is dead after swiglu
                ffn as u32,
            )?;
            self.prof.count(SEG_FFN_D);
            self.gemm_exec_plan(stream, &pl.down)?;
            self.prof.count(SEG_FFN_D);
            self.prof.end_segment(SEG_FFN_D, stream)?;
            self.prof.start_segment(SEG_FFN_RMS, stream)?;
            self.diff.launch_add_mask(
                dev,
                stream,
                self.x32.as_ptr() as *mut f32,
                self.c_o.as_ptr() as *const f32,
                h as u32,
            )?;
            self.prof.count(SEG_FFN_RMS);
            self.prof.end_segment(SEG_FFN_RMS, stream)?;
        }

        // final norm (f32) → lm_head (f32 in / f32 out)
        self.prof.start_segment(SEG_LM_HEAD, stream)?;
        self.diff.launch_rms_norm(
            dev,
            self.x32.as_ptr() as *const f32,
            self.final_norm.as_ptr() as *const f32,
            self.xn32.as_ptr() as *mut f32,
            h as u32,
            eps,
        )?;
        self.prof.count(SEG_LM_HEAD);
        self.gemm_exec_plan(stream, &self.plans.lm_head)?;
        self.prof.count(SEG_LM_HEAD);
        self.prof.end_segment(SEG_LM_HEAD, stream)?;
        self.prof.end_wall();
        self.prof.finalize(stream)?;
        Ok(())
    }

    /// 006 T3E / S1-3: decode 步图捕获尝试。首个进入桶[kv_len] 的 step 发起
    /// capture（持全局 CAPTURE_LOCK；捕获期 REINFER_GRAPH_NO_OVERLAP 生效；
    /// 闭包 = 同 eager 的 step_decode_launches）。
    ///
    /// S1-3: 声明全步 KernelSpec（`GraphStepDecl`，launch 序镜像）→
    /// `capture` → 成功（节点数 == specs 数）绑定 exec 到桶并计数；
    /// 失败（含 specs 数不匹配——finish 计数校验 fail-closed）→ eager +
    /// 计数 + 该桶本进程不再重试（新桶仍尝试；跨进程自然重试）。
    fn try_graph_capture(&mut self, token: u32, pos: usize, kv_len: usize) {
        let Some(decl) = &self.graph_decl else {
            return; // f32 channel or no graph — always eager
        };
        let pool = self.graph.clone();
        if !pool.enabled() {
            return;
        }
        let b = bucket_index(kv_len as u32);
        if self.graph_failed.contains(&b) {
            return;
        }
        // Graph V2: the decl-driven (cell-args) capture gate — all-custom
        // declaration AND the flash decode path. The declaration only knows
        // the flash node, so with REINFER_DECODE_FLASH=off the capture
        // closure records naive nodes and replay could never be correct —
        // stay eager (no capture, no fallback counting: explicit off).
        let use_decl = decl.is_all_custom()
            && !decode_flash_disabled_from_env(std::env::var(DECODE_FLASH_ENV).ok().as_deref());
        let specs = decl.specs().to_vec();
        // Constant update list (copied — the capture closure borrows self
        // mutably, so the seed must not borrow `decl` across it).
        let updates = decl.updates().to_vec();
        // Per-step refresh nodes (copied for the same reason).
        let refresh: Vec<(usize, usize)> = decl.refresh().to_vec();
        let stream = self.stream.clone();
        let mut no_trace: Option<Vec<Vec<f32>>> = None;
        let mut no_detail: Option<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>> = None;
        let mut step = |s: &CudaStream| {
            self.step_decode_launches(s, token, pos, kv_len, &mut no_trace, &mut no_detail).map_err(
                |e| {
                    eprintln!("reinfer-cuda graph: capture step launch failed: {e}");
                    LaunchError::Fatal
                },
            )
        };
        match pool.capture(&stream, kv_len as u32, &specs, &refresh, &mut step) {
            Ok(mut exec) => {
                self.graph_captures += 1;
                // Graph V2: the capture recorded the C3 cell addresses (the
                // decl-driven launch) — seed the staging from the same
                // constant update list so the first replay is clean: no
                // SetParams, no ExecUpdate, no re-instantiate; replay reads
                // the cell contents (per-step writes) at every launch. A
                // seed failure is fail-closed: drop the exec, mark the
                // bucket failed and serve eager.
                if use_decl {
                    if let Err(e) = exec.seed_staging(&updates) {
                        eprintln!(
                            "reinfer-cuda graph: bucket {b} (seq_len {kv_len}) staging seed \
                             failed — eager: {e}"
                        );
                        self.graph_failed.insert(b);
                        self.graph_eager_fallbacks += 1;
                        return;
                    }
                }
                let nodes = exec.node_count();
                let declared = specs.len();
                if nodes == declared {
                    eprintln!(
                        "reinfer-cuda graph: bucket {b} (seq_len {kv_len}) captured — \
                         {nodes} kernel nodes == {declared} declared specs; replay bound"
                    );
                } else {
                    eprintln!(
                        "reinfer-cuda graph: bucket {b} (seq_len {kv_len}) captured with \
                         {nodes} kernel nodes vs {declared} declared specs — replay served with \
                         the count mismatch (fail-closed on replay path)"
                    );
                }
                self.check_capture_alignment(&exec);
                self.graph_execs.insert(b, exec);
            }
            Err(e) => {
                if self.graph_failed.insert(b) {
                    eprintln!(
                        "reinfer-cuda graph: bucket {b} (seq_len {kv_len}) capture failed — \
                         eager: {e}"
                    );
                }
                self.graph_eager_fallbacks += 1;
            }
        }
    }

    /// S1-3 / Graph V2: replay the bucket's exec for the current step with
    /// the constant update list. The per-step variables (token/pos/phys/
    /// off) were already written into the stable C3 cells by `step_impl`
    /// (and the pinned lens cell with it) — the graph nodes permanently
    /// reference the cells (cell-args capture) and read the current
    /// contents at every launch, so replay is a plain launch: no
    /// SetParams, no ExecUpdate, no re-instantiate. Replay failure is
    /// fail-closed: the caller drops the exec, marks the bucket failed
    /// and serves eager.
    fn replay_step(&mut self, stream: &CudaStream, b: usize) -> Result<(), EngineError> {
        // The graph launch (and its exec drop) requires a current context.
        let _guard = CtxGuard::set_current(self.dev)?;
        let decl = self.graph_decl.as_mut().expect("graph decl present");
        let exec = self.graph_execs.get_mut(b).expect("bucket exec present");
        let updates = decl.updates();
        exec.replay(stream, updates)?;
        self.graph_replays += 1;
        Ok(())
    }

    /// S1-3: post-capture cublas alignment diagnostics. Every
    /// `CublasGemm` spec position is compared against the driver
    /// read-back: pre-13.2 runtimes own the argument values → per-node
    /// m/n/k geometry equality (launch order == declaration order
    /// evidence). 13.x V2 read-backs are metadata-only → values
    /// unavailable; the order assumption (= declaration order) is logged
    /// once. Diagnostics only — capture success does not depend on this.
    fn check_capture_alignment(&self, exec: &GraphExec) {
        let Some(decl) = &self.graph_decl else {
            return;
        };
        let mut aligned = 0usize;
        let mut mismatched = 0usize;
        let mut meta_only = 0usize;
        for (node, spec) in decl.specs().iter().enumerate() {
            if spec.role != NodeRole::CublasGemm {
                continue;
            }
            let Some((dm, dn, dk)) = decl.gemm_cells(node) else {
                continue;
            };
            match exec.node_params(node) {
                Some(rb) => match rb.gemm_mnk(&spec.layout) {
                    Some((m, n, k)) => {
                        if (m, n, k) == (dm, dn, dk) {
                            aligned += 1;
                        } else {
                            mismatched += 1;
                            eprintln!(
                                "reinfer-cuda graph: node {node} gemm read-back ({m},{n},{k}) \
                                 != declared ({dm},{dn},{dk}) — cublas order/geometry \
                                 misaligned with the declaration"
                            );
                        }
                    }
                    None => meta_only += 1,
                },
                None => {
                    eprintln!("reinfer-cuda graph: node {node} cublas read-back missing");
                }
            }
        }
        if meta_only > 0 {
            eprintln!(
                "reinfer-cuda graph: {meta_only} cublas nodes metadata-only (V2 read-back) — \
                 order assumption = declaration order"
            );
        } else if mismatched == 0 {
            eprintln!(
                "reinfer-cuda graph: {aligned} cublas nodes geometry-aligned with the \
                 declaration (launch order == declaration order)"
            );
        }
    }

    /// 006 T3E / S1-3: 图路径 eager 回退累计（捕获失败 + 重放失败；
    /// 诊断/测试可见）。
    #[must_use]
    pub fn graph_eager_fallbacks(&self) -> u64 {
        self.graph_eager_fallbacks
    }

    /// S1-3: 图捕获成功数（绑定到桶的 exec 数；诊断/测试可见）。
    #[must_use]
    pub fn graph_captures(&self) -> u64 {
        self.graph_captures
    }

    /// S1-3: 图重放成功数（诊断/测试可见）。
    #[must_use]
    pub fn graph_replays(&self) -> u64 {
        self.graph_replays
    }

    /// 006-2 T2: decode-attn flash 档回退累计（smem 预算/launch 失败 →
    /// naive GQA；诊断/测试可见）。
    #[must_use]
    pub fn decode_flash_fallbacks(&self) -> u64 {
        self.decode_flash_fallbacks
    }

    /// Batched prefill（006 T1）：单次 forward 处理整条提示——batched QKV
    /// GEMM（m = S 单次调用）+ FMHA 全序列注意力 + 末位 logits + KV 页写
    /// （页序 = li*pp + s/32，与逐 token 路径逐位一致——后续 decode
    /// step_impl 无缝续接）。
    ///
    /// 006 T2/T-305：提供者选择经 `select_fmha(cfg, db, avail)`（TuneDb
    /// 实测优先；无可用档 → 恒 dense）；每次成功执行记录实测 score 到
    /// TuneDb（op 命名空间 "fmha" 与选择器一致；provider=jit_fmha/
    /// jit_dense；host round-trip 计时；进程内 SelectionCache 首测决定——
    /// "首测慢/二测快"）。FMHA 装载/launch 失败 → 自动回退既有逐 token
    /// 步进路径（同输入结果与旧 prefill 逐位一致），不伪造降级。
    pub fn prefill_batch(&mut self, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        if ids.is_empty() {
            return Err(EngineError::Sts("empty prompt".into()));
        }
        if self.parity_f32 {
            // 014 S0-3b: parity-f32 criterion tier — per-token f32 steps
            // (the criterion tier only ever sees B=1 traces; per-token is
            // numerically identical to a batched f32 path and avoids a
            // second batched attention implementation).
            return self.prefill_fallback_f32(ids);
        }
        // 可用性探测：FMHA 惰性装载（失败 → jit_fmha 不可用 → 选择恒 dense）
        // NOTE: serve 的推理在 spawn_blocking worker 线程首次进入本路径——
        // cuModuleLoadData（JLib::from_bytes）绑定当前线程的 CUDA context，
        // 无 CtxGuard 时加载落在 NULL/主 context 上（测试在主线程成功；
        // serve 失败——B 修复点）。加载必须持 CtxGuard 与 step_impl 同款。
        let fmha_loaded = if self.fmha.is_some() && self.prefill.is_some() {
            true
        } else {
            let dev = self.device();
            match CtxGuard::set_current(dev).and_then(|_| {
                PrefillKernels::new(&self.arch, None).and_then(|p| {
                    FmhaKernels::new(&self.arch, None, self.stream.clone()).map(|f| (p, f))
                })
            }) {
                Ok((p, f)) => {
                    self.prefill = Some(p);
                    self.fmha = Some(f);
                    true
                }
                Err(e) => {
                    eprintln!("reinfer-cuda: FMHA load failed ({e}); per-token prefill fallback");
                    false
                }
            }
        };
        let cfg = OpConfig {
            op: "fmha",
            device: DeviceId::new(self.dev),
            in_dt: DType::F16,
            out_dt: DType::F16,
            head_dim: self.cfg.head_dim,
            batch: 1,
            seq: ids.len(),
        };
        let avail = ProviderSet { vendor: false, jit_fmha: fmha_loaded, jit_dense: true };
        match self.sel.select_fmha(&cfg, &self.db, avail) {
            // FMHA 档（TuneDb 实测优先或语义序；Vendor 档本引擎不可用，
            // 选择器不会返回）
            ProviderChoice::JitFmha => self.prefill_fmha_timed(&cfg, ids),
            // 恒 dense（无 GPU/不可用/实测更优）
            _ => self.prefill_dense_timed(&cfg, ids),
        }
    }

    /// FMHA 批 prefill 计时执行 + TuneDb 记录；launch 失败 → 原语义回退
    /// 逐 token dense（回退同样计时记录——成功后数据可用）。
    fn prefill_fmha_timed(&mut self, cfg: &OpConfig, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let t0 = std::time::Instant::now();
        match self.prefill_batch_fmha(ids) {
            Ok(l) => {
                self.record_tuned(cfg, ProviderChoice::JitFmha, t0.elapsed(), true);
                Ok(l)
            }
            Err(e) => {
                eprintln!("reinfer-cuda: FMHA prefill failed ({e}); per-token prefill fallback");
                self.prefill_dense_timed(cfg, ids)
            }
        }
    }

    /// 逐 token dense 回退：计时执行 + TuneDb 记录（provider=jit_dense）。
    fn prefill_dense_timed(
        &mut self,
        cfg: &OpConfig,
        ids: &[u32],
    ) -> Result<Vec<f32>, EngineError> {
        let t0 = std::time::Instant::now();
        let out = self.prefill_fallback(ids);
        self.record_tuned(cfg, ProviderChoice::JitDense, t0.elapsed(), out.is_ok());
        out
    }

    /// 006 T2: 成功执行后记录实测 score（µs，越低越好）并原子保存；保存
    /// 失败静默（调优数据尽力而为，不影响生成）。
    fn record_tuned(
        &mut self,
        cfg: &OpConfig,
        choice: ProviderChoice,
        elapsed: std::time::Duration,
        ok: bool,
    ) {
        if !ok {
            return;
        }
        let score = elapsed.as_secs_f64() * 1e6;
        self.db.record(cfg.op, &reinfer_kernels::tune::shape_key(cfg), choice.tune_name(), score);
        if let Err(e) = self.db.save() {
            eprintln!("reinfer-cuda: tune.json save failed ({e})");
        }
    }

    /// 逐 token 回退路径（既有 step 循环；返回末位 logits）。
    fn prefill_fallback(&mut self, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let mut logits = Vec::new();
        for (i, &tok) in ids.iter().enumerate() {
            logits = self.step(tok, i, i + 1)?;
        }
        Ok(logits)
    }

    /// 014 S0-3b: parity-f32 prefill — per-token steps over the f32 channel
    /// (step dispatches on the parity_f32 flag; writes the same KV slots the
    /// f16 path writes, returns the last-position logits).
    fn prefill_fallback_f32(&mut self, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let mut logits = Vec::new();
        for (i, &tok) in ids.iter().enumerate() {
            logits = self.step(tok, i, i + 1)?;
        }
        Ok(logits)
    }

    /// FMHA 批 prefill 实现（单序列 B=1；缓冲按本次 S 分配）。
    fn prefill_batch_fmha(&mut self, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        for &t in ids {
            if t as usize >= self.cfg.vocab_size {
                return Err(EngineError::EmbeddingOov(t));
            }
        }
        let s = ids.len();
        let pages_need = s.div_ceil(BLOCK_LEN);
        if pages_need > self.pp {
            return Err(EngineError::Sts(format!(
                "prompt {s} tokens needs {pages_need} pages > max_kv pages {}",
                self.pp
            )));
        }
        let dev = self.dev;
        let stream = &self.stream;
        let fmha = self.fmha.as_ref().ok_or_else(|| EngineError::Sts("fmha not loaded".into()))?;
        let pk = self
            .prefill
            .as_ref()
            .ok_or_else(|| EngineError::Sts("prefill kernels not loaded".into()))?;
        let h = self.cfg.hidden_size;
        let nqk = self.cfg.q_heads * self.cfg.head_dim;
        let kvk = self.cfg.kv_heads * self.cfg.head_dim;
        let ffn = self.cfg.ffn_hidden;
        let d = self.cfg.head_dim;
        let kv_heads = self.cfg.kv_heads;
        let q_heads = self.cfg.q_heads;
        let total_pages = self.cfg.n_layer * self.pp;
        let eps = self.cfg.rms_eps;
        let devid = DeviceId::new(dev);
        let a16 = |n: usize| DeviceBuffer::alloc(devid, n * 2).map_err(EngineError::Launch);
        let c32 = |n: usize| DeviceBuffer::alloc(devid, n * 4).map_err(EngineError::Launch);
        // S1-7: fused QKV weight present when the loader built it (all layers
        // uniformly); the separated q/k/v GEMM path stays as the fallback
        // (parity-f32 tier / pre-fusion caches) — same math, same downstream.
        // REINFER_PREFILL_SEP_QKV=1 forces the separated path (A/B
        // microbenchmarking only; never set by the default flow).
        let fused_qkv = self.layers.first().is_some_and(|w| w.qkv_proj.is_some())
            && std::env::var_os("REINFER_PREFILL_SEP_QKV").is_none();
        let x = a16(s * h)?;
        let xn = a16(s * h)?;
        // QKV output buffers: contiguous per-section [s x nqk] q and
        // [s x kvk] k/v — same layout for both legs (the fused path's
        // single cast writes them directly, S1-7).
        let q = Some(a16(s * nqk)?);
        let k = Some(a16(s * kvk)?);
        let v = Some(a16(s * kvk)?);
        let o = a16(s * nqk)?;
        let oadd = a16(s * h)?;
        let gate = a16(s * ffn)?;
        let up = a16(s * ffn)?;
        let down = a16(s * ffn)?;
        let l2 = a16(s * h)?;
        let (c_qkv, c_q, c_k, c_v) = if fused_qkv {
            (Some(c32(s * (nqk + 2 * kvk))?), None, None, None)
        } else {
            (
                None,
                Some(c32(s * nqk)?),
                Some(c32(s * kvk)?),
                Some(c32(s * kvk)?),
            )
        };
        let c_o = c32(s * h)?;
        let c_g = c32(s * ffn)?;
        let c_u = c32(s * ffn)?;
        let c_d = c32(s * h)?;
        let logits = c32(s * self.cfg.vocab_size)?;
        // FMHA 内核把 LSE 按 kBlockM 行块写入（末块含 seqlen_q_rounded 内
        // 的越界行——上游 flash-attn 约定 LSE 缓冲为 rounded 尺寸）；O 仅
        // 写有效行（掩码），故 O 缓冲 s·nqk 足够，LSE 必须取 rounded。
        // S1-7: rounding block = the heuristics-picked variant's kBlockM.
        let sq_r = {
            let bm = self.fmha.as_ref().expect("fmha loaded").block_m(s as u32) as usize;
            s.div_ceil(bm) * bm
        };
        let lse = c32(q_heads * sq_r)?;
        self.prefill_prof.begin(stream, s as u32)?;

        // 批 embed：token 表上传 + 单次 gather（grid = S）
        let toks = {
            let mut raw = Vec::with_capacity(s * 4);
            for t in ids {
                raw.extend_from_slice(&t.to_le_bytes());
            }
            let hb = HostBuffer::alloc(raw.len())?;
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), hb.as_ptr() as *mut u8, raw.len());
            }
            let db = DeviceBuffer::alloc(devid, raw.len())?;
            copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), raw.len(), None)?;
            db
        };
        pk.launch_gather_rows(
            dev,
            stream,
            self.embed.as_ptr() as *const u16,
            x.as_ptr() as *mut u16,
            toks.as_ptr() as *const u32,
            s as u32,
            h as u32,
        )?;
        self.prefill_prof.mark(stream, PTag::Gather)?;

        let qs = 1.0 / (d as f32).sqrt();
        for (li, w) in self.layers.iter().enumerate() {
            // attn norm（逐行 grid = S；与单行版同数学）
            pk.launch_rms_norm_rows(
                dev,
                stream,
                x.as_ptr() as *const u16,
                xn.as_ptr() as *mut u16,
                w.attn_norm.as_ptr() as *const u16,
                s as u32,
                h as u32,
                eps,
            )?;
            self.prefill_prof.mark(stream, PTag::RmsAttn)?;
            // QKV 投影：fused weight → 一次大 GEMM（m = S, n = nqk+2kvk；
            // 32F 计算 → 一次 cast f16；q/k/v 为同一缓冲的列偏移），否则
            // 分离 3 次 GEMM（数值同——每输出列相同 K 归约，仅 cublas 分块
            // 顺序可能不同；D7 GEMM 档 rel 1e-4 实测富余）。
            let (q_ptr, k_ptr, v_ptr): (*mut u16, *mut u16, *mut u16) = if fused_qkv {
                let wqkv = w.qkv_proj.as_ref().expect("fused weight present");
                let c_qkv = c_qkv.as_ref().expect("fused scratch present");
                self.gemm1r(&xn, wqkv, s, nqk + 2 * kvk, h, c_qkv)?;
                self.prefill_prof.mark(stream, PTag::GemmQkv)?;
                // Single fused cast: c_qkv [s x (nqk+2kvk)] f32 → q/k/v
                // contiguous f16, the exact layout the separated leg
                // produces (bit-identical element conversion), so every
                // downstream kernel is layout-agnostic.
                let q_buf = q.as_ref().expect("q present");
                let k_buf = k.as_ref().expect("k present");
                let v_buf = v.as_ref().expect("v present");
                pk.launch_cast_split_qkv(
                    dev,
                    stream,
                    c_qkv.as_ptr() as *const f32,
                    q_buf.as_ptr() as *mut u16,
                    k_buf.as_ptr() as *mut u16,
                    v_buf.as_ptr() as *mut u16,
                    s as u32,
                    nqk as u32,
                    kvk as u32,
                )?;
                self.prefill_prof.mark(stream, PTag::CastQkv)?;
                (
                    q_buf.as_ptr() as *mut u16,
                    k_buf.as_ptr() as *mut u16,
                    v_buf.as_ptr() as *mut u16,
                )
            } else {
                    let c_q = c_q.as_ref().expect("separated scratch present");
                    let c_k = c_k.as_ref().expect("separated scratch present");
                    let c_v = c_v.as_ref().expect("separated scratch present");
                    let q = q.as_ref().expect("separated q present");
                    let k = k.as_ref().expect("separated k present");
                    let v = v.as_ref().expect("separated v present");
                    self.gemm1r(&xn, &w.q_proj, s, nqk, h, c_q)?;
                    self.prefill_prof.mark(stream, PTag::GemmQkv)?;
                    self.diff.launch_cast_f32_f16(
                        dev,
                        stream,
                        c_q.as_ptr() as *const f32,
                        q.as_ptr() as *mut u16,
                        (s * nqk) as u32,
                    )?;
                    self.gemm1r(&xn, &w.k_proj, s, kvk, h, c_k)?;
                    self.prefill_prof.mark(stream, PTag::GemmQkv)?;
                    self.diff.launch_cast_f32_f16(
                        dev,
                        stream,
                        c_k.as_ptr() as *const f32,
                        k.as_ptr() as *mut u16,
                        (s * kvk) as u32,
                    )?;
                    self.gemm1r(&xn, &w.v_proj, s, kvk, h, c_v)?;
                    self.prefill_prof.mark(stream, PTag::GemmQkv)?;
                    self.diff.launch_cast_f32_f16(
                        dev,
                        stream,
                        c_v.as_ptr() as *const f32,
                        v.as_ptr() as *mut u16,
                        (s * kvk) as u32,
                    )?;
                    self.prefill_prof.mark(stream, PTag::CastQkv)?;
                    (
                        q.as_ptr() as *mut u16,
                        k.as_ptr() as *mut u16,
                        v.as_ptr() as *mut u16,
                    )
                };
            let qc = q_ptr as *const u16;
            let kc = k_ptr as *const u16;
            let vc = v_ptr as *const u16;
            // q/k head norm（Qwen3 系：RoPE 前；行 = s×heads，w 共享）
            if let (Some(qn), Some(kn)) = (&w.q_norm, &w.k_norm) {
                self.kernels.launch_rms_heads(
                    dev,
                    stream,
                    qc,
                    q_ptr,
                    qn.as_ptr() as *const u16,
                    (s * q_heads) as u32,
                    d as u32,
                    eps,
                )?;
                self.prefill_prof.mark(stream, PTag::RmsHeadsQ)?;
                self.kernels.launch_rms_heads(
                    dev,
                    stream,
                    kc,
                    k_ptr,
                    kn.as_ptr() as *const u16,
                    (s * kv_heads) as u32,
                    d as u32,
                    eps,
                )?;
                self.prefill_prof.mark(stream, PTag::RmsHeadsK)?;
            }
            // RoPE 批（行 = s×heads；行内 pos = s）
            pk.launch_rope_rows(
                dev,
                stream,
                q_ptr,
                (d / 2) as u32,
                q_heads as u32,
                s as u32,
                self.cfg.rope_theta,
            )?;
            self.prefill_prof.mark(stream, PTag::RopeQ)?;
            pk.launch_rope_rows(
                dev,
                stream,
                k_ptr,
                (d / 2) as u32,
                kv_heads as u32,
                s as u32,
                self.cfg.rope_theta,
            )?;
            self.prefill_prof.mark(stream, PTag::RopeK)?;
            // 注意力缩放 1/sqrt(d)（与逐 token 路径同一缩放点）
            self.kernels.launch_scale(dev, stream, q_ptr, (s * nqk) as u32, qs)?;
            self.prefill_prof.mark(stream, PTag::ScaleQ)?;
            // FMHA：整序列注意力（B=1；K/V 读本层新鲜值）
            fmha.launch_batched_prefill(
                dev,
                qc,
                kc,
                vc,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                s as u32,
                1,
                q_heads as u32,
                kv_heads as u32,
                d as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::Fmha)?;
            // KV 页写（页序与逐 token 路径逐位一致；page_base = li*pp
            // 与逐 token 显式 phys 的层偏移语义对齐）
            pk.launch_kv_write_seq(
                dev,
                stream,
                kc,
                vc,
                self.kv.data.as_ptr() as *mut u16,
                s as u32,
                BLOCK_LEN as u32,
                kv_heads as u32,
                d as u32,
                li as u32 * self.pp as u32,
                total_pages as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::KvWrite)?;
            // o 投影 → 残差
            self.gemm1r(&o, &w.o_proj, s, h, nqk, &c_o)?;
            self.prefill_prof.mark(stream, PTag::GemmO)?;
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                c_o.as_ptr() as *const f32,
                oadd.as_ptr() as *mut u16,
                (s * h) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::CastO)?;
            self.kernels.launch_add(
                dev,
                stream,
                x.as_ptr() as *mut u16,
                oadd.as_ptr() as *const u16,
                (s * h) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::AddO)?;
            // FFN
            pk.launch_rms_norm_rows(
                dev,
                stream,
                x.as_ptr() as *const u16,
                xn.as_ptr() as *mut u16,
                w.ffn_norm.as_ptr() as *const u16,
                s as u32,
                h as u32,
                eps,
            )?;
            self.prefill_prof.mark(stream, PTag::RmsFfn)?;
            self.gemm1r(&xn, &w.gate_proj, s, ffn, h, &c_g)?;
            self.prefill_prof.mark(stream, PTag::GemmGate)?;
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                c_g.as_ptr() as *const f32,
                gate.as_ptr() as *mut u16,
                (s * ffn) as u32,
            )?;
            self.gemm1r(&xn, &w.up_proj, s, ffn, h, &c_u)?;
            self.prefill_prof.mark(stream, PTag::GemmUp)?;
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                c_u.as_ptr() as *const f32,
                up.as_ptr() as *mut u16,
                (s * ffn) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::CastGU)?;
            self.kernels.launch_swiglu(
                dev,
                stream,
                gate.as_ptr() as *const u16,
                up.as_ptr() as *const u16,
                down.as_ptr() as *mut u16,
                (s * ffn) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::Swiglu)?;
            self.gemm1r(&down, &w.down_proj, s, h, ffn, &c_d)?;
            self.prefill_prof.mark(stream, PTag::GemmDown)?;
            self.diff.launch_cast_f32_f16(
                dev,
                stream,
                c_d.as_ptr() as *const f32,
                l2.as_ptr() as *mut u16,
                (s * h) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::CastD)?;
            self.kernels.launch_add(
                dev,
                stream,
                x.as_ptr() as *mut u16,
                l2.as_ptr() as *const u16,
                (s * h) as u32,
            )?;
            self.prefill_prof.mark(stream, PTag::AddD)?;
            self.prefill_prof.mark(stream, PTag::LayerBoundary)?;
        }

        // final norm → lm_head（m = S；回读末位 (S-1) 行 logits）
        pk.launch_rms_norm_rows(
            dev,
            stream,
            x.as_ptr() as *const u16,
            xn.as_ptr() as *mut u16,
            self.final_norm.as_ptr() as *const u16,
            s as u32,
            h as u32,
            eps,
        )?;
        self.prefill_prof.mark(stream, PTag::FinalRms)?;
        self.gemm1r(&xn, &self.lm_head, s, self.cfg.vocab_size, h, &logits)?;
        self.prefill_prof.mark(stream, PTag::LmHead)?;
        stream.synchronize()?;
        // S1-7: fold and print the profile (events completed — the sync
        // above closed the window). No-op when the profiler is off.
        self.prefill_prof.finalize();
        let bytes = self.cfg.vocab_size * 4;
        let hb = HostBuffer::alloc(bytes)?;
        let off = (s - 1) * self.cfg.vocab_size * 4;
        let rc = unsafe {
            cudarc::runtime::sys::cudaMemcpy(
                hb.as_ptr() as *mut c_void,
                (logits.as_ptr() as *const u8).add(off) as *const c_void,
                bytes,
                cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        rc.result().map_err(from_runtime_error)?;
        Ok(unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.cfg.vocab_size).to_vec()
        })
    }

    /// 生成：prefill（逐 token 步进写 KV）+ 采样 argmax-first（temp=0）。
    ///
    /// `eos` 命中即停；`max_tokens` 为提示增量 token 数硬限。
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: u32,
        eos_id: Option<u32>,
        temperature: f32,
    ) -> Result<Vec<u32>, EngineError> {
        if temperature != 0.0 {
            return Err(EngineError::Sts(
                "temperature > 0 sampling not wired (record: host sampler pipeline)".into(),
            ));
        }
        if prompt_ids.is_empty() {
            return Err(EngineError::Sts("empty prompt".into()));
        }
        // prefill：单次 forward（006 T1 FMHA 批；装载/launch 失败自动
        // 回退既有逐 token 步进路径——同输入结果逐位一致）
        self.prefill_batch(prompt_ids)?;
        let mut cur = *prompt_ids.last().unwrap();
        let mut pos = prompt_ids.len();
        let mut out = Vec::new();
        while out.len() < max_tokens as usize {
            let logits = self.step(cur, pos, pos + 1)?;
            if logits.iter().all(|l| l.is_nan()) {
                return Err(EngineError::NaNLogits);
            }
            let next = argmax_first(&logits);
            if Some(next) == eos_id {
                break;
            }
            out.push(next);
            cur = next;
            pos += 1;
        }
        Ok(out)
    }

    /// 行主序输出 GEMM（prefill 批路径专用）：`C = A·B` 直接写出行主序
    /// [m×n]（gemm_f32acc 的 OP_T/OP_T 约定输出列主序视图，仅 m == 1 时
    /// 与行主序重合——逐 token 解码路径因此从未暴露；批路径 m > 1 必须
    /// 用本变体）。推导：行主序 C = A·B ⇔ C^T = B^T·A^T；以 OP_N/OP_N
    /// 交换操作数调用（cublas A = B（ld n）、cublas B = A（ld k）、
    /// m' = n、n' = m、ldc = n），cublas 元素 (i,j) 落在 i + j·n ==
    /// 引擎行主序 (j,i) 地址——逐位一致，无需转置。
    fn gemm1r(
        &self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        c: &DeviceBuffer,
    ) -> Result<(), EngineError> {
        let amat = GpuMat {
            ptr: b.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: n as c_int,
        };
        let bmat = GpuMat {
            ptr: a.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: k as c_int,
        };
        let mut cmat = GpuMat {
            ptr: c.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_32F,
            ld: n as c_int,
        };
        self.gemm.gemm_exec(
            &self.stream,
            n as c_int,
            m as c_int,
            k as c_int,
            &amat,
            &bmat,
            &mut cmat,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            blas::cublasOperation_t::CUBLAS_OP_N,
            blas::cublasOperation_t::CUBLAS_OP_N,
            1.0,
            0.0,
        )?;
        Ok(())
    }

    /// S1-2: 单次稳定参数格 GEMM（`GemmPlan` 执行；替代旧 gemm1/gemm1_32f
    /// 的逐调参数构建——decode 步全部走预建 plans，见 `DecodeGemmPlans`）。
    ///
    /// JitGemm dispatch: m=1 f16-in plans (the decode step's production
    /// channel: q/k/v/o/gate/up/down per layer + lm_head) run the
    /// `gemv_m1_f16f32` kernel instead of cublas when `self.jgemm` is
    /// loaded (REINFER_JGEMM default on). Numeric tier is the same
    /// 32F-acc criterion (f16 in / f32 out, fp32 accumulation); the
    /// reduction order differs -> D7-level drift vs cublas, recorded.
    /// A jgemm launch failure falls back to cublas with the
    /// `jgemm_fallbacks` counter incremented. The parity-f32 tier
    /// (row_major_f32 plans) and any non-m=1 plan are never matched —
    /// they keep the cublas path bit-identical.
    #[inline]
    fn gemm_exec_plan(&self, stream: &CudaStream, plan: &GemmPlan) -> Result<(), EngineError> {
        if let Some(jg) = &self.jgemm
            && jg.matches(plan)
        {
            match jg.launch(stream, plan) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    self.jgemm_fallbacks.fetch_add(1, Ordering::Relaxed);
                    eprintln!("reinfer-cuda: jgemm launch failed — cublas fallback: {e}");
                }
            }
        }
        self.gemm.execute(stream, plan)?;
        Ok(())
    }

    /// Whether the JitGemm kernel is loaded for this engine (diagnostics).
    pub fn jgemm_enabled(&self) -> bool {
        self.jgemm.is_some()
    }

    /// JitGemm launch failures that fell back to cublas (diagnostics/tests).
    pub fn jgemm_fallbacks(&self) -> u64 {
        self.jgemm_fallbacks.load(Ordering::Relaxed)
    }

    /// 设备 f16 缓冲回读（f32 向量；debug trace 用）。
    fn readback_f16(&self, buf: &DeviceBuffer) -> Option<Vec<f32>> {
        let n = buf.size() / 2;
        let hb = HostBuffer::alloc(buf.size()).ok()?;
        copy(&mut MemRef::Host(&hb), &MemRef::Device(buf), buf.size(), None).ok()?;
        self.stream.synchronize().ok()?;
        let row: Vec<f32> = unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const u16, n)
                .iter()
                .map(|v| f16_bits_to_f32(*v))
                .collect()
        };
        Some(row)
    }

    /// 006 T3E: 长度 H2D（预分配 pinned + 流内异步；S1-2 起每步一次——
    /// 步内 kv_len 恒定，旧实现每层重复上传 28 次）。
    fn upload_lens(&self, stream: &CudaStream, kv_len: u32) -> Result<(), EngineError> {
        unsafe {
            std::ptr::copy_nonoverlapping(
                &kv_len as *const u32 as *const u8,
                self.lens_hb.as_ptr() as *mut u8,
                4,
            )
        }
        copy(&mut MemRef::Device(&self.lens_dev), &MemRef::Host(&self.lens_hb), 4, Some(stream))?;
        Ok(())
    }
}

/// 生成语义（014 T9）：argmax（tie-break 首个最大——012 语义）。
pub fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, l) in logits.iter().enumerate().skip(1) {
        if l > &logits[best] {
            best = i;
        }
    }
    best as u32
}

/// 张量 → f16 行主序字节（[out,in] → 转置 [in,out]：gemm B 的 [k×n] 约定）。
fn to_f16_rm(
    view: &reinfer_safetensors::TensorView<'_>,
    out: usize,
    inp: usize,
) -> Result<Vec<u8>, EngineError> {
    let mut w16 = vec![0u8; out * inp * 2];
    let mut tmp = vec![0u8; inp * 2];
    let sz = elemsize(view.dtype.clone());
    for r in 0..out {
        let row = &view.bytes[r * inp * sz..(r + 1) * inp * sz];
        row_to_f16(view.dtype.clone(), row, &mut tmp)?;
        for (c, ch) in tmp.chunks_exact(2).enumerate() {
            w16[c * out * 2 + r * 2] = ch[0];
            w16[c * out * 2 + r * 2 + 1] = ch[1];
        }
    }
    Ok(w16)
}

/// 张量 → f16 行主序字节（2-d；**无转置**——行序 = 原 [out,in] 顺序）。
fn to_f16_rows(view: &reinfer_safetensors::TensorView<'_>) -> Result<Vec<u8>, EngineError> {
    let rows = view.shape[0] as usize;
    let cols = view.shape[1] as usize;
    let sz = elemsize(view.dtype.clone());
    let mut w16 = vec![0u8; rows * cols * 2];
    for r in 0..rows {
        let row = &view.bytes[r * cols * sz..(r + 1) * cols * sz];
        row_to_f16(view.dtype.clone(), row, &mut w16[r * cols * 2..(r + 1) * cols * 2])?;
    }
    Ok(w16)
}

/// 张量 → f16 字节（1-d 向量；无转置）。
fn to_f16_vec(view: &reinfer_safetensors::TensorView<'_>) -> Result<Vec<u8>, EngineError> {
    let n = view.shape[0] as usize;
    let mut out = vec![0u8; n * 2];
    row_to_f16(view.dtype.clone(), view.bytes, &mut out)?;
    Ok(out)
}

fn elemsize(d: StDtype) -> usize {
    match d {
        StDtype::F32 => 4,
        StDtype::F16 | StDtype::Bf16 => 2,
        StDtype::Other(_) => 2,
    }
}

/// 行字节（→ f16 位）。
fn row_to_f16(d: StDtype, bytes: &[u8], out: &mut [u8]) -> Result<(), EngineError> {
    match d {
        StDtype::F16 => {
            out.copy_from_slice(bytes);
            Ok(())
        }
        StDtype::Bf16 => {
            for (i, ch) in bytes.chunks_exact(2).enumerate() {
                let b = u16::from_le_bytes([ch[0], ch[1]]);
                let h = f32_to_f16_bits(bf16_to_f32(b));
                out[i * 2] = h as u8;
                out[i * 2 + 1] = (h >> 8) as u8;
            }
            Ok(())
        }
        StDtype::F32 => {
            for (i, ch) in bytes.chunks_exact(4).enumerate() {
                let f = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                let h = f32_to_f16_bits(f);
                out[i * 2] = h as u8;
                out[i * 2 + 1] = (h >> 8) as u8;
            }
            Ok(())
        }
        StDtype::Other(o) => Err(EngineError::UnsupportedDtype(o.to_string())),
    }
}

/// f16 位 → f32（与内核 hbits_to_f32 同语义）。
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h >> 15)) << 31;
    let exp = (u32::from(h >> 10)) & 0x1f;
    let man = u32::from(h) & 0x3ff;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let mut m = man;
        let mut s = 0u32;
        while m & 0x400 == 0 {
            m <<= 1;
            s += 1;
        }
        return f32::from_bits(sign | ((113 - s) << 23) | ((m & 0x3ff) << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

/// 014 S0-3b: expand a byte buffer of f16 values into an f32 byte buffer —
/// bit-identical values (criterion-tier weight loading; cublas
/// CUBLAS_COMPUTE_32F rejects mixed f16-B / f32-A inputs, and the llama.cpp
/// CPU referee computes with f32 activations against f16-valued weights).
fn expand_f16_to_f32(w16: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w16.len() * 2);
    for chunk in w16.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&f16_bits_to_f32(h).to_le_bytes());
    }
    out
}

/// bf16 u16 → f32。
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 → f16 位（RNE，与内核 f32_to_hbits 同语义——规则位构造）。
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000u32;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return (sign | 0x7c00 | ((man >> 13) & 0x3ff)) as u16;
    }
    let half_exp = exp - 127 + 15;
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign as u16;
        }
        return (sign | ((man | 0x80_0000) >> (1 - half_exp + 13))) as u16;
    }
    if half_exp >= 31 {
        return (sign | 0x7c00) as u16;
    }
    (sign | ((half_exp as u32) << 10) | (man >> 13)) as u16
}

/// host → device（同步拷贝；简单窗口按整块）。
fn upl(dev: DeviceId, host: &[u8]) -> Result<DeviceBuffer, LaunchError> {
    let hb = HostBuffer::alloc(host.len())?;
    unsafe {
        std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
    }
    // （host 指针已为 u8——见上行；无额外转 type）
    let db = DeviceBuffer::alloc(dev, host.len())?;
    copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None)?;
    Ok(db)
}

// ---------------------------------------------------------------------------
// S1-7 prefill profiler (REINFER_PREFILL_PROFILE, default off).
//
// cudaEvent marks at every prefill launch site (the batched path, one
// forward pass over the whole prompt). Each `mark` records a timestamped
// event on the engine stream; `finalize` (called after the prefill's final
// stream sync) folds the consecutive-mark deltas per layer and per kernel
// tag, printing a per-layer and per-tag GPU attribution table plus the
// host-side wall time — the S1-7 measurement surface for QKV fusion,
// FMHA heuristic choice and small-kernel batching decisions.
//
// Environment-gated like DecodeProfiler: when off, marks are no-ops and no
// events are created (zero probe overhead). Event budget bounded; overflow
// deactivates the profiler rather than growing.
// ---------------------------------------------------------------------------

/// REINFER_PREFILL_PROFILE parsing: unset -> **off** (default).
#[must_use]
fn prefill_profile_enabled_from_env(value: Option<&str>) -> bool {
    match value {
        None => false,
        Some(v) => {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
        }
    }
}

/// Prefill profiler env var name (public for tests).
pub const PREFILL_PROFILE_ENV: &str = "REINFER_PREFILL_PROFILE";

/// Kernel tag (one per launch site of the batched prefill loop; the tag is
/// the print label for the cudaEvent delta that ends at that mark).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum PTag {
    Gather = 0,
    RmsAttn,
    GemmQkv,
    CastQkv,
    RmsHeadsQ,
    RmsHeadsK,
    RopeQ,
    RopeK,
    ScaleQ,
    Fmha,
    KvWrite,
    GemmO,
    CastO,
    AddO,
    RmsFfn,
    GemmGate,
    GemmUp,
    CastGU,
    Swiglu,
    GemmDown,
    CastD,
    AddD,
    FinalRms,
    LmHead,
    LayerBoundary,
}

impl PTag {
    fn name(self) -> &'static str {
        match self {
            PTag::Gather => "gather",
            PTag::RmsAttn => "rms_attn",
            PTag::GemmQkv => "gemm_qkv",
            PTag::CastQkv => "cast_qkv",
            PTag::RmsHeadsQ => "rms_heads_q",
            PTag::RmsHeadsK => "rms_heads_k",
            PTag::RopeQ => "rope_q",
            PTag::RopeK => "rope_k",
            PTag::ScaleQ => "scale_q",
            PTag::Fmha => "fmha",
            PTag::KvWrite => "kv_write",
            PTag::GemmO => "gemm_o",
            PTag::CastO => "cast_o",
            PTag::AddO => "add_o",
            PTag::RmsFfn => "rms_ffn",
            PTag::GemmGate => "gemm_gate",
            PTag::GemmUp => "gemm_up",
            PTag::CastGU => "cast_gate_up",
            PTag::Swiglu => "swiglu",
            PTag::GemmDown => "gemm_down",
            PTag::CastD => "cast_down",
            PTag::AddD => "add_d",
            PTag::FinalRms => "final_rms",
            PTag::LmHead => "lm_head",
            PTag::LayerBoundary => "layer_boundary",
        }
    }
}

/// S1-7 prefill profiler (owned by Engine; inert unless env-enabled).
#[derive(Debug)]
pub struct PrefillProfiler {
    active: bool,
    dev: DeviceId,
    /// Recorded marks (event + tag), stream-ordered.
    events: Vec<CudaEvent>,
    tags: Vec<PTag>,
    /// Layers seen so far (counted by LayerBoundary marks).
    layers: u32,
    /// Prompt length of the window being profiled.
    seq: u32,
    /// Event budget (bounded — deactivate on overflow).
    max_events: usize,
    /// Host-side wall clock for the whole prefill.
    t0: Option<std::time::Instant>,
}

/// One folded per-tag aggregate over the prefill's layers.
#[derive(Debug, Clone, Copy)]
pub struct PrefillTagMean {
    /// Kernel tag (PTag).
    pub tag: &'static str,
    /// Mean GPU time per layer attributed to this tag (ms).
    pub ms: f32,
    /// Launch count (= number of marks with this tag).
    pub launches: u32,
}

/// Aggregated prefill profile after `finalize`.
#[derive(Debug, Clone, Default)]
pub struct PrefillProfile {
    /// Prompt length in tokens.
    pub seq: u32,
    /// Number of transformer layers folded.
    pub layers: u32,
    /// Per-tag per-layer means (sorted by ms descending).
    pub tags: Vec<PrefillTagMean>,
    /// Host-side wall time (ms) — launch/sync overhead view.
    pub wall_ms: f32,
    /// Total GPU busy time (ms) = last mark - first mark.
    pub gpu_ms: f32,
    /// Total launch marks.
    pub launches: u32,
}

impl PrefillProfiler {
    /// Create an inert profiler (env off) or an armed one (env on).
    pub fn new(dev: DeviceId) -> Self {
        let active =
            prefill_profile_enabled_from_env(std::env::var(PREFILL_PROFILE_ENV).ok().as_deref());
        Self {
            active,
            dev,
            events: Vec::new(),
            tags: Vec::new(),
            layers: 0,
            seq: 0,
            max_events: 4096,
            t0: None,
        }
    }

    /// Whether the profiler is armed (env-on and not deactivated).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a prefill profile window (host wall clock + first mark).
    pub fn begin(&mut self, stream: &CudaStream, seq: u32) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        self.t0 = Some(std::time::Instant::now());
        self.events.clear();
        self.tags.clear();
        self.layers = 0;
        self.seq = seq;
        self.mark(stream, PTag::Gather)?;
        Ok(())
    }

    /// Record a mark: an event on the stream + its tag. No-op when off.
    fn mark(&mut self, stream: &CudaStream, tag: PTag) -> Result<(), LaunchError> {
        if !self.active {
            return Ok(());
        }
        if self.events.len() >= self.max_events {
            eprintln!(
                "[reinfer-cuda] REINFER_PREFILL_PROFILE: event budget exceeded — \
                 profile deactivated"
            );
            self.active = false;
            self.events.clear();
            self.tags.clear();
            return Ok(());
        }
        let ev = CudaEvent::new(self.dev)?;
        ev.record(stream)?;
        self.events.push(ev);
        self.tags.push(tag);
        if tag == PTag::LayerBoundary {
            self.layers += 1;
        }
        Ok(())
    }

    /// Finalize: compute consecutive-mark deltas, fold per-tag per-layer
    /// means and print the table. The stream must be synchronized (the
    /// prefill's logits readback already is). No-op when off.
    pub fn finalize(&mut self) -> PrefillProfile {
        let empty = PrefillProfile::default();
        if !self.active || self.events.len() < 2 {
            return empty;
        }
        let seq = self.seq;
        // Deltas: mark i -> i+1 attributed to tag i+1 (the kernel that ends
        // at that mark). The first delta (gather mark -> rms_attn of layer 0)
        // is the gather kernel itself.
        let n = self.events.len();
        let mut ms_of: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n - 1 {
            let d = self.events[i].elapsed_ms(&self.events[i + 1]).unwrap_or(0.0);
            ms_of.push(d);
        }
        let gpu_ms: f32 = ms_of.iter().sum();
        let launches = n as u32;
        let layers = self.layers.max(1);
        // Per-tag aggregate (sum over layers; layer-boundary deltas are the
        // rms_attn->layer tail, attributed to the next layer's first kernel —
        // excluded from tag folding here).
        let mut sums: std::collections::HashMap<u16, (f32, u32)> = std::collections::HashMap::new();
        for (i, tag) in self.tags.iter().enumerate().skip(1) {
            if *tag == PTag::LayerBoundary {
                continue;
            }
            let e = sums.entry(*tag as u16).or_insert((0.0, 0));
            e.0 += ms_of[i - 1];
            e.1 += 1;
        }
        let mut tags: Vec<PrefillTagMean> = sums
            .iter()
            .map(|(k, (ms, cnt))| PrefillTagMean {
                tag: PTag::from_u16(*k).name(),
                ms: ms / layers as f32,
                launches: *cnt,
            })
            .collect();
        tags.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));
        let wall_ms = self.t0.map(|t| t.elapsed().as_secs_f32() * 1e3).unwrap_or(0.0);
        let prof = PrefillProfile { seq, layers, tags, wall_ms, gpu_ms, launches };
        println!(
            "[reinfer-cuda] REINFER_PREFILL_PROFILE: seq={seq} layers={layers} \
             wall={wall_ms:.1}ms gpu={gpu_ms:.1}ms launches={launches}"
        );
        for t in &prof.tags {
            println!(
                "[reinfer-cuda] REINFER_PREFILL_PROFILE:   {:<14} {:.3} ms/layer x{}",
                t.tag, t.ms, t.launches / layers
            );
        }
        prof
    }
}

impl PTag {
    fn from_u16(v: u16) -> Self {
        if v < 25 {
            // PTag is a dense u16 enum 0..=24; construct via the discriminant
            // of a placeholder array to keep the mapping total.
            const ALL: [PTag; 25] = [
                PTag::Gather,
                PTag::RmsAttn,
                PTag::GemmQkv,
                PTag::CastQkv,
                PTag::RmsHeadsQ,
                PTag::RmsHeadsK,
                PTag::RopeQ,
                PTag::RopeK,
                PTag::ScaleQ,
                PTag::Fmha,
                PTag::KvWrite,
                PTag::GemmO,
                PTag::CastO,
                PTag::AddO,
                PTag::RmsFfn,
                PTag::GemmGate,
                PTag::GemmUp,
                PTag::CastGU,
                PTag::Swiglu,
                PTag::GemmDown,
                PTag::CastD,
                PTag::AddD,
                PTag::FinalRms,
                PTag::LmHead,
                PTag::LayerBoundary,
            ];
            ALL[v as usize]
        } else {
            PTag::LayerBoundary
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GraphStepDecl, PtrUpdate, SpecAcc, expected_node_count, graph_enabled_from_env,
        parity_f32_enabled_from_env,
    };
    use std::ffi::c_void;

    #[test]
    fn parity_f32_env_default_off() {
        // 014 S0-3b: REINFER_PARITY_F32 — unset defaults to off (the f16
        // storage layer stays the product path)
        assert!(!parity_f32_enabled_from_env(None));
        // explicit on values
        assert!(parity_f32_enabled_from_env(Some("1")));
        assert!(parity_f32_enabled_from_env(Some("on")));
        assert!(parity_f32_enabled_from_env(Some("ON")));
        assert!(parity_f32_enabled_from_env(Some("true")));
        assert!(parity_f32_enabled_from_env(Some("yes")));
        assert!(parity_f32_enabled_from_env(Some(" 1 ")));
        // off / junk values
        assert!(!parity_f32_enabled_from_env(Some("0")));
        assert!(!parity_f32_enabled_from_env(Some("off")));
        assert!(!parity_f32_enabled_from_env(Some("false")));
        assert!(!parity_f32_enabled_from_env(Some("no")));
        assert!(!parity_f32_enabled_from_env(Some("")));
        assert!(!parity_f32_enabled_from_env(Some("garbage")));
    }

    #[test]
    fn graph_env_default_off() {
        // 未设置 → 默认 off（fail-closed：cublas 节点重放位级证明前不默认开启）
        assert!(!graph_enabled_from_env(None));
        // 显式开启值
        assert!(graph_enabled_from_env(Some("1")));
        assert!(graph_enabled_from_env(Some("on")));
        assert!(graph_enabled_from_env(Some("ON")));
        assert!(graph_enabled_from_env(Some("yes")));
        assert!(graph_enabled_from_env(Some(" true ")));
        // 关闭值（含空串）
        assert!(!graph_enabled_from_env(Some("0")));
        assert!(!graph_enabled_from_env(Some("off")));
        assert!(!graph_enabled_from_env(Some("OFF")));
        assert!(!graph_enabled_from_env(Some("false")));
        assert!(!graph_enabled_from_env(Some("no")));
        assert!(!graph_enabled_from_env(Some("")));
        assert!(!graph_enabled_from_env(Some("  ")));
    }

    #[test]
    fn expected_node_count_formula() {
        // cublas path (JitGemm off): 20n+3 / 18n+3 — gather + per-layer
        // (13/11 small kernels + 7 gemms) + final rms_norm + lm_head gemm.
        assert_eq!(expected_node_count(28, true, false, false), 563);
        assert_eq!(expected_node_count(28, false, false, false), 507);
        assert_eq!(expected_node_count(1, true, false, false), 23);
        assert_eq!(expected_node_count(4, true, false, false), 83);
        assert_eq!(expected_node_count(4, false, false, false), 75);
        // Graph V2 (JitGemm loaded): every m=1 decode GEMM doubles into
        // the jgemm phase pair — 27n+4 / 25n+4.
        assert_eq!(expected_node_count(28, true, true, false), 760);
        assert_eq!(expected_node_count(28, false, true, false), 704);
        assert_eq!(expected_node_count(1, true, true, false), 31);
        assert_eq!(expected_node_count(4, true, true, false), 112);
        assert_eq!(expected_node_count(4, false, true, false), 104);
        // S1-9 fused decode: 4 + 8n — the bookends (gather, rms_attn(0),
        // lm phase pair) plus the 8 fused nodes per layer (p1_qkv, p2_qkv,
        // kv_write, flash, p2_o, p2_gu, p1_ogud, p2_down), independent of
        // head_norm/jgemm (the fused kernels cover both variants).
        assert_eq!(expected_node_count(28, true, true, true), 228);
        assert_eq!(expected_node_count(28, false, true, true), 228);
        assert_eq!(expected_node_count(28, true, false, true), 228);
        assert_eq!(expected_node_count(1, true, true, true), 12);
        assert_eq!(expected_node_count(4, true, true, true), 36);
        assert_eq!(expected_node_count(4, false, true, true), 36);
    }

    /// Fabricated declaration for the cell-index bookkeeping test (mirrors
    /// the real build's layout: gather node 0, then per-layer rope q/k and
    /// kv_write with the same slot positions).
    fn fake_decl(n_layers: usize) -> GraphStepDecl {
        let mut acc = SpecAcc::default();
        let g1 = cudarc::runtime::sys::dim3 { x: 1, y: 1, z: 1 };
        let b256 = cudarc::runtime::sys::dim3 { x: 256, y: 1, z: 1 };
        // gather (4 slots)
        acc.custom(std::ptr::null_mut(), 4, g1, b256, &[1, 2, 0, 4]);
        let cell_token = acc.cell_of(2);
        let mut refresh = vec![(acc.specs.len() - 1, 2)];
        let mut rope_q = Vec::new();
        let mut rope_k = Vec::new();
        let mut phys = Vec::new();
        let mut off = Vec::new();
        for _ in 0..n_layers {
            // rope q/k (6 slots, pos at 3)
            acc.custom(std::ptr::null_mut(), 6, g1, b256, &[0; 6]);
            refresh.push((acc.specs.len() - 1, 3));
            rope_q.push(acc.cell_of(3));
            acc.custom(std::ptr::null_mut(), 6, g1, b256, &[0; 6]);
            refresh.push((acc.specs.len() - 1, 3));
            rope_k.push(acc.cell_of(3));
            // kv_write (9 slots, phys/off at 3/4)
            acc.custom(std::ptr::null_mut(), 9, g1, b256, &[0; 9]);
            refresh.push((acc.specs.len() - 1, 3));
            refresh.push((acc.specs.len() - 1, 4));
            phys.push(acc.cell_of(3));
            off.push(acc.cell_of(4));
        }
        let mut updates = Vec::new();
        for (node, spec) in acc.specs.iter().enumerate() {
            let base = acc.cell_off[node];
            for (slot, _) in &spec.ptr_slots {
                let cell = (&mut acc.cells[base + *slot] as *mut u64) as *mut c_void;
                updates.push(PtrUpdate { node, slot: *slot, ptr: cell });
            }
        }
        // Fake handles: launch_fns stays null (the bookkeeping tests never
        // launch through the declaration).
        let launch_fns = vec![std::ptr::null_mut(); acc.specs.len()];
        let arg_slots: Vec<*mut c_void> = acc
            .cells
            .iter()
            .map(|c| (c as *const u64) as *mut c_void)
            .collect();
        GraphStepDecl {
            specs: acc.specs,
            launch_fns,
            cell_off: acc.cell_off,
            cells: acc.cells,
            arg_slots,
            updates,
            cell_token,
            cell_rope_q_pos: rope_q,
            cell_rope_k_pos: rope_k,
            cell_kv_phys: phys,
            cell_kv_off: off,
            refresh,
        }
    }

    #[test]
    fn write_step_cell_indices() {
        let mut decl = fake_decl(3);
        let n = decl.specs.len();
        assert_eq!(n, 3 * 3 + 1); // gather + 3 layers × 3 nodes
        let snapshot = decl.cells.clone();
        decl.write_step(42, 33, 3, 8); // lp = 1, off = 1
        // token lands in the gather node's row cell
        assert_eq!(decl.cells[decl.cell_token], 42);
        // pos lands in every rope q/k pos cell
        for li in 0..3 {
            assert_eq!(decl.cells[decl.cell_rope_q_pos[li]], 33);
            assert_eq!(decl.cells[decl.cell_rope_k_pos[li]], 33);
            // phys = li*pp + lp, off = pos % BLOCK_LEN
            assert_eq!(decl.cells[decl.cell_kv_phys[li]], (li * 8 + 1) as u64);
            assert_eq!(decl.cells[decl.cell_kv_off[li]], 1);
        }
        // every other cell untouched
        for (i, (before, after)) in snapshot.iter().zip(decl.cells.iter()).enumerate() {
            if i != decl.cell_token
                && !decl.cell_rope_q_pos.contains(&i)
                && !decl.cell_rope_k_pos.contains(&i)
                && !decl.cell_kv_phys.contains(&i)
                && !decl.cell_kv_off.contains(&i)
            {
                assert_eq!(before, after, "cell {i} must not change");
            }
        }
    }
}
