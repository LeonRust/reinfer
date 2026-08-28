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
use crate::diff::DiffKernels;
use crate::decode::DecodeKernels;
use crate::gemm::{Gemm, GpuMat};
use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;
use cudarc::cublas::sys as blas;
use reinfer_arch::llama::{from_hf_config, LlamaConfig};
use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use reinfer_safetensors::{SafeFile, StDtype};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

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
    add: KernelFn,
    swiglu: KernelFn,
    kv_write: KernelFn,
    gather: KernelFn,
    scale: KernelFn,
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
        let add = lib.kernel("add_f16_to_f16")?;
        let swiglu = lib.kernel("swiglu_f16")?;
        let kv_write = lib.kernel("kv_write_row")?;
        let gather = lib.kernel("gather_row")?;
        let scale = lib.kernel("scale_f16")?;
        Ok(Self { lib, rms_row, rms_heads, rope, add, swiglu, kv_write, gather, scale })
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
        unsafe { crate::jit::launch_rows(self.add, stream, dev, n.div_ceil(256), 256, args.as_mut_ptr()) }
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
        unsafe { crate::jit::launch_rows(self.swiglu, stream, dev, n.div_ceil(256), 256, args.as_mut_ptr()) }
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
        let nv: [i32; 5] = [
            phys as i32,
            off as i32,
            block_len as i32,
            kv_heads as i32,
            d as i32,
        ];
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
        unsafe { crate::jit::launch_rows(self.kv_write, stream, dev, per_tok.div_ceil(256) as u32, 256, args.as_mut_ptr()) }
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
        unsafe { crate::jit::launch_rows(self.scale, stream, dev, n.div_ceil(256), 256, args.as_mut_ptr()) }
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
        unsafe { crate::jit::launch_rows(self.gather, stream, dev, n.div_ceil(256), 256, args.as_mut_ptr()) }
    }
}

/// 层的设备权重（f16 位；矩阵已转置为 [k×n] 行主序——gemm B 约定）。
#[derive(Debug)]
pub struct LayerWeights {
    attn_norm: DeviceBuffer,
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
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
    kernels: DenseKernels,
    diff: DiffKernels,
    decode: DecodeKernels,
    stream: CudaStream,
    embed: DeviceBuffer,
    lm_head: DeviceBuffer,
    final_norm: DeviceBuffer,
    layers: Vec<LayerWeights>,
    kv: crate::decode::KvStore,
    pp: usize,
    pages_host: Vec<u32>,
    pages_dev: DeviceBuffer,
    lens_dev: DeviceBuffer,
    // f16 中间缓冲
    x: DeviceBuffer,
    xn: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn: DeviceBuffer,
    oadd: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
    l2: DeviceBuffer,
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
}

const BLOCK_LEN: usize = 32;

impl Engine {
    /// 装载：读 config.json + model.safetensors（模型目录；权重 f16 化上传）。
    pub fn load(
        dev: DeviceId,
        arch: &str,
        cache_dir: Option<PathBuf>,
        model_dir: &Path,
        max_kv: usize,
    ) -> Result<Self, EngineError> {
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

        let t = |name: &str, out_rows: usize, in_cols: usize| -> Result<DeviceBuffer, EngineError> {
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
            let w16 = to_f16_rm(&view, out_rows, in_cols)?;
            upl(dev, &w16).map_err(EngineError::Launch)
        };
        let tv = |name: &str, d: usize| -> Result<DeviceBuffer, EngineError> {
            let view = safe.tensor(name).map_err(|e| EngineError::Sts(e.to_string()))?;
            if view.shape.len() != 1 || view.shape[0] != d as u64 {
                return Err(EngineError::WeightShape(format!(
                    "{name}: expected [{d}] got {:?}",
                    view.shape
                )));
            }
            let w16 = to_f16_vec(&view)?;
            upl(dev, &w16).map_err(EngineError::Launch)
        };

        // embed 行主序取（gather 行长=token——**不做** gemm 转置；lm_head 走 gemm B 转置）。
        let embed = {
            let view = safe
                .tensor("model.embed_tokens.weight")
                .map_err(|e| EngineError::Sts(e.to_string()))?;
            if view.shape.len() != 2
                || view.shape[0] != vocab as u64
                || view.shape[1] != h as u64
            {
                return Err(EngineError::WeightShape(format!(
                    "model.embed_tokens.weight: expected [{vocab},{h}] got {:?}",
                    view.shape
                )));
            }
            let w16 = to_f16_rows(&view)?;
            upl(dev, &w16).map_err(EngineError::Launch)?
        };
        let lm_head = t("lm_head.weight", vocab, h)?;
        let final_norm = tv("model.norm.weight", h)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            layers.push(LayerWeights {
                attn_norm: tv(&p("input_layernorm.weight"), h)?,
                q_proj: t(&p("self_attn.q_proj.weight"), nqk, h)?,
                k_proj: t(&p("self_attn.k_proj.weight"), kvk, h)?,
                v_proj: t(&p("self_attn.v_proj.weight"), kvk, h)?,
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
        let kernels = DenseKernels::new(arch, cache_dir)?;
        let diff = DiffKernels::new(arch, None, stream.clone())?;
        let decode = DecodeKernels::new(arch, None, stream.clone())?;
        let gemm = Gemm::new(dev.index())?;

        let pp = max_kv.div_ceil(BLOCK_LEN);
        let kv = crate::decode::KvStore::alloc(dev, cfg.n_layer * pp, BLOCK_LEN, cfg.kv_heads, cfg.head_dim)?;

        let a16 = |n: usize| DeviceBuffer::alloc(dev, n * 2);
        let c32 = |n: usize| DeviceBuffer::alloc(dev, n * 4);
        Ok(Self {
            dev: dev.index(),
            cfg,
            gemm,
            kernels,
            diff,
            decode,
            stream,
            embed,
            lm_head,
            final_norm,
            layers,
            kv,
            pp,
            pages_host: vec![0u32; pp],
            pages_dev: DeviceBuffer::alloc(dev, pp * 4)?,
            lens_dev: DeviceBuffer::alloc(dev, 4)?,
            x: a16(h)?,
            xn: a16(h)?,
            q: a16(nqk)?,
            k: a16(kvk)?,
            v: a16(kvk)?,
            attn: a16(nqk)?,
            oadd: a16(h)?,
            gate: a16(ffn)?,
            up: a16(ffn)?,
            down: a16(ffn)?,
            l2: a16(h)?,
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
        })
    }

    /// 配置。
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// 设备索引。
    pub fn device(&self) -> u32 {
        self.dev
    }

    /// 单 token 步进：写 KV（pos）并返回 logits（f32 行主序 [vocab]）。
    pub fn step(
        &mut self,
        token: u32,
        pos: usize,
        kv_len: usize,
    ) -> Result<Vec<f32>, EngineError> {
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
        let dev = self.dev;
        let stream = &self.stream;
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

        // embed 行 → x
        self.kernels.launch_gather(
            dev,
            stream,
            self.embed.as_ptr() as *const u16,
            self.x.as_ptr() as *mut u16,
            token,
            h as u32,
        )?;

        let lp = pos / BLOCK_LEN;
        let off = pos % BLOCK_LEN;

        for (li, w) in self.layers.iter().enumerate() {
            // attn norm
            self.kernels.launch_rms_norm(
                dev, stream,
                self.x.as_ptr() as *const u16,
                self.xn.as_ptr() as *mut u16,
                w.attn_norm.as_ptr() as *const u16,
                h as u32, eps,
            )?;
            let x0_snap = if detail.is_some() { self.readback_f16(&self.x) } else { None };
            let xn_snap = if detail.is_some() { self.readback_f16(&self.xn) } else { None };
            // q/k/v 投影（32F compute → cast f16）
            self.gemm1(&self.xn, &w.q_proj, 1, nqk, h, &self.c_q)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_q.as_ptr() as *const f32, self.q.as_ptr() as *mut u16, nqk as u32)?;
            self.gemm1(&self.xn, &w.k_proj, 1, kvk, h, &self.c_k)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_k.as_ptr() as *const f32, self.k.as_ptr() as *mut u16, kvk as u32)?;
            self.gemm1(&self.xn, &w.v_proj, 1, kvk, h, &self.c_v)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_v.as_ptr() as *const f32, self.v.as_ptr() as *mut u16, kvk as u32)?;

            // q/k head norm（Qwen3 系：RoPE 前）
            if let (Some(qn), Some(kn)) = (&w.q_norm, &w.k_norm) {
                self.kernels.launch_rms_heads(
                    dev, stream,
                    self.q.as_ptr() as *const u16,
                    self.q.as_ptr() as *mut u16,
                    qn.as_ptr() as *const u16,
                    q_heads as u32, d as u32, eps,
                )?;
                self.kernels.launch_rms_heads(
                    dev, stream,
                    self.k.as_ptr() as *const u16,
                    self.k.as_ptr() as *mut u16,
                    kn.as_ptr() as *const u16,
                    kv_heads as u32, d as u32, eps,
                )?;
            }
            // RoPE（半分割 NEOX；q 全头、k kv 头）
            for hh in 0..q_heads {
                self.kernels.launch_rope_row(
                    dev, stream,
                    unsafe { (self.q.as_ptr() as *mut u16).add(hh * d) },
                    (d / 2) as u32, pos as u32, self.cfg.rope_theta,
                )?;
            }
            for hh in 0..kv_heads {
                self.kernels.launch_rope_row(
                    dev, stream,
                    unsafe { (self.k.as_ptr() as *mut u16).add(hh * d) },
                    (d / 2) as u32, pos as u32, self.cfg.rope_theta,
                )?;
            }
            let q_snap = if detail.is_some() { self.readback_f16(&self.q) } else { None };
            // 注意力 score 缩放：1/sqrt(head_dim)（transformers
            // scaled_dot_product_attention 语义；T7/T8 内核不含——engine 缩放点）
            let qs = 1.0 / (d as f32).sqrt();
            self.kernels.launch_scale(dev, stream, self.q.as_ptr() as *mut u16, nqk as u32, qs)?;
            // KV 写（物理页 = li*pp + lp；K/V 连续两区）
            let phys = (li * self.pp + lp) as u32;
            self.kernels.launch_kv_write(
                dev, stream,
                self.k.as_ptr() as *const u16,
                self.v.as_ptr() as *const u16,
                self.kv.data.as_ptr() as *mut u16,
                phys, off as u32, BLOCK_LEN as u32, kv_heads as u32, d as u32, total_pages as u32,
            )?;

            // decode（本层页表：物理 = li*pp + j，j < log_pages）
            let log_pages = kv_len.div_ceil(BLOCK_LEN);
            for j in 0..log_pages {
                self.pages_host[j] = (li * self.pp + j) as u32;
            }
            self.upload_pages(log_pages)?;
            self.upload_lens(kv_len as u32)?;
            stream.synchronize()?; // h2d 后核发
            self.decode.launch_decode_step_gqa(
                dev,
                self.q.as_ptr() as *const u16,
                self.pages_dev.as_ptr() as *const u32,
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

            let attn_snap = if detail.is_some() { self.readback_f16(&self.attn) } else { None };
                        if let (Some(det), Some(x0), Some(xn), Some(qq), Some(aa)) =
                (detail.as_mut(), x0_snap, xn_snap, q_snap, attn_snap)
            {
                det.push((x0, xn, qq, aa));
            }
// o 投影 → 残差
            self.gemm1(&self.attn, &w.o_proj, 1, h, nqk, &self.c_o)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_o.as_ptr() as *const f32, self.oadd.as_ptr() as *mut u16, h as u32)?;
            self.kernels.launch_add(dev, stream, self.x.as_ptr() as *mut u16, self.oadd.as_ptr() as *const u16, h as u32)?;

            // FFN
            self.kernels.launch_rms_norm(
                dev, stream,
                self.x.as_ptr() as *const u16,
                self.xn.as_ptr() as *mut u16,
                w.ffn_norm.as_ptr() as *const u16,
                h as u32, eps,
            )?;
            self.gemm1(&self.xn, &w.gate_proj, 1, ffn, h, &self.c_g)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_g.as_ptr() as *const f32, self.gate.as_ptr() as *mut u16, ffn as u32)?;
            self.gemm1(&self.xn, &w.up_proj, 1, ffn, h, &self.c_u)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_u.as_ptr() as *const f32, self.up.as_ptr() as *mut u16, ffn as u32)?;
            self.kernels.launch_swiglu(
                dev, stream,
                self.gate.as_ptr() as *const u16,
                self.up.as_ptr() as *const u16,
                self.down.as_ptr() as *mut u16,
                ffn as u32,
            )?;
            self.gemm1(&self.down, &w.down_proj, 1, h, ffn, &self.c_d)?;
            self.diff.launch_cast_f32_f16(dev, stream, self.c_d.as_ptr() as *const f32, self.l2.as_ptr() as *mut u16, h as u32)?;
            self.kernels.launch_add(dev, stream, self.x.as_ptr() as *mut u16, self.l2.as_ptr() as *const u16, h as u32)?;
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

        // final norm → lm_head → logits
        self.kernels.launch_rms_norm(
            dev, stream,
            self.x.as_ptr() as *const u16,
            self.xn.as_ptr() as *mut u16,
            self.final_norm.as_ptr() as *const u16,
            h as u32, eps,
        )?;
        self.gemm1(&self.xn, &self.lm_head, 1, self.cfg.vocab_size, h, &self.logits)?;
        stream.synchronize()?;

        copy(
            &mut MemRef::Host(&self.logits_host),
            &MemRef::Device(&self.logits),
            self.cfg.vocab_size * 4,
            None,
        )?;
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.logits_host.as_ptr() as *const f32,
                self.cfg.vocab_size,
            )
            .to_vec()
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
        // prefill：每个 prompt token 一步（KV 写 pos）
        for (i, &tok) in prompt_ids.iter().enumerate() {
            self.step(tok, i, i + 1)?;
        }
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

    fn gemm1(
        &self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        c: &DeviceBuffer,
    ) -> Result<(), EngineError> {
        let amat = GpuMat {
            ptr: a.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: k as c_int,
        };
        let bmat = GpuMat {
            ptr: b.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: n as c_int,
        };
        let mut cmat = GpuMat {
            ptr: c.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_32F,
            ld: m as c_int,
        };
        self.gemm
            .gemm_f32acc(&self.stream, m as c_int, n as c_int, k as c_int, &amat, &bmat, &mut cmat, 1.0, 0.0)?;
        Ok(())
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

    fn upload_pages(&self, log_pages: usize) -> Result<(), EngineError> {
        let bytes = log_pages * 4;
        let hb = HostBuffer::alloc(bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.pages_host.as_ptr() as *const u8,
                hb.as_ptr() as *mut u8,
                bytes,
            );
        }
        copy(&mut MemRef::Device(&self.pages_dev), &MemRef::Host(&hb), bytes, None)?;
        Ok(())
    }

    fn upload_lens(&self, kv_len: u32) -> Result<(), EngineError> {
        let hb = HostBuffer::alloc(4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(&kv_len as *const u32 as *const u8, hb.as_ptr() as *mut u8, 4)
        }
        copy(&mut MemRef::Device(&self.lens_dev), &MemRef::Host(&hb), 4, None)?;
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
