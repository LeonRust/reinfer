//! CPU 后端模型（fp32 累加 naive；007 T1 —「无加速卡也能推理」）。
//!
//! 负载：GGUF → 权重字节全部读入（Q8_0 blob 677MB 级 → dequant 按需按层
//! 进行（layer 计算时 f16/解码产物进入 f32 临时缓冲；KV 连续矩形缓存
//! [layers][kv_heads][max_seq][d]——CPU 不做 paged。
//!
//! 生成语义（014 T9 必备块）：EOS 停 / `-n` 硬限 / logits 全 NaN 显式错误 /
//! embedding OOV → RunError / `-t 0` 短路 argmax（tie-break 首个最大）。

use crate::RunError;
use reinfer_arch::llama::from_gguf_meta;
use reinfer_arch::llama::LlamaConfig;

use reinfer_gguf::codes;
use reinfer_gguf::GgufDtype;
use reinfer_gguf::GgufReader;

/// 层原始权重（blob 字节——按 dtype 按需解量化）。
#[derive(Debug)]
pub struct LayerRaw {

    /// attn_norm（f16/f32 字节）
    pub attn_norm: Vec<u8>,
    /// q (prow [d*n_heads, hidden])——GGUF 布局：[out, in] 行主序
    pub q: Vec<u8>,
    /// 权重字节 blob。
    pub k: Vec<u8>,
    /// 权重字节 blob。
    pub v: Vec<u8>,
    /// 权重字节 blob。
    pub o: Vec<u8>,
    /// 投影偏置（f16 blob；可选）。
    pub q_bias: Option<Vec<u8>>,
    /// 投影偏置（f16 blob；可选）。
    pub k_bias: Option<Vec<u8>>,
    /// 投影偏置（f16 blob；可选）。
    pub v_bias: Option<Vec<u8>>,
    /// 权重字节 blob。
    pub ffn_norm: Vec<u8>,
    /// 权重字节 blob。
    pub ffn_gate: Vec<u8>,
    /// 权重字节 blob。
    pub ffn_up: Vec<u8>,
    /// 权重字节 blob。
    pub ffn_down: Vec<u8>,
    /// 各权重 dtype（存储类型；布局头）。
    /// 矩阵权重 dtype（q/k/v/o/gate/up/down 同）。
    pub dtype_q: GgufDtype,
    /// norm 权重 dtype。
    pub dtype_attn: GgufDtype,
}

impl LayerRaw {
    /// 空层（占位/测试）。
    pub fn empty() -> Self {
        Self {
            attn_norm: vec![],
            q: vec![],
            k: vec![],
            v: vec![],
            o: vec![],
            q_bias: None,
            k_bias: None,
            v_bias: None,
            ffn_norm: vec![],
            ffn_gate: vec![],
            ffn_up: vec![],
            ffn_down: vec![],
            dtype_q: GgufDtype::F16,
            dtype_attn: GgufDtype::F16,
        }
    }
}

/// 后端模型（所有权重 blob + 配置 + KV 缓存）。
#[derive(Debug)]
pub struct Model {
    /// 架构配置。
    pub cfg: LlamaConfig,
    /// embedding 权重 [vocab×hidden]（dequant blob——大小大：0.5B vocab
    /// 151936×896 f32 = 545MB——以 f16 存储按需转语义：store as blob，
    /// 逐 token 查表时转换）。
    /// embedding 权重（存储 blob）。
    pub embed_blob: Vec<u8>,
    /// lm_head（output.weight；字节 blob）。若与 embed 同张量名则 None。
    /// lm_head 权重（tied 时 None）。
    pub lm_head_blob: Option<Vec<u8>>,
    /// 全部层权重。
    /// 全部层权重。
    pub layers: Vec<LayerRaw>,
    /// final norm（`output_norm.weight`）。
    /// final norm 权重（output_norm.weight）。
    pub final_norm: Vec<u8>,
    /// embed / lm_head / final_norm 的存储 dtype。
    /// embedding 存储 dtype。
    pub embed_dtype: GgufDtype,
    /// lm_head 存储 dtype。
    pub head_dtype: GgufDtype,
    /// final norm 存储 dtype。
    pub final_dtype: GgufDtype,
    /// KV：K/V 连续矩形 [layers][kv_heads][max_seq][d]（f32）。
    /// K 缓存 [layers][kv_heads][ctx][d]。
    pub kv_k: Vec<f32>,
    /// V 缓存（同 K 布局）。
    pub kv_v: Vec<f32>,
}

impl Model {
    /// 从 GGUF reader 装载整个模型（所有权重字节入内存）。
    pub fn load(reader: &GgufReader) -> Result<Self, RunError> {
        let meta = reader.metadata();
        let cfg = from_gguf_meta(meta).map_err(RunError::Arch)?;
        let n = cfg.n_layer;
        let embed = reader
            .tensor("token_embd.weight")
            .ok_or_else(|| RunError::MissingTensor("token_embd.weight".into()))?;
        let embed_blob = reader.tensor_data(embed).map_err(RunError::Gguf)?;
        let embed_dtype = embed.dtype;

        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let name = |s: &str| format!("blk.{i}.{s}");
            let get = |s: &str| -> Result<Vec<u8>, RunError> {
                let t = reader
                    .tensor(&name(s))
                    .ok_or_else(|| RunError::MissingTensor(name(s)))?;
                reader.tensor_data(t).map_err(RunError::Gguf)
            };
            let dt = |s: &str| -> Result<GgufDtype, RunError> {
                reader
                    .tensor(&name(s))
                    .map(|t| t.dtype)
                    .ok_or_else(|| RunError::MissingTensor(name(s)))
            };
            let opt = |s: &str| -> Option<Vec<u8>> {
                reader.tensor(&name(s)).and_then(|t| reader.tensor_data(t).ok())
            };
            eprintln!(
                "[debug] blk0 q dtype={:?} len={:?} k len={:?} o len={:?}",
                reader.tensor(&name("attn_q.weight")).map(|t| t.dtype),
                reader.tensor(&name("attn_q.weight")).map(|t| t.length),
                reader.tensor(&name("attn_k.weight")).map(|t| t.length),
                reader.tensor(&name("attn_output.weight")).map(|t| t.length)
            );
            layers.push(LayerRaw {
                attn_norm: get("attn_norm.weight")?,
                q: get("attn_q.weight")?,
                k: get("attn_k.weight")?,
                v: get("attn_v.weight")?,
                o: get("attn_output.weight")?,
                q_bias: opt("attn_q.bias"),
                k_bias: opt("attn_k.bias"),
                v_bias: opt("attn_v.bias"),
                ffn_norm: get("ffn_norm.weight")?,
                ffn_gate: get("ffn_gate.weight")?,
                ffn_up: get("ffn_up.weight")?,
                ffn_down: get("ffn_down.weight")?,
                dtype_q: dt("attn_q.weight")?,
                dtype_attn: dt("attn_norm.weight")?,
            });
        }

        let (final_norm, final_dtype) = {
            let t = reader
                .tensor("output_norm.weight")
                .ok_or_else(|| RunError::MissingTensor("output_norm.weight".into()))?;
            (reader.tensor_data(t).map_err(RunError::Gguf)?, t.dtype)
        };

        let (lm_head_blob, head_dtype) = match reader.tensor("output.weight") {
            Some(t) => (Some(reader.tensor_data(t).map_err(RunError::Gguf)?), t.dtype),
            None => (None, embed_dtype),
        };

        let kv_cap = n * cfg.kv_heads * cfg.ctx_len.max(512) * cfg.head_dim;
        Ok(Self {
            cfg,
            embed_blob,
            lm_head_blob,
            layers,
            final_norm,
            embed_dtype,
            head_dtype,
            final_dtype,
            kv_k: vec![0.0f32; kv_cap],
            kv_v: vec![0.0f32; kv_cap],
        })
    }

    /// embedding（OOV → 错误）。
    pub fn embed_vec(&self, token: u32) -> Result<Vec<f32>, RunError> {
        let d = self.cfg.hidden_size;
        let idx = token as usize;
        let row_bytes = crate::ops::row_bytes(self.embed_dtype, d)?;
        let start = idx * row_bytes;
        if start + row_bytes > self.embed_blob.len() {
            return Err(RunError::EmbeddingOov(token));
        }
        let blob = &self.embed_blob[start..start + row_bytes];
        match self.embed_dtype {
            GgufDtype::Q8_0 => {
                let mut out = vec![0.0f32; d];
                codes::dequantize_q8_0(blob, &mut out)?;
                Ok(out)
            }
            _ => crate::ops::weight_to_f32(blob, self.embed_dtype),
        }
    }

    /// KV 索引归一（layers·kv_heads·seq·d）。
    #[inline]
    pub fn kv_slot(&self, layer: usize, kv_head: usize, pos: usize) -> (usize, usize) {
        let d = self.cfg.head_dim;
        let stride = d;
        let base =
            ((layer * self.cfg.kv_heads + kv_head) * self.cfg.ctx_len.max(512) + pos) * stride;
        (base, base + d)
    }

    /// logits 投影（lm_head 或 tied embedding）。
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, RunError> {
        let vocab = self.cfg.vocab_size;
        let d = self.cfg.hidden_size;
        assert_eq!(hidden.len(), d);
        let head = match &self.lm_head_blob {
            Some(b) => b,
            None => &self.embed_blob,
        };
        let row_bytes = crate::ops::row_bytes(self.head_dtype, d)?;
        if head.len() / row_bytes < vocab {
            return Err(RunError::WeightShape("lm_head".into()));
        }
        let mut out = vec![0.0f32; vocab];
        for v in 0..vocab {
            let row = &head[v * row_bytes..(v + 1) * row_bytes];
            let wrow = match self.head_dtype {
                GgufDtype::Q8_0 => {
                    let mut tmp = vec![0.0f32; d];
                    codes::dequantize_q8_0(row, &mut tmp)?;
                    tmp
                }
                _ => crate::ops::weight_to_f32(row, self.head_dtype)?,
            };
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += hidden[j] * wrow[j];
            }
            out[v] = acc;
        }
        Ok(out)
    }
}
