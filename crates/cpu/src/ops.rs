//! CPU 后端算子（fp32 累加 naive；007 T1 — 数值基准语义）。
//!
//! 单线程顺序（IEEE fp32）：同输入跨进程位级稳定（确定性声明范围：同机
//! 跨机）。矩阵乘累加顺序与 `kernels::refs::matmul_ref` 一致（i→t→j）。

use crate::model::Model;
use crate::RunError;
use reinfer_gguf::codes;
use reinfer_gguf::GgufDtype;

/// 单行字节数（行 = `d` 元素；Q8_0 行按 32 元素块对齐——d 需为 32 的倍数）。
pub fn row_bytes(dtype: GgufDtype, d: usize) -> Result<usize, RunError> {
    match dtype {
        GgufDtype::F16 => Ok(2 * d),
        GgufDtype::F32 => Ok(4 * d),
        GgufDtype::Q8_0 => {
            if !d.is_multiple_of(32) {
                return Err(RunError::UnsupportedDtype(format!(
                    "q8_0 row with d={d} (not block-aligned)"
                )));
            }
            Ok(d / 32 * 34)
        }
        other => Err(RunError::UnsupportedDtype(format!("{other:?}"))),
    }
}

/// 权重字节 → f32 行主序矩阵（支持 F16/F32/Q8_0；其它 → 错误）。
pub fn weight_to_f32(blob: &[u8], dtype: GgufDtype) -> Result<Vec<f32>, RunError> {
    match dtype {
        GgufDtype::F16 => {
            if !blob.len().is_multiple_of(2) {
                return Err(RunError::WeightShape(format!("f16 bytes {}", blob.len())));
            }
            Ok(blob
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| codes::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GgufDtype::F32 => Ok(blob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GgufDtype::Q8_0 => {
            let mut out = vec![0.0f32; blob.len() / 34 * 32];
            codes::dequantize_q8_0(blob, &mut out)?;
            Ok(out)
        }
        other => Err(RunError::UnsupportedDtype(format!("{other:?}"))),
    }
}

/// 行主序 matmul：C[m×n] = A[m×k] · B[k×n]（fp32 累加）。
pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for t in 0..k {
            let av = a[i * k + t];
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += av * b[t * n + j];
            }
        }
    }
    c
}

/// SwiGLU FFN（Qwen 式：gate → silu(gate) * up → down）。
pub fn ffn_swiglu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let n = x.len() / hidden;
    let gate = matmul(x, w_gate, n, inter, hidden);
    let up = matmul(x, w_up, n, inter, hidden);
    let gat = gate
        .into_iter()
        .zip(up)
        .map(|(g, u)| g * (1.0 / (1.0 + (-g).exp())) * u)
        .collect::<Vec<f32>>();
    matmul(&gat, w_down, n, hidden, inter)
}

/// Neox RoPE（半维旋转——与 012 rope_ref 同公式）。
pub fn rope_inplace(x: &mut [f32], half: usize, pos: u32, eta: f32) {
    let sin_cos = |p: usize| -> (f32, f32) {
        let theta = (pos as f32) * eta.powf(-2.0 * (p as f32) / (2.0 * half as f32));
        (theta.sin(), theta.cos())
    };
    for p in 0..half {
        let (s, c) = sin_cos(p);
        let (a, b) = (x[p], x[half + p]);
        x[p] = a * c - b * s;
        x[half + p] = a * s + b * c;
    }
}

/// 单 q 行的 attention（K/V 已 RoPE/写缓存；连续行主序 [kv_len][d]）。
pub fn attention_query(q: &[f32], k: &[f32], v: &[f32], d: usize, kv_len: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; kv_len];
    for t in 0..kv_len {
        let mut acc = 0.0f32;
        for i in 0..d {
            acc += q[i] * k[t * d + i];
        }
        s[t] = acc;
    }
    let maxv = s.iter().copied().reduce(f32::max).unwrap_or(f32::NEG_INFINITY);
    let mut sum = 0.0f32;
    let mut p = vec![0.0f32; kv_len];
    if maxv.is_finite() {
        for (i, x) in s.iter().enumerate() {
            let e = (x - maxv).exp();
            sum += e;
            p[i] = e;
        }
        if sum > 0.0 {
            for vv in p.iter_mut() {
                *vv /= sum;
            }
        }
    }
    let mut out = vec![0.0f32; d];
    for t in 0..kv_len {
        let pv = p[t];
        if pv == 0.0 {
            continue;
        }
        for i in 0..d {
            out[i] += pv * v[t * d + i];
        }
    }
    out
}

/// 单 decode 步前向（已在 KV 写好的位置 `pos`）。
pub fn decode_step(
    model: &mut Model,
    emb: &[f32],
    pos: usize,
    kv_len: usize,
) -> Result<Vec<f32>, RunError> {
    let cfg = &model.cfg;
    let hidden = cfg.hidden_size;
    let d = cfg.head_dim;
    let n_heads = cfg.q_heads;
    let kv_heads = cfg.kv_heads;
    let ratio = n_heads / kv_heads;
    let mut x = emb.to_vec();

    for (li, layer) in model.layers.iter().enumerate() {
        // attn_norm → qkv
        let norm = weight_to_f32(&layer.attn_norm, GgufDtype::F16)?;
        let used = rmsnorm(&x, &norm, cfg.rms_eps);        let qw = weight_to_f32(&layer.q, layer.dtype_q)?;
        let kw = weight_to_f32(&layer.k, layer.dtype_q)?;
        let vw = weight_to_f32(&layer.v, layer.dtype_q)?;
        let ow = weight_to_f32(&layer.o, layer.dtype_q)?;
        let fg = weight_to_f32(&layer.ffn_gate, layer.dtype_q)?;
        let fu = weight_to_f32(&layer.ffn_up, layer.dtype_q)?;
        let fd = weight_to_f32(&layer.ffn_down, layer.dtype_q)?;
        // qkv 投影（+ bias）
        let mut q = matmul(&used, &qw, 1, n_heads * d, hidden);
        let mut k = matmul(&used, &kw, 1, kv_heads * d, hidden);
        let mut v = matmul(&used, &vw, 1, kv_heads * d, hidden);
        if let Some(b) = &layer.q_bias {
            add_bias_to(&mut q, b, layer.dtype_attn)?;
        }
        if let Some(b) = &layer.k_bias {
            add_bias_to(&mut k, b, layer.dtype_attn)?;
        }
        if let Some(b) = &layer.v_bias {
            add_bias_to(&mut v, b, layer.dtype_attn)?;
        }
        // RoPE（q 全头、k kv 头）
        for h in 0..n_heads {
            rope_inplace(&mut q[h * d..(h + 1) * d], d / 2, pos as u32, cfg.rope_theta);
        }
        for h in 0..kv_heads {
            rope_inplace(&mut k[h * d..(h + 1) * d], d / 2, pos as u32, cfg.rope_theta);
        }
        // KV 写入
        let (kb, _ke) = model.kv_slot(li, 0, pos);
        let stride = d;
        for h in 0..kv_heads {
            let (base, _) = model.kv_slot(li, h, pos);
            model.kv_k[base..base + stride].copy_from_slice(&k[h * d..(h + 1) * d]);
            model.kv_v[base..base + stride].copy_from_slice(&v[h * d..(h + 1) * d]);
        }
        let _ = kb;
        // attention
        let mut attn = vec![0.0f32; hidden];
        for h in 0..n_heads {
            let kh = h / ratio;
            let kbase = ((li * kv_heads + kh) * cfg.ctx_len.max(512)) * d;
            let vbase = ((li * kv_heads + kh) * cfg.ctx_len.max(512)) * d;
            let kv_slice_k = &model.kv_k[kbase..kbase + kv_len * d];
            let kv_slice_v = &model.kv_v[vbase..vbase + kv_len * d];
            let head = attention_query(
                &q[h * d..(h + 1) * d],
                kv_slice_k,
                kv_slice_v,
                d,
                kv_len.min(cfg.ctx_len),
            );
            attn[h * d..(h + 1) * d].copy_from_slice(&head);
        }

        let o = matmul(&attn, &ow, 1, hidden, hidden);
        for i in 0..hidden {
            x[i] += o[i];
        }
        // ffn
        let n1 = rmsnorm(&x, &weight_to_f32(&layer.ffn_norm, GgufDtype::F16)?, cfg.rms_eps);
        let f = ffn_swiglu(&n1, &fg, &fu, &fd, hidden, cfg.ffn_hidden);        for i in 0..hidden {
            x[i] += f[i];
        }
    }

    // final norm（Qwen: `output_norm.weight`）→ logits
    let fin = weight_to_f32(&model.final_norm.clone(), model.final_dtype)?;
    let xn = rmsnorm(&x, &fin, cfg.rms_eps);
    let out = model.logits(&xn)?;
    Ok(out)
}

/// RMSNorm 一行（与 refs::rms_norm_ref 相同语义——0 行 → 0）。
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum = 0.0f32;
    for v in x {
        sum += v * v;
    }
    let rstd = 1.0 / (sum / n as f32 + eps).sqrt();
    x.iter().zip(w.iter()).map(|(a, b)| a * rstd * b).collect()
}


/// 与投影同长的 bias（f16）逐元素加入。
fn add_bias_to(x: &mut [f32], bias: &[u8], dtype: GgufDtype) -> Result<(), RunError> {
    let b = weight_to_f32(bias, dtype)?;
    if b.len() != x.len() {
        return Err(RunError::WeightShape(format!(
            "bias len {} vs out {}",
            b.len(),
            x.len()
        )));
    }
    for (xi, bi) in x.iter_mut().zip(b) {
        *xi += bi;
    }
    Ok(())
}
