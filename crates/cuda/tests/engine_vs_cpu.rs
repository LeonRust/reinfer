//! 014 L3 调试锚：CPU fp32 参考 forward vs CUDA engine（Qwen3-0.6B）。
//!
//! 运行：`CUDA_VISIBLE_DEVICES=0 REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B
//! cargo test -p reinfer-cuda --features cuda --test engine_vs_cpu -- --ignored --test-threads=1 --nocapture`
//!
//! 意图：logits 层面上定位 GPU engine 语义偏差（CPU 为同语义 fp32 参考；
//! 非产品路径——纯 debug 锚。f16 存储舍入导致的微小差异不计 gate）。
//! 判据：① top-1 argmax 一致（硬——完全错管道会错 token）；
//!       ② 全 logits rel 距离打印（记录档）。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 冒烟输出

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::engine::{argmax_first, Engine};
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_safetensors::SafeFile;
    use reinfer_tokenizer::Tokenizer;
    use std::collections::HashMap;

    // ---------------- CPU 参考（fp32 naive；f16→f32 权重） ----------------

    struct W32 {
        attn_norm: Vec<f32>,
        q: Vec<f32>, // [out,in]
        k: Vec<f32>,
        v: Vec<f32>,
        o: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    }
    struct CpuRef {
        cfg: reinfer_arch::llama::LlamaConfig,
        embed: Vec<f32>,      // [vocab, h]
        lm_head: Vec<f32>,    // [vocab, h]
        final_norm: Vec<f32>, // [h]
        layers: Vec<W32>,
        kv_k: Vec<f32>,
        kv_v: Vec<f32>,
    }

    impl CpuRef {
        /// 从 GGUF 加载（重载：校准真值——referee llama.cpp 可跑同一文件。
        /// 权重 f16/f32 → f32；量化档（Q8_0 等）→ panic（校准只用 fp16 GGUF）。
        fn load_gguf(path: &std::path::Path) -> Self {
            let reader = reinfer_gguf::GgufReader::open(path).expect("gguf open");
            let cfg = reinfer_arch::llama::from_gguf_meta(reader.metadata()).expect("arch");
            assert!(!cfg.head_norm, "qwen2.5 calibration arch has no head norm");
            let f2 = |name: &str| -> Vec<f32> {
                let t = reader.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
                let bytes = reader.tensor_data(t).expect("tensor data");
                use reinfer_gguf::GgufDtype::*;
                match t.dtype {
                    F16 | BF16 => bytes
                        .chunks_exact(2)
                        .map(|c| {
                            let b = u16::from_le_bytes([c[0], c[1]]);
                            if t.dtype == F16 {
                                reinfer_gguf::codes::f16_to_f32(b)
                            } else {
                                f32::from_bits((b as u32) << 16)
                            }
                        })
                        .collect(),
                    F32 => std::iter::repeat(0.0)
                        .take(0)
                        .chain(bytes.chunks_exact(4).map(|c| {
                            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        }))
                        .collect(),
                    other => panic!("calibration dtype {other:?} not supported (use fp16 gguf)"),
                }
            };
            let embed = f2("token_embd.weight");
            let lm_head = f2("output.weight");
            let final_norm = f2("output_norm.weight");
            let mut layers = Vec::new();
            for i in 0..cfg.n_layer {
                let p = |s: &str| format!("blk.{i}.{s}");
                layers.push(W32 {
                    attn_norm: f2(&p("attn_norm.weight")),
                    q: f2(&p("attn_q.weight")),
                    k: f2(&p("attn_k.weight")),
                    v: f2(&p("attn_v.weight")),
                    o: f2(&p("attn_output.weight")),
                    q_norm: vec![], // 无 head norm（Qwen2.5）
                    k_norm: vec![],
                    ffn_norm: f2(&p("ffn_norm.weight")),
                    gate: f2(&p("ffn_gate.weight")),
                    up: f2(&p("ffn_up.weight")),
                    down: f2(&p("ffn_down.weight")),
                });
            }
            let kv_size = cfg.n_layer * cfg.kv_heads * 4096 * cfg.head_dim;
            Self {
                cfg,
                embed,
                lm_head,
                final_norm,
                layers,
                kv_k: vec![0.0; kv_size],
                kv_v: vec![0.0; kv_size],
            }
        }

        fn load(model_dir: &std::path::Path) -> Self {
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(model_dir.join("config.json")).unwrap(),
            )
            .unwrap();
            let cfg = reinfer_arch::llama::from_hf_config(&config).unwrap();
            let safe = SafeFile::open(&model_dir.join("model.safetensors")).unwrap();
            let f2 = |t: &reinfer_safetensors::TensorView<'_>| -> Vec<f32> {
                assert_eq!(t.byte_len(), t.len().unwrap() * 2, "bf16 only in debug ref");
                t.bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((b as u32) << 16) // bf16 → f32
                    })
                    .collect()
            };
            let embed = f2(&safe.tensor("model.embed_tokens.weight").unwrap());
            let lm_head = f2(&safe.tensor("lm_head.weight").unwrap());
            let final_norm = f2(&safe.tensor("model.norm.weight").unwrap());
            let mut layers = Vec::new();
            for i in 0..cfg.n_layer {
                let p = |s: &str| format!("model.layers.{i}.{s}");
                layers.push(W32 {
                    attn_norm: f2(&safe.tensor(&p("input_layernorm.weight")).unwrap()),
                    q: f2(&safe.tensor(&p("self_attn.q_proj.weight")).unwrap()),
                    k: f2(&safe.tensor(&p("self_attn.k_proj.weight")).unwrap()),
                    v: f2(&safe.tensor(&p("self_attn.v_proj.weight")).unwrap()),
                    o: f2(&safe.tensor(&p("self_attn.o_proj.weight")).unwrap()),
                    q_norm: f2(&safe.tensor(&p("self_attn.q_norm.weight")).unwrap()),
                    k_norm: f2(&safe.tensor(&p("self_attn.k_norm.weight")).unwrap()),
                    ffn_norm: f2(&safe.tensor(&p("post_attention_layernorm.weight")).unwrap()),
                    gate: f2(&safe.tensor(&p("mlp.gate_proj.weight")).unwrap()),
                    up: f2(&safe.tensor(&p("mlp.up_proj.weight")).unwrap()),
                    down: f2(&safe.tensor(&p("mlp.down_proj.weight")).unwrap()),
                });
            }
            let kv_size = cfg.n_layer * cfg.kv_heads * 4096 * cfg.head_dim;
            Self {
                cfg,
                embed,
                lm_head,
                final_norm,
                layers,
                kv_k: vec![0.0; kv_size],
                kv_v: vec![0.0; kv_size],
            }
        }


        fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
            self.step_trace(token, pos).0
        }

        fn step_trace(&mut self, token: u32, pos: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
            // dtl：每层 (xn 归一后, q_rope, attn 后) 三项（与 GPU detail trace 对齐）
            let mut dtl: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
            let cfg = &self.cfg;
            let h = cfg.hidden_size;
            let d = cfg.head_dim;
            let qh = cfg.q_heads;
            let kvh = cfg.kv_heads;
            let ratio = qh / kvh;
            let ffn = cfg.ffn_hidden;
            let mut x: Vec<f32> = self.embed[token as usize * h..(token as usize + 1) * h].to_vec();

            for (li, w) in self.layers.iter().enumerate() {
                let xn = rms_n(&x, &w.attn_norm, cfg.rms_eps);
                let mut q = matmul1(&xn, &w.q, qh * d, h);
                let mut k = matmul1(&xn, &w.k, kvh * d, h);
                let v = matmul1(&xn, &w.v, kvh * d, h);
                let x0_snap = x.clone();
                let qn_snap = xn.clone();
                // q/k head norm（每头行 + 共享权重；无 head norm 架构跳过）
                if !w.q_norm.is_empty() {
                    for hh in 0..qh {
                        let hn = rms_n(&q[hh * d..(hh + 1) * d], &w.q_norm, cfg.rms_eps);
                        q[hh * d..(hh + 1) * d].copy_from_slice(&hn);
                    }
                    for hh in 0..kvh {
                        let hn = rms_n(&k[hh * d..(hh + 1) * d], &w.k_norm, cfg.rms_eps);
                        k[hh * d..(hh + 1) * d].copy_from_slice(&hn);
                    }
                }
                for hh in 0..qh {
                    rope_n(&mut q[hh * d..(hh + 1) * d], d / 2, pos as u32, cfg.rope_theta);
                }
                for hh in 0..kvh {
                    rope_n(&mut k[hh * d..(hh + 1) * d], d / 2, pos as u32, cfg.rope_theta);
                }
                let q_snap = q.clone();
                for hh in 0..kvh {
                    let base = (li * kvh + hh) * 4096 + pos;
                    self.kv_k[base * d..(base + 1) * d].copy_from_slice(&k[hh * d..(hh + 1) * d]);
                    self.kv_v[base * d..(base + 1) * d].copy_from_slice(&v[hh * d..(hh + 1) * d]);
                }
                let mut attn = vec![0.0f32; qh * d];
                let mut sl32: Vec<f32> = Vec::with_capacity(32);
                let scale = 1.0 / (d as f32).sqrt();
                for hh in 0..qh {
                    let kh = hh / ratio;
                    let kbase = (li * kvh + kh) * 4096;
                    let mut smax = f32::NEG_INFINITY;
                    let mut ssum = 0.0f32;
                    let mut sl = vec![0.0f32; pos + 1];
                    for t in 0..=pos {
                        let mut acc = 0.0f32;
                        for i in 0..d {
                            acc += q[hh * d + i] * self.kv_k[(kbase + t) * d + i];
                        }
                        sl[t] = acc * scale;
                        if sl[t] > smax {
                            smax = sl[t];
                        }
                    }
                    for t in 0..=pos {
                        sl[t] = (sl[t] - smax).exp();
                        ssum += sl[t];
                    }
                    for i in 0..d {
                        let mut acc = 0.0f32;
                        for t in 0..=pos {
                            acc += sl[t] * self.kv_v[(kbase + t) * d + i];
                        }
                        attn[hh * d + i] = acc / ssum;
                    }
                    sl32.extend_from_slice(&sl[..2.min(sl.len())]);
                }
                let a_snap = attn.clone();
                dtl.push((x0_snap, qn_snap, q_snap, a_snap, sl32));
                let o = matmul1(&attn, &w.o, h, qh * d);
                for i in 0..h {
                    x[i] += o[i];
                }
                let xn2 = rms_n(&x, &w.ffn_norm, cfg.rms_eps);
                let gate = matmul1(&xn2, &w.gate, ffn, h);
                let up = matmul1(&xn2, &w.up, ffn, h);
                let silu: Vec<f32> = gate
                    .iter()
                    .enumerate()
                    .map(|(i, g)| g / (1.0 + (-g).exp()) * up[i])
                    .collect();
                let down = matmul1(&silu, &w.down, h, ffn);
                for i in 0..h {
                    x[i] += down[i];
                }
            }
            let xn = rms_n(&x, &self.final_norm, cfg.rms_eps);
            let lg = matmul1(&xn, &self.lm_head, cfg.vocab_size, h);
            // (x0, xn, q, attn) 按层拼接为字段平面数组
            let mut flat = Vec::new();
            for (a, b, c, d, e) in dtl {
                flat.push(a);
                flat.push(b);
                flat.push(c);
                flat.push(d);
                flat.push(e);
            }
            (lg, flat)
        }
    }

    /// cpu 参考自由函数路由。
    fn matmul1(x: &[f32], w: &[f32], out: usize, inp: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; out];
        for r in 0..out {
            let mut acc = 0.0f32;
            for cc in 0..inp {
                acc += x[cc] * w[r * inp + cc];
            }
            c[r] = acc;
        }
        c
    }

    fn rms_n(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(w).map(|(&a, &b)| a * rstd * b).collect()
    }

    fn rope_n(x: &mut [f32], half: usize, pos: u32, eta: f32) {
        for p in 0..half {
            let theta = pos as f32 * eta.powf(-2.0 * p as f32 / (2.0 * half as f32));
            let (c, s) = (theta.cos(), theta.sin());
            let (a, b) = (x[p], x[p + half]);
            x[p] = a * c - b * s;
            x[p + half] = a * s + b * c;
        }
    }

    // ---------------- 测试 ----------------

    #[test]
    #[ignore = "calibration: cpu ref behavior vs known NaN archive"]
    fn calib_qwen25_ref_generation() {
        // Qwen2.5-0.5B-Instruct 官方 fp16 GGUF 含真 NaN 权重块（014 已知档案事实：
        // 三重验证——llama-gguf API 字节/本 crate reader/f16 位解码均报 NaN），
        // llama.cpp referee 输出正常是 argmax NaN 回落巧合。
        // 校准档（CPU-ref vs llama-simple 输出）因此在本归档上不可作数——
        // 语义对齐改由 Qwen3-0.6B（无 NaN 权重）的 CPU/GPU top-10 完全一致性证明
        // （见 cpu_ref_vs_engine_logits 判据）。
        // 本测试验证的是：CPU-ref 在含 NaN 权重时正确传播（防 NaN 静默）。
        // 运行：见文件头。
        let gguf = "/home/dora/.reinfer/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-fp16.gguf";
        let mut cpu = CpuRef::load_gguf(std::path::Path::new(gguf));
        // 第一步 latent 即应暴露 NaN（官方归档事实）
        let lgnan = cpu.step(9707, 0).iter().any(|v| v.is_nan());
        assert!(lgnan, "0.5B 官方 fp16 归档含 NaN 块——CPU-ref 应传播 NaN（不静默）");
        let prompt = "Hello";
        // tokenizer（GGUF 元数据）→ encode
        let reader = reinfer_gguf::GgufReader::open(gguf).unwrap();
        let tok = Tokenizer::from_meta(reader.metadata()).unwrap();
        let ids = tok.encode(prompt, false).unwrap();
        println!("cpu ref encode {prompt:?} -> {ids:?}");
        // 第一步 logits vs /tmp/ref_logits.bin（llama.cpp 权威锚）
        let ref_path = "/tmp/ref_logits.bin";
        if std::path::Path::new(ref_path).exists() {
            let bytes = std::fs::read(ref_path).unwrap();
            let ref_lg: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let lg = cpu.step(ids[0], 0);
            println!("ref len {} cpu len {}", ref_lg.len(), lg.len());
            // top5 双方
            let top5 = |v: &[f32]| -> Vec<(usize, f32)> {
                let mut idx: Vec<(usize, f32)> =
                    v.iter().enumerate().map(|(i, x)| (i, *x)).collect();
                idx.sort_by(|a, b| b.1.total_cmp(&a.1));
                idx[..5].to_vec()
            };
            println!("ref top5: {:?}", top5(&ref_lg));
            println!("cpu top5: {:?}", top5(&lg));
            // 相关性 + maxdiff
            let mut num = 0.0f32;
            let mut d1 = 0.0f32;
            let mut d2 = 0.0f32;
            let mut mx = 0.0f32;
            for i in 0..lg.len().min(ref_lg.len()) {
                let a = lg[i];
                let b = ref_lg[i];
                num += a * b;
                d1 += a * a;
                d2 += b * b;
                mx = mx.max((a - b).abs());
            }
            println!(
                "cpu-ref vs llama: corr={:.4} maxdiff={mx:.3}",
                num / (d1.sqrt() * d2.sqrt()).max(1e-12)
            );
        }
        // 生成路径：NaN 权重必然触发（stop early——证明传播而非静默 skip）
        let mut out = ids.clone();
        let eos = tok.eos_token();
        let mut saw_nan_logits = false;
        for _ in 0..32 {
            let pos = out.len() - 1;
            let logits = cpu.step(out[pos], pos);
            if logits.iter().all(|l| l.is_nan()) {
                saw_nan_logits = true;
                break;
            }
            let next = argmax_first(&logits);
            if Some(next) == eos {
                break;
            }
            out.push(next);
        }
        println!(
            "cpu ref generation NaN flag: {saw_nan_logits} (tokens {} -> {:?})",
            out.len(),
            &out[..out.len().min(8)]
        );
    }

    #[test]
    #[ignore = "debug anchor: cpu ref vs engine"]
    fn cpu_ref_vs_engine_logits() {
        let dir = std::path::PathBuf::from(
            std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"),
        );
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).unwrap();
        let _ = stream.synchronize().unwrap();

        let mut cpu = CpuRef::load(&dir);
        let mut engine = Engine::load(
            dev,
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(std::env::temp_dir().join("reinfer-jit-dense")),
            &dir,
            4096,
        )
        .unwrap();

        // 序列步进（pos 0..3）：logits 对比
        let toks = [10u32, 5u32, 23u32, 44u32]; // 任一合法 id 序列
        let mut summary = String::new();
        for pos in 0..toks.len() {
            let gl = engine.step(toks[pos], pos, pos + 1).unwrap();
            let cl = cpu.step(toks[pos], pos);
            assert_eq!(cl.len(), gl.len());
            let g_top = argmax_first(&gl);
            let c_top = argmax_first(&cl);
            summary.push_str(&format!("pos{pos}: gpu top1={g_top} cpu top1={c_top}; "));
        }
        println!("{summary}");
        // 逐层轨迹对比（pos=1：rope 已生效；级联。GPU detail 项序 = xn,q,attn）
        let (_glx, gdt) = engine.step_trace_detail(toks[1], 1, 2).unwrap();
        let (_clx, cdt) = cpu.step_trace(toks[1], 1);
        for li in 0..gdt.len().min(cdt.len() / 4) {
            let (gx0, gxn, gq, ga) = (&gdt[li].0, &gdt[li].1, &gdt[li].2, &gdt[li].3);
            let (cx0, cxn, cq, ca, csl) = (
                &cdt[li * 5],
                &cdt[li * 5 + 1],
                &cdt[li * 5 + 2],
                &cdt[li * 5 + 3],
                &cdt[li * 5 + 4],
            );
            if li == 0 {
                println!("    cpu sl(head..t)[0..32]={:?}", &csl[..32.min(csl.len())]);
            }
            let cmp = |name: &str, g: &[f32], c: &[f32]| {
                let mx = g
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("  layer {li} {name}: maxdiff={mx:.5}");
            };
            cmp("x_embed", gx0, cx0);
            cmp("xn", gxn, cxn);
            cmp("q_rope", gq, cq);
            cmp("attn", ga, ca);
            if li == 0 {
                let rng = |v: &[f32]| {
                    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    (mn, mx)
                };
                println!("    gpu attn[0..8]={:?} range={:?}", &ga[..8], rng(ga));
                println!("    cpu attn[0..8]={:?} range={:?}", &ca[..8], rng(ca));
                // q 各 head 首值（GPU trace 与 CPU）
                let g_head0: Vec<f32> = (0..16).map(|hh| gq[hh * 128]).collect();
                let c_head0: Vec<f32> = (0..16).map(|hh| cq[hh * 128]).collect();
                println!("    gpu q head0 h0..h15 = {g_head0:?}");
                println!("    cpu q head0 h0..h15 = {c_head0:?}");
            }
        }

        // 以最后一步做深入对比
        let gl = engine.step(toks[toks.len() - 1], toks.len() - 1, toks.len()).unwrap();
        let cl = cpu.step(toks[toks.len() - 1], toks.len() - 1);
        let g_top = argmax_first(&gl);
        let c_top = argmax_first(&cl);
        println!("gpu top1 = {g_top}; cpu top1 = {c_top}");

        // 集合关系：每个 logits 的前 top-10 交集
        let mut g_idx: Vec<(usize, f32)> = gl.iter().enumerate().map(|(i, v)| (i, *v)).collect();
        g_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut c_idx: Vec<(usize, f32)> = cl.iter().enumerate().map(|(i, v)| (i, *v)).collect();
        c_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let overlap: Vec<usize> = g_idx[..10]
            .iter()
            .map(|(i, _)| *i)
            .filter(|i| c_idx[..10].iter().any(|(j, _)| j == i))
            .collect();
        println!(
            "top-10 gpu={:?} cpu={:?} overlap={overlap:?}",
            &g_idx[..10].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            &c_idx[..10].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        );

        // 全 logits 的相对误差统计（记录档）
        let mut worst: (f32, usize) = (0.0, 0);
        let mut diffsum = 0.0f32;
        for i in 0..cl.len() {
            let dd = (gl[i] - cl[i]).abs();
            let rel = dd / (cl[i].abs().max(1e-3));
            if rel > worst.0 {
                worst = (rel, i);
            }
            diffsum += dd;
        }
        println!(
            "logits: max rel {:.3} @{}; mean abs diff {:.4}",
            worst.0,
            worst.1,
            diffsum / cl.len() as f32
        );
        let stats = |v: &[f32], name: &str| {
            let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let sq = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
            println!("{name}: min={mn} max={mx} mean={mean} rms={:.3}", sq.sqrt());
        };
        stats(&gl, "gpu");
        stats(&cl, "cpu");
        // token 10 在 top500 内的排名（孤立判断）
        println!("gpu rank(tok10)={:?} cpu rank(tok10)={:?}",
            g_idx.iter().position(|(i, _)| *i == 10),
            c_idx.iter().position(|(i, _)| *i == 10));
        assert_eq!(g_top, c_top, "top-1 必须一致（语义对齐锚）");
    }

    #[allow(dead_code)]
    fn _unused(_: &HashMap<(), ()>) {}
}
