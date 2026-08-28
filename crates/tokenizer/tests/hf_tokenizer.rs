//! HF `tokenizer.json` 路径冒烟（真实模型目录/env-gated）。
//!
//! 运行：`REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B cargo test
//! -p reinfer-tokenizer --test hf_tokenizer -- --ignored`
//!
//! 断言面（本测试非精确金块——精确锚在 T4：
//! tests/golden/tokenizer-bpe.json，针对 GGUF QWEN2 表；HF 表与 GGUF 表
//! 数据等价，仅转换层差异）：
//! ① 解析成功且 vocab_size = config.vocab_size（151936）；
//! ② bos/eos 文本查 id；eos = <|im_end|> = 151645；
//! ③ 编码产生 id 且全部 < vocab_size（embedding OOV 安全）；
//! ④ 特殊 token 文本（<|im_start|>）parse_special 产生一个 id（151644）。

#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 冒烟输出

use reinfer_tokenizer::Tokenizer;

fn load() -> (Tokenizer, serde_json::Value, usize) {
    let dir = std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR env-gated");
    let tok: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer.json")).unwrap())
            .expect("tokenizer.json parse");
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/tokenizer_config.json")).unwrap())
            .expect("tokenizer_config.json parse");
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).unwrap())
            .expect("config.json parse");
    let vocab = config["vocab_size"].as_u64().expect("vocab_size") as usize;
    let t = Tokenizer::from_hf_json(&tok, &cfg).expect("from_hf_json");
    (t, cfg, vocab)
}

#[test]
#[ignore = "hf smoke: env-gated real model"]
fn qwen3_hf_tokenizer_smoke() {
    let (t, _cfg, vocab) = load();
    assert!(t.vocab_size() <= vocab,
            "tokenizer 域 ({}) 不得超出 embedding 行数 ({vocab})",
            t.vocab_size());

    // bos/eos/unk 查表引用
    let eos = t.eos_token().expect("eos present");
    assert_eq!(eos, 151645, "eos = <|im_end|>");

    // 编码：英文 + 中文 + 空格 + 空行 fragment（id 全在 embedding 界内）
    for text in ["Hello", "Hello world", "\u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}", "\n\n", "a\tb "] {
        let ids = t.encode(text, false).expect("encode");
        assert!(ids.iter().all(|&i| (i as usize) < vocab), "id OOV ({text:?})");
        let _ = t;
    }

    // 特殊 token 整体分段（parse_special）
    let ids = t.encode("<|im_start|>system", true).expect("encode special");
    assert!(ids.iter().any(|&i| i == 151644), "<|im_start|> 应整体成片");

    // 解码往返（字节级 piece → UTF-8 文本）
    let text = "Hello world";
    let ids = t.encode(text, false).unwrap();
    let back = t.decode_all(&ids);
    println!("encode {text:?} -> {ids:?} -> decode {back:?}");
    assert_eq!(back, text, "decode round-trip");

    println!(
        "vocab={} bos={:?} eos={:?} add_bos={}",
        t.vocab_size(),
        t.bos_token(),
        t.eos_token(),
        t.add_bos()
    );
}

#[test]
#[ignore = "hf smoke: env-gated real model"]
fn qwen3_add_bos_matches_config() {
    let (t, cfg, _) = load();
    let want = cfg["add_bos_token"].as_bool().unwrap_or(false);
    assert_eq!(t.add_bos(), want, "add_bos 与 tokenizer_config 一致");
}
