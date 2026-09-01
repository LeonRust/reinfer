//! 007 T2 / 014 T9 smoke: CPU 全链路（REINFER_MODEL_GGUF env 注入）。
#![allow(clippy::unwrap_used)]
use reinfer_cpu::{Model, generate};
use reinfer_gguf::GgufReader;
use reinfer_tokenizer::Tokenizer;

#[test]
fn cpu_generates_tokens() {
    let Some(path) = std::env::var_os("REINFER_MODEL_GGUF") else {
        eprintln!("skipped: REINFER_MODEL_GGUF unset");
        return;
    };
    let reader = GgufReader::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&reader).unwrap();
    let mut model = Model::load(&reader).unwrap();
    let ids = tok.encode("Hello", false).unwrap();
    let g = generate(&mut model, &ids, 12, 0.0, tok.eos_token()).unwrap();
    assert!(!g.tokens.is_empty(), "no tokens generated");
    let text = tok.decode_all(&g.tokens);
    eprintln!("cpu generated: {:?} (eos:{})", text, g.ended_by_eos);
    let mut m2 = Model::load(&reader).unwrap();
    let g2 = generate(&mut m2, &ids, 12, 0.0, tok.eos_token()).unwrap();
    assert_eq!(g.tokens, g2.tokens, "cpu determinism");
}
