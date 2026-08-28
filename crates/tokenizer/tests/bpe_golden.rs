//! 014 T4 golden gate: tokenizer BPE encode vs llama-tokenize reference
//! (tests/golden/tokenizer-bpe.json, f280b2698, --no-parse-special).
//!
//! Env-gated on REINFER_MODEL_GGUF (the tokenizer comes from the GGUF
//! metadata; env-injected, no hardcoded model identity). Skips cleanly
//! when unset or when the golden file is missing.

use reinfer_gguf::GgufReader;
use reinfer_tokenizer::Tokenizer;

#[test]
fn bpe_encode_matches_goldens_100() {
    let Some(path) = std::env::var_os("REINFER_MODEL_GGUF") else {
        eprintln!("skipped: REINFER_MODEL_GGUF unset");
        return;
    };
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/tokenizer-bpe.json");
    if !golden_path.exists() {
        eprintln!("skipped: no tokenizer-bpe.json (run scripts/golden/gen_bpe_golden.sh)");
        return;
    }
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).unwrap()).unwrap();
    let reader = GgufReader::open(&path).expect("archive open");
    let tok = Tokenizer::from_gguf(&reader).expect("tokenizer load");
    assert_eq!(tok.add_bos(), doc["add_bos"].as_bool().unwrap());

    for item in doc["items"].as_array().unwrap() {
        let text = item["text"].as_str().unwrap();
        let expect_ids: Vec<u32> = item["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got_ids = tok.encode(text, false).unwrap();
        assert_eq!(got_ids, expect_ids, "encode mismatch for {text:?}");
        // decode roundtrip must reproduce the text (llama-side too).
        let decoded = tok.decode_all(&got_ids);
        assert_eq!(decoded, text, "decode roundtrip mismatch for {text:?}");
    }
}
