//! Metadata dump probe — 014 T1 real-model archive check tooling.
//!
//! Usage: `cargo run -p reinfer-gguf --example dump_meta -- <path.gguf>`
//! Outputs every metadata key (with type tag) and the tensor table summary
//! to stdout as a line-oriented text stream: `key <TYPE> <value...>` and
//! `tensor <name> <dtype> <nelements>`.
//!
//! The paired `llama-gguf dump` (referee, 014 T0) output is compared key by
//! key by `scripts/golden/archive_check.sh` (014 T1 verification gate).
//! Real-model paths are injected via argv — no model identity is hardcoded.

use reinfer_gguf::{ArrayValue, GgufReader, MetaValue};
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dump_meta <path.gguf>");
        return ExitCode::FAILURE;
    };
    let reader = match GgufReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("open failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let meta = reader.metadata();
    for (key, value) in meta.iter() {
        println!("key {key} {}", dump_value(value));
    }
    for t in reader.tensors() {
        println!(
            "tensor {} {:?} {}",
            t.name, t.dtype, t.element_count().unwrap_or(0)
        );
    }
    ExitCode::SUCCESS
}

/// Element type tag of an array (for the summary line).
fn array_kind(v: &ArrayValue) -> &'static str {
    match v {
        ArrayValue::U8(_) => "u8",
        ArrayValue::I8(_) => "i8",
        ArrayValue::U16(_) => "u16",
        ArrayValue::I16(_) => "i16",
        ArrayValue::U32(_) => "u32",
        ArrayValue::I32(_) => "i32",
        ArrayValue::F32(_) => "f32",
        ArrayValue::Bool(_) => "bool",
        ArrayValue::Str(_) => "str",
        ArrayValue::U64(_) => "u64",
        ArrayValue::I64(_) => "i64",
        ArrayValue::F64(_) => "f64",
        ArrayValue::Nested(_) => "nested",
    }
}

/// Element count of an array (Nested = items count).
fn array_len(v: &ArrayValue) -> usize {
    match v {
        ArrayValue::U8(x) => x.len(),
        ArrayValue::I8(x) => x.len(),
        ArrayValue::U16(x) => x.len(),
        ArrayValue::I16(x) => x.len(),
        ArrayValue::U32(x) => x.len(),
        ArrayValue::I32(x) => x.len(),
        ArrayValue::F32(x) => x.len(),
        ArrayValue::Bool(x) => x.len(),
        ArrayValue::Str(x) => x.len(),
        ArrayValue::U64(x) => x.len(),
        ArrayValue::I64(x) => x.len(),
        ArrayValue::F64(x) => x.len(),
        ArrayValue::Nested(x) => x.len(),
    }
}

/// One-value line-oriented form: `<TYPE> <value>` — the type tag is the first
/// token so `scripts/golden/archive_check.sh` can compare types without
/// parsing array payloads in full.
fn dump_value(v: &MetaValue) -> String {
    match v {
        MetaValue::U8(x) => format!("u8 {x}"),
        MetaValue::I8(x) => format!("i8 {x}"),
        MetaValue::U16(x) => format!("u16 {x}"),
        MetaValue::I16(x) => format!("i16 {x}"),
        MetaValue::U32(x) => format!("u32 {x}"),
        MetaValue::I32(x) => format!("i32 {x}"),
        MetaValue::F32(x) => format!("f32 {x}"),
        MetaValue::Bool(x) => format!("bool {x}"),
        MetaValue::Str(x) => format!("str {x}"),
        MetaValue::Array(x) => format!("array {:?} {} items", array_kind(x), array_len(x)),
        MetaValue::U64(x) => format!("u64 {x}"),
        MetaValue::I64(x) => format!("i64 {x}"),
        MetaValue::F64(x) => format!("f64 {x}"),
    }
}
