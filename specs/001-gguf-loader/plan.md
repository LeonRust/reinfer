# Plan: GGUF loader and typed core layer

> Derived from specs/001-gguf-loader/spec.md · Owner: implementer to confirm with maintainers

## Architecture Decision

- **Three-segment read**: header (magic/version/alignment) → metadata key-value list → tensor table; weights stay as raw mmap byte ranges, decoded lazily on kernel launch
- **Handle indirection**: `TensorId(u32)` hash map (name → id) built once at open; arch/model builders never parse strings again
- **Trait-based quant decode**: `trait QuantCodec { fn dequant_to_f32(&self, bytes: &[u8]) -> Vec<f32> }` with a naive reference implementation first (constitution §2.2: every OpKind needs a CPU reference); SIMD fast path is a later task in this same slice
- **Crate boundaries**: `reinfer-core` provides `DType`, `TensorId`, `Error`; `reinfer-gguf` implements reader + codecs; `reinfer-arch` maps GGUF metadata → typed model config. core must never import gguf.

## Module Breakdown

1. `crates/core` — `dtype.rs` (F16/BF16/FP32/FP8 spec), `tensor_id.rs`, `error.rs`
2. `crates/gguf` — `header.rs`, `metadata.rs` (typed keys), `tensor.rs` (table + mmap views), `quant.rs` (codec trait + Q8_0/Q4_0/F16/F32 codecs), `error.rs`
3. `crates/arch` — `config.rs` (typed model config from metadata), `llama.rs` (dim validation for P0 scope)
4. `bin/reinfer` — `cli/info.rs` subcommand wiring

## Interface Contracts

```rust
// crates/gguf
pub struct GgufReader { /* file handle + mmap range views + tensor table */ }
impl GgufReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError>;
    pub fn metadata(&self) -> &GgufMetadata;             // typed, namespaced keys
    pub fn tensor(&self, name: &str) -> Option<TensorId>;
    pub fn tensor_descriptor(&self, id: TensorId) -> TensorDescriptor; // shape/dtype/offset
    pub fn bytes(&self, id: TensorId) -> Option<&[u8]>;  // lazy range view
}

// crates/gguf::quant
pub trait QuantCodec {
    fn kind(&self) -> QuantKind;
    fn dequant_to_f32(&self, block: &[u8]) -> Vec<f32>;  // naive reference first
}

// crates/core
pub type TensorId = u32;
pub enum DType { F16, BF16, FP32, Q8_0, Q4_0, ... }

// bin/reinfer
fn cmd_info(model: &Path) -> Result<(), CliError>;
```

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| GGUF format drift vs llama.cpp (new quant kind / new metadata key) | High | Implement known set only; unknown → explicit error with byte offset, RFC for new kinds |
| SIMD decode slower than llama.cpp at first | Medium | Naive first, latency to P0 fallback (60% baseline allows it); SIMD task included in this slice as stretch |
| Golden-file tests need converted weights | Medium | Pre-generate tiny weights via llama.cpp conversion script once, commit under `tests/data/` (< 10 MB) or CI cache |
| Endianness/alignment misuse in mmap views | Medium | byteorder + explicit alignment checks + proptest fuzzing of offsets |
