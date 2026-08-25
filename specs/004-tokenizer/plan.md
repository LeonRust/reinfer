# Plan: GGUF tokenizer

> Derived from specs/004-tokenizer/spec.md

## Architecture Decisions

- **D1 新 crate** `crates/tokenizer`：避免 gguf 膨胀；tokenizer 语义（BPE/SPM）是引擎资产（换 CUDA 实现仍需要 → reinfer 侧），数据源经 001 的 `GgufReader` 类型化读取。
- **D2 增量解码状态机**：`IncrementalDecoder { read_len, surr: [u8; 2], pending }` —— 每个 decode 调用先冲刷尾部不完整 UTF-8（记录于状态），实现"任意分块解码结果与整批一致"（mini-sglang 双 offset 模式）。
- **D3 合并顺序**：vocab 分数为 f32（对 SPM 用 merge-rank，对 BPE 用"字节级拼接后按 token 长度+分数"排序 — 与 llama.cpp `ggml-tokenizer` 对齐）；golden 文件做一次性对拍锚定，避免实现期猜测。
- **D4 错误面**：所有异常路径收敛为单一 `TokenizerError`（unk-missing / corrupt merges / unsupported model）；编码路径返回 `Result`；解码路径对坏 token id 输出 `[UNK]`（与 llama.cpp 一致），不 panic。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/tokenizer/src/model/mod.rs` | `TokenizerModel` trait + `Spm`/`Bpe` 类型 (+ unknown → error) |
| `crates/tokenizer/src/spm.rs` | byte-level unicode 映射表 + trie/linear 最贪心匹配 + unk fallback |
| `crates/tokenizer/src/bpe.rs` | 从 merges 构建 rune→id 表；`split.unicode.whitespace` 预处理 |
| `crates/tokenizer/src/decode.rs` | `IncrementalDecoder`（read/surr 状态机） |
| `crates/tokenizer/src/gguf.rs` | 元数据读取（`tokenizer.ggml.model` 等 12+ 键 + 特殊 token 标志位） |
| `crates/tokenizer/tests/golden/*.json` | llama.cpp 生成的对拍样例 |

## Interface Contracts (slice-local)

```rust
pub struct Tokenizer { model: TokenizerModel, special: SpecialTokens }
impl Tokenizer {
    pub fn from_gguf(g: &GgufReader) -> Result<Self, TokenizerError>;
    pub fn encode(&self, s: &str) -> Result<Vec<u32>, TokenizerError>; // 含 BOS 规则
    pub fn decode_all(&self, ids: &[u32]) -> Result<Cow<str>, TokenizerError>;
    pub fn unmatched_token(&self) -> u32;
}
pub struct IncrementalDecoder<'t> { t: &'t Tokenizer, read_len: usize, surr: [u8; 2] }
impl<'t> IncrementalDecoder<'t> {
    pub fn decode_chunk(&mut self, tokens: &[u32]) -> Result<String, TokenizerError>; // 字节级拼接+UTF8 洗牌
}
```

## Reference assets

- llama.cpp `src/llama-tokenizer.cpp` 与 `convert_hf_to_gguf.py` 的 `tokenizer.ggml.model` 数据格式 → 字段与合并规则
- mini-sglang `tokenizer_worker`（read/surr 双 offset + 增量 UTF-8）→ `IncrementalDecoder`
- 上游 `sentencepiece` BPE/SPM API 语义（编码时数字与空格的特殊处理：Llama SPM 数字按字符切、合并规则等价于 llama.cpp）

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| SPM/BPE 合并顺序细节与 llama.cpp 不一致导致 +1 token | High | golden 对拍（20 提示词 ×2 模型）作为发布门禁；golden 文件纳入仓库（< 200KB） |
| GGUF tokenizer 键名变体（`tokenizer.ggml.add_bos` 等） | Medium | 严格模式：未知必需键报错，可选键按标准默认；badgen 记录 |
| 增量 UTF-8 边界（CJK/emoji 跨 chunk） | Medium | fuzz 10k 随机字节 + 专测"任意切分等价性" |
