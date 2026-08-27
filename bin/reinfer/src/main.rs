//! reinfer —— 支持 CUDA / 昇腾 CANN 的 Rust 推理引擎（server | cli | bench）
fn main() {
    // 开发测试环境：本目录 .env（gitignored；模板 .env.example）；无文件静默跳过
    let _ = dotenvy::dotenv();
    println!("reinfer {}", env!("CARGO_PKG_VERSION"));
}
