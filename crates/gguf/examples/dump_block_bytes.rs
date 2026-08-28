//! Raw Q8_0 block-byte extractor — 014 T2 golden input producer.
//!
//! Usage: `cargo run -p reinfer-gguf --example dump_block_bytes -- <path.gguf> <tensor-name> <num-blocks>`
//! Prints `num_blocks` raw 34-byte Q8_0 blocks (from the tensor's start)
//! as hex to stdout: one line per block, 68 capital-hex digits.
//! Paired with `scripts/golden/q8_0_refdump.c` (referee dequantize_row_q8_0)
//! inside `scripts/golden/gen_q8_0_golden.sh` (014 T2 gate).

use reinfer_gguf::GgufReader;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(name), Some(nblk)) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: dump_block_bytes <path.gguf> <tensor-name> <num-blocks>");
        return ExitCode::FAILURE;
    };
    let nblk: usize = match nblk.parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad block count: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reader = match GgufReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("open failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let Some(t) = reader.tensor(&name) else {
        eprintln!("tensor not found: {name}");
        return ExitCode::FAILURE;
    };
    let data = match reader.tensor_data(t) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("tensor data failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let take = nblk * 34;
    let take = take.min(data.len()).min(68 * 1024); // cap output (35 MiB worst)
    for chunk in data[..take].chunks(34) {
        if chunk.len() < 34 {
            break;
        }
        let mut line = String::with_capacity(69);
        for b in chunk {
            line.push_str(&format!("{b:02X}"));
        }
        println!("{line}");
    }
    ExitCode::SUCCESS
}
