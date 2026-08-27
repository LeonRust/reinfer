//! 跨进程并发首发测试（012 E1 / T8 验收：两进程同 key 只编译一次）。
//!
//! 父测试 spawn 两个子进程（同一测试二进制），各自对**同一缓存目录、
//! 同一 key** 执行 `build_once`（compile 回调真实触发 nvcc 之外的假产物，
//! 仅计数）；父进程断言：两个子进程退出成功、产物一致、compile 计数器
//! 恰好 1 行（锁 + 双检生效，无重复编译、无竞争撕裂）。

use reinfer_jit::{JitCache, JitKey, KernelSource, ToolchainId};
use std::path::PathBuf;
use std::process::Command;

const ENV_CHILD: &str = "REINFER_JIT_CONCURRENT_CHILD";
const ENV_CACHE: &str = "REINFER_JIT_CONCURRENT_CACHE";
const ENV_COUNT: &str = "REINFER_JIT_CONCURRENT_COUNT";

fn key() -> (KernelSource, ToolchainId) {
    let src = KernelSource {
        name: "conc",
        src: "kernel-body",
        headers: vec![],
        flags: vec!["-gencode".into(), "arch=compute_120,code=sm_120a".into()],
        arch: "sm_120a".into(),
        toolchain_ver: "release 12.8".into(),
    };
    let tc = ToolchainId {
        ver_line: "release 12.8".into(),
        realpath: PathBuf::from("/usr/local/cuda-12.8/bin/nvcc"),
        ccbin: (PathBuf::from("/usr/bin/g++"), "g++ 12".into()),
    };
    (src, tc)
}

/// 子进程模式：build_once（compile 回调追加计数行）；普通运行 = no-op。
#[test]
fn child_build() {
    if std::env::var(ENV_CHILD).as_deref() != Ok("1") {
        return; // 仅作为子进程入口
    }
    let cache_dir = PathBuf::from(std::env::var(ENV_CACHE).expect("cache env"));
    let count_file = std::env::var(ENV_COUNT).expect("count env");
    let (src, tc) = key();
    let cache = JitCache::open(Some(cache_dir)).expect("open");
    let k = JitKey::new(&src, &tc);
    // compile 假产物：向计数文件追加一行（模拟编译开销）
    let closure_count = count_file.clone();
    let bytes = b"elf-42";
    let (_, path) = cache
        .build_once(&k, &src, || {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&closure_count)
                .and_then(|mut f| std::io::Write::write_all(&mut f, b"ran\n"))
                .expect("count");
            Ok(bytes.to_vec())
        })
        .expect("build_once");
    assert!(path.exists());
    // 二次命中：不再编译
    let (_, _) =
        cache.build_once(&k, &src, || panic!("must not compile twice in child")).expect("hit");
}

/// 父进程：两个子进程并发首发 → 编译恰一次。
#[test]
fn two_processes_compile_once() {
    let root = std::env::temp_dir().join(format!("reinfer-jit-conc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("root dir");
    let cache_dir = root.join("cache");
    let count_file = root.join("count.log");

    let exe = std::env::current_exe().expect("exe path");

    fn spawn_child(
        exe: &std::path::Path,
        cache_dir: &std::path::Path,
        count_file: &std::path::Path,
    ) -> std::process::Output {
        let name = "child_build";
        Command::new(exe)
            .arg("--exact")
            .arg(name)
            .arg("--test-threads=1")
            .arg("--nocapture")
            .env(ENV_CHILD, "1")
            .env(ENV_CACHE, cache_dir)
            .env(ENV_COUNT, count_file)
            .output()
            .expect("spawn child")
    }
    let exe1 = exe.clone();
    let cache1 = cache_dir.clone();
    let count1 = count_file.clone();
    let c1 = std::thread::spawn(move || spawn_child(&exe1, &cache1, &count1));
    let exe2 = exe.clone();
    let cache2 = cache_dir.clone();
    let count2 = count_file.clone();
    let c2 = std::thread::spawn(move || spawn_child(&exe2, &cache2, &count2));
    let o1 = c1.join().expect("c1");
    let o2 = c2.join().expect("c2");
    let child_out1 =
        format!("{}{}", String::from_utf8_lossy(&o1.stdout), String::from_utf8_lossy(&o1.stderr));
    assert!(o1.status.success(), "child1: {child_out1}");
    let child_out2 =
        format!("{}{}", String::from_utf8_lossy(&o2.stdout), String::from_utf8_lossy(&o2.stderr));
    assert!(o2.status.success(), "child2: {child_out2}");

    let runs = std::fs::read_to_string(&count_file).expect("count file");
    assert_eq!(runs.trim().lines().count(), 1, "compile count must be exactly 1");
    let _ = std::fs::remove_dir_all(&root);
}
