// llama_referee.cpp — 014 L3 parity referee driver (golden chain: llama.cpp
// libllama, pinned f280b2698, CPU build).
//
// Produces token-level greedy ground truth for the reinfer engine parity
// harness (crates/cuda/tests/parity.rs). Protocol mirrors the engine side:
//   - plain-text completion (no chat template),
//   - add_special=true / parse_special=false tokenization (Qwen3 GGUF has
//     add_bos=false, so no BOS is inserted on either side),
//   - greedy sampling, temperature 0, first-max tie-break
//     (llama_sampler_greedy uses strict `>` — same rule as engine argmax_first),
//   - exactly N steps, no EOS stop.
//
// Usage:
//   llama-referee -m MODEL.gguf -n NTOK -t THREADS -o OUT.bin
//   prompt text is read from stdin (exact UTF-8 bytes).
//
// OUT.bin layout (little-endian):
//   u32 magic 0x50415250 ("RPAR") | u32 version 1 | u32 n_vocab
//   u32 n_prompt | u32[n_prompt] prompt token ids
//   u32 n_steps
//   per step: u32 token | f32[n_vocab] logits (the logits the token was
//   sampled from)
//
// Build (see bench/notes.md 014 parity record):
//   g++ -std=c++17 -O2 -I <llama.cpp>/include -I <llama.cpp>/ggml/include \
//       llama_referee.cpp -L <llama.cpp>/build/bin -lllama \
//       -Wl,-rpath,<llama.cpp>/build/bin -o <llama.cpp>/build/bin/llama-referee

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void write_u32(FILE * f, uint32_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

static void write_f32s(FILE * f, const float * p, size_t n) {
    fwrite(p, sizeof(float), n, f);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string out_path;
    int n_steps = 64;
    int threads = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s -m MODEL.gguf -n NTOK -t THREADS -o OUT.bin\n", argv[0]);
            return 2;
        }
    }
    if (model_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL.gguf -n NTOK -t THREADS -o OUT.bin\n", argv[0]);
        return 2;
    }

    // Prompt from stdin (exact bytes — includes newlines / NBSP in the set).
    std::string prompt;
    {
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) {
            prompt.append(buf, n);
        }
    }

    // Mirror llama-simple: load dynamic backends, then the model.
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU referee
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load model %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize the prompt: add_special=true (BOS only if configured — Qwen3:
    // none), parse_special=false (mirror engine encode(prompt, false)).
    std::vector<llama_token> pids;
    {
        const int n0 = -llama_tokenize(
            vocab, prompt.data(), (int32_t) prompt.size(), nullptr, 0, true, false);
        if (n0 < 0) {
            fprintf(stderr, "error: tokenize size failed\n");
            return 1;
        }
        pids.resize(n0);
        const int n1 = llama_tokenize(
            vocab, prompt.data(), (int32_t) prompt.size(),
            pids.data(), (int32_t) pids.size(), true, false);
        if (n1 != n0) {
            fprintf(stderr, "error: tokenize failed\n");
            return 1;
        }
    }

    llama_context_params cpar = llama_context_default_params();
    cpar.n_ctx = (uint32_t) (pids.size() + (size_t) n_steps + 16);
    cpar.n_batch = (uint32_t) std::max<size_t>(pids.size(), 256);
    cpar.n_threads = threads;
    cpar.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cpar);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create llama_context\n");
        return 1;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    FILE * out = fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        fprintf(stderr, "error: cannot open %s\n", out_path.c_str());
        return 1;
    }
    write_u32(out, 0x50415250); // "RPAR"
    write_u32(out, 1);          // version
    write_u32(out, (uint32_t) n_vocab);
    write_u32(out, (uint32_t) pids.size());
    for (llama_token t : pids) {
        write_u32(out, (uint32_t) t);
    }
    write_u32(out, (uint32_t) n_steps);

    // One token_data per vocab entry; greedy only reads logit, sets selected.
    std::vector<llama_token_data> tdata((size_t) n_vocab);
    llama_token_data_array cur_p = { tdata.data(), (size_t) n_vocab, -1, false };

    // Prefill: whole prompt in one batch; the resulting logits (last prompt
    // position) predict generated token #1.
    llama_batch batch = llama_batch_get_one(pids.data(), (int32_t) pids.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: prefill decode failed\n");
        return 1;
    }

    for (int i = 0; i < n_steps; i++) {
        float * logits = llama_get_logits(ctx);
        for (int32_t k = 0; k < n_vocab; k++) {
            tdata[(size_t) k].id = (llama_token) k;
            tdata[(size_t) k].logit = logits[k];
            tdata[(size_t) k].p = 0.0f;
        }
        cur_p.selected = -1;
        llama_sampler_apply(smpl, &cur_p);
        llama_token next = cur_p.data[cur_p.selected].id;

        write_u32(out, (uint32_t) next);
        write_f32s(out, logits, (size_t) n_vocab);
        printf("step %d tok %d\n", i, next);
        fflush(stdout);

        llama_batch b1 = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, b1) != 0) {
            fprintf(stderr, "error: decode step %d failed\n", i);
            return 1;
        }
    }

    fclose(out);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
