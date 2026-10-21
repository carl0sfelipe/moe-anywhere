// moe-trace: record MoE routing (top-k experts + weights per layer per token) from llama.cpp.
//
// Usage: moe-trace --prompts prompts.jsonl --out trace.tsv [-n 256] [llama.cpp args: -m -ngl -ot ...]
//
// prompts.jsonl: one {"prompt_id": int, "task": str, "prompt": str} per line.
// Output TSV, one row per (token, layer):
//   prompt_id  task  token_pos  phase(prefill|decode)  layer  e0,e1,..  w0,w1,..
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

struct layer_capture {
    std::vector<int32_t> experts;  // [n_tokens * k]
    std::vector<float>   weights;  // [n_tokens * k]
    int64_t k = 0, n_tokens = 0;
    bool has_norm = false;
};

struct trace_state {
    std::map<int, layer_capture> layers;
};

static int layer_of(const char * name, const char * prefix) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    return atoi(name + n);
}

static void copy_tensor(const ggml_tensor * t, void * dst, size_t nbytes) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(dst, t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, dst, 0, nbytes);
    }
}

// Copies a [ne0, ne1] tensor row by row so strided views (top-k of an argsort) work too.
static void copy_rows(const ggml_tensor * t, void * dst) {
    const size_t row = t->ne[0] * ggml_type_size(t->type);
    for (int64_t i = 0; i < t->ne[1]; ++i) {
        char * d = (char *) dst + i * row;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(d, (const char *) t->data + i * t->nb[1], row);
        } else {
            ggml_backend_tensor_get(t, d, i * t->nb[1], row);
        }
    }
}

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;
    int il_topk = layer_of(t->name, "ffn_moe_topk-");
    int il_norm = layer_of(t->name, "ffn_moe_weights_norm-");
    int il_raw  = layer_of(t->name, "ffn_moe_weights-");
    if (ask) {
        return il_topk >= 0 || il_norm >= 0 || il_raw >= 0;
    }
    if (il_topk >= 0) {
        if (t->ne[2] != 1 || strstr(t->name, "(")) {
            return true;
        }
        auto & c = st->layers[il_topk];
        c.k = t->ne[0];
        c.n_tokens = t->ne[1];
        c.experts.resize(c.k * c.n_tokens);
        copy_rows(t, c.experts.data());
    } else if (il_norm >= 0 || il_raw >= 0) {
        // shapes: [k, n_tokens] (norm) or [1, k, n_tokens] (raw); both are k*n_tokens floats
        auto & c = st->layers[il_norm >= 0 ? il_norm : il_raw];
        if (il_raw >= 0 && c.has_norm) {
            return true;
        }
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || strstr(t->name, "(")) {
            return true;
        }
        c.weights.resize(ggml_nelements(t));
        copy_tensor(t, c.weights.data(), c.weights.size() * sizeof(float));
        c.has_norm = il_norm >= 0;
    }
    return true;
}

static void flush(trace_state & st, FILE * out, int prompt_id, const std::string & task, int pos0, const char * phase) {
    if (st.layers.empty()) {
        return;
    }
    int64_t n_tokens = 0;
    for (auto & [il, c] : st.layers) {
        n_tokens = std::max(n_tokens, c.n_tokens);
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (auto & [il, c] : st.layers) {
            // the last layer only computes output rows (the batch's last token in prefill)
            int64_t r = i - (n_tokens - c.n_tokens);
            if (r < 0) {
                continue;
            }
            fprintf(out, "%d\t%s\t%lld\t%s\t%d\t", prompt_id, task.c_str(), (long long) (pos0 + i), phase, il);
            for (int64_t j = 0; j < c.k; ++j) {
                fprintf(out, j ? ",%d" : "%d", c.experts[r * c.k + j]);
            }
            fputc('\t', out);
            for (int64_t j = 0; j < c.k; ++j) {
                float w = c.weights.size() == (size_t) (c.k * c.n_tokens) ? c.weights[r * c.k + j] : 0.0f;
                fprintf(out, j ? ",%.5f" : "%.5f", w);
            }
            fputc('\n', out);
        }
    }
    st.layers.clear();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string prompts_path, out_path;
    int n_gen = 256;
    bool chatml = true;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompts" && i + 1 < argc) {
            prompts_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (a == "--raw") {
            chatml = false;
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (prompts_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s --prompts prompts.jsonl --out trace.tsv [-n 256] [--raw] [llama.cpp args]\n", argv[0]);
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    trace_state st;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler * smpl = llama_sampler_init_greedy();

    FILE * out = fopen(out_path.c_str(), "w");
    fprintf(out, "prompt_id\ttask\ttoken_pos\tphase\tlayer\texperts\tweights\n");

    std::ifstream in(prompts_path);
    std::string line;
    int n_prompts = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto j = nlohmann::json::parse(line);
        int prompt_id = j.value("prompt_id", n_prompts);
        std::string task = j.value("task", "unknown");
        std::string text = j.at("prompt").get<std::string>();
        if (chatml) {
            text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        st.layers.clear();

        std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
        int pos = 0;
        // prefill in chunks of n_batch so every capture covers one decode call
        for (size_t off = 0; off < tokens.size(); off += params.n_batch) {
            int n = (int) std::min<size_t>(params.n_batch, tokens.size() - off);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off, n))) {
                LOG_ERR("prefill failed on prompt %d\n", prompt_id);
                return 1;
            }
            flush(st, out, prompt_id, task, pos, "prefill");
            pos += n;
        }
        for (int i = 0; i < n_gen; ++i) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
                LOG_ERR("decode failed on prompt %d\n", prompt_id);
                return 1;
            }
            flush(st, out, prompt_id, task, pos, "decode");
            pos += 1;
        }
        fflush(out);
        ++n_prompts;
        LOG_INF("prompt %d (%s): %d tokens traced\n", prompt_id, task.c_str(), pos);
    }
    fclose(out);
    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
