| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU              |           tg128 |         49.39 ± 0.10 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         46.88 ± 0.08 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         44.93 ± 0.08 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         56.03 ± 0.39 |

build: 36cbf415f (11824)
