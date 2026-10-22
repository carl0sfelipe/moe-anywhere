| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU              |           tg128 |         49.39 ± 0.13 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         45.93 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         43.20 ± 0.03 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         53.67 ± 0.25 |

build: 0a1ef309c (11796)
