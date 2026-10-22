| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU              |           tg128 |         27.26 ± 7.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         26.63 ± 2.04 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         26.31 ± 3.04 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         37.89 ± 3.93 |

build: 36cbf41 (1)
