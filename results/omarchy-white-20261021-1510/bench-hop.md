| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU              |           tg128 |         49.37 ± 0.11 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         46.25 ± 0.39 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         44.34 ± 0.06 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         55.63 ± 0.16 |

build: 0a1ef309c (11796)
