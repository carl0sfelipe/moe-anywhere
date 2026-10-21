| model                          |       size |     params | backend    | ngl | threads | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       4 | exps=CPU              |           tg128 |         48.47 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       6 | exps=CPU              |           tg128 |         49.31 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       8 | exps=CPU              |           tg128 |         49.46 ± 0.08 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      12 | exps=CPU              |           tg128 |         47.77 ± 0.24 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      16 | exps=CPU              |           tg128 |         47.30 ± 0.44 |

build: 0a1ef309c (11796)
