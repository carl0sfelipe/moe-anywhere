| model                          |       size |     params | backend    | ngl | threads | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       4 | exps=CPU              |           tg128 |         39.36 ± 1.80 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       6 | exps=CPU              |           tg128 |         26.69 ± 4.83 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       8 | exps=CPU              |           tg128 |         29.59 ± 0.85 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      12 | exps=CPU              |           tg128 |         26.84 ± 1.25 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      16 | exps=CPU              |           tg128 |         25.83 ± 1.64 |

build: 36cbf41 (1)
