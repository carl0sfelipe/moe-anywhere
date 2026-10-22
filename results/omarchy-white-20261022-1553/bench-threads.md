| model                          |       size |     params | backend    | ngl | threads | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       4 | exps=CPU              |           tg128 |         48.50 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       6 | exps=CPU              |           tg128 |         49.05 ± 0.12 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |       8 | exps=CPU              |           tg128 |         48.97 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      12 | exps=CPU              |           tg128 |         44.56 ± 0.11 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | CUDA       |  99 |      16 | exps=CPU              |           tg128 |         40.83 ± 0.13 |

build: 0a1ef309c (11796)
