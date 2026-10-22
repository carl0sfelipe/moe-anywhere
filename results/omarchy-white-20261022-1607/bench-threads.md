| model                          |       size |     params | backend    | ngl | threads | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------------- | --------------: | -------------------: |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       4 | exps=CPU              |           tg128 |         21.08 ± 0.56 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       6 | exps=CPU              |           tg128 |         26.47 ± 0.06 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       8 | exps=CPU              |           tg128 |         29.76 ± 0.04 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |      12 | exps=CPU              |           tg128 |         25.43 ± 0.03 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |      16 | exps=CPU              |           tg128 |         28.63 ± 0.08 |

build: 36cbf415f (11824)
