| model                          |       size |     params | backend    | ngl | threads | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------------- | --------------: | -------------------: |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       4 | exps=CPU              |           tg128 |         20.31 ± 0.66 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       6 | exps=CPU              |           tg128 |         19.32 ± 1.49 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |       8 | exps=CPU              |           tg128 |         14.37 ± 2.45 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |      12 | exps=CPU              |           tg128 |         15.57 ± 0.30 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 |      16 | exps=CPU              |           tg128 |         14.26 ± 1.71 |

build: 36cbf41 (1)
