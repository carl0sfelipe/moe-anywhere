| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU              |           tg128 |         18.87 ± 1.96 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         15.09 ± 8.01 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         17.52 ± 1.36 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         17.39 ± 9.99 |

build: 36cbf41 (1)
