| model                          |       size |     params | backend    | ngl | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------------- | --------------: | -------------------: |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU              |           tg128 |         29.70 ± 0.07 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU;attn_norm=CPU |           tg128 |         28.06 ± 0.05 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | exps=CPU;attn_norm=CPU;post_attention_norm=CPU |           tg128 |         28.09 ± 0.06 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | CUDA       |  99 | blk\.[0-7]\.ffn_.*_exps=CUDA0;exps=CPU |           tg128 |         30.38 ± 0.03 |

build: 36cbf415f (11824)
