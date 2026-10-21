# S1 — Esqueleto de docs do moe-anywhere (README + research/)

Repo: /home/carl0sfelipe/src/moe-anywhere. Idioma dos docs: inglês (público global).
moe-anywhere é o lab open source de offload/prefetch de experts MoE em GPUs de
consumidor, patrocinado pelo bestmodel.run (compatibility engine de IA local
baseado em runs testados pela comunidade). Tagline exata:
"Run giant MoE models on the GPU you already have."

## O que fazer

Criar estes arquivos, cada um com EXATAMENTE os cabeçalhos `##` listados
(pode ter texto e sub-cabeçalhos `###` dentro):

1. `README.md` — título `# moe-anywhere`, a tagline, e:
   `## Sponsor` (texto: "moe-anywhere is sponsored by [bestmodel.run](https://bestmodel.run) —
   see what actually runs on your machine. Every result here is submitted to the
   bestmodel.run pool."), `## Problem`, `## Results` (tabela vazia com colunas
   Model | Quant | Hardware | Technique | tok/s | Date), `## Roadmap` (E0–E3 da
   seção Dados verificados), `## Contributing` (aponta para research/README.md),
   `## License` (Apache-2.0, link https://www.apache.org/licenses/LICENSE-2.0).
2. `research/README.md` — `## Map` (uma linha por arquivo de research/),
   `## How to contribute` (adicionar paper, adicionar preditor, enviar run).
3. `research/literature.md` — `## Prefetching`, `## Offloading`, `## Placement`;
   uma entrada `###` por paper da lista "Papers" abaixo, na seção indicada,
   com a frase de resumo fornecida.
4. `research/prior-art.md` — `## Engines`, `## Prefetch projects`, `## Gap`;
   uma entrada `###` por projeto da lista "Projetos" abaixo, com a frase fornecida.
5. `research/models.md` — `## Sage-A1.5` e `## DeepSeek-V4.1-Flash` com as frases
   abaixo (a tabela do Sage é responsabilidade da S2 — deixe a linha
   "<!-- model_facts table -->").
6. `research/bandwidth.md` — `## Tiers`, `## Method` (texto curto: tok/s limitado
   por banda = 1 / soma(bytes_por_token_no_tier / banda_do_tier) + latência por salto).
7. `research/experiments.md` — `## E0`, `## E1`, `## E2`, `## E3` (descrições abaixo,
   campo "Status: planned").
8. `research/open-questions.md` — `## Questions` com as perguntas abaixo.

## Regras

Não invente número, link, autor, ano ou resultado além dos listados aqui. URL só
das que aparecem neste spec (o oráculo recusa qualquer outra). Paper sem link aqui
fica sem link. Não use declare const nem nenhum workaround para burlar o
oráculo; não edite tests/check_docs.py.

## Dados verificados

Papers (seção: frase):
- Prefetching: Pre-gated MoE — uses the current layer's router to predict next-layer experts.
- Prefetching: MoE-Infinity — activation history drives expert prefetch and caching.
- Prefetching: ProMoE — predictive expert prefetch with caching.
- Prefetching: AdapMoE — adaptive gating with prefetch and caching.
- Prefetching: SeqMoE — predictive, graph-compatible offloading; prefill prefetch by profiled frequency. Link: https://arxiv.org/pdf/2609.12978
- Prefetching: SpecPrefetch — parameter-efficient expert prefetching for sparse MoE. Link: https://arxiv.org/pdf/2607.24787
- Offloading: HOBBIT — loads less important experts in lower precision to save bandwidth.
- Offloading: Fiddler — computes offloaded experts on the CPU instead of copying them to the GPU.
- Offloading: MoE-Lightning — CPU-GPU pipelined MoE inference on memory-constrained GPUs.
- Placement: EPLB — DeepSeek's expert-parallel load balancer: counts expert use, replicates hot experts, reshuffles placement.

Projetos (seção: frase: link):
- Engines: DeepInfer — DeepSeek's open-source engine (Apache-2.0); SM100 / Ascend 950 only; includes EPLB: https://github.com/deepseek-ai/DeepInfer
- Engines: cachedMoE — Vulkan/Slang engine for DeepSeek-V4.1-Flash; LRU expert cache streamed from NVMe; no prediction; bridged into DeepInfer by PR 142: https://github.com/acupof-ai/cachedMoE e https://github.com/deepseek-ai/DeepInfer/pull/142
- Engines: Flare — NVIDIA local inference server, SM120/SM121 only; KV/prefix-cache offload but no expert offload: https://github.com/NVIDIA/flare-local-inference
- Engines: llama.cpp — `-ot exps=CPU` keeps routed experts in RAM and computes them on CPU: https://github.com/ggml-org/llama.cpp
- Engines: KTransformers — CPU/GPU hybrid MoE inference (sem link)
- Engines: Unsloth — dynamic low-bit GGUF quants for large MoEs (sem link)
- Prefetch projects: llama-moe-cache — expert cache + predictive prefetch for llama.cpp: https://github.com/ongunm/llama-moe-cache
- Prefetch projects: moe-prefetch — learned per-layer prefetch heads for SSD-streamed experts: https://github.com/DimitriChrysafis/moe-prefetch
- Prefetch projects: vLLM — RFC for GPU expert cache + async pipeline: https://github.com/vllm-project/vllm/issues/38256
- Prefetch projects: FreeToken — router lookahead prefetch of next-layer experts: https://github.com/FlashML-org/FreeToken/issues/677
- Prefetch projects: moe-offload — SSD + per-layer LRU expert cache for Apple Silicon: https://github.com/Jakevin/moe-offload
- Gap (texto): no one does predictive expert prefetch distributed across multiple consumer GPUs and nodes, and there is no open dataset of routing traces.

Modelos:
- Sage-A1.5: InternScience fine-tune of Qwen3.6-35B-A3B, Apache-2.0, Q4_K_M GGUF 21.7 GB: https://huggingface.co/InternScience/Sage-A1.5
- DeepSeek-V4.1-Flash: reported 552B total parameters, about 16B active per decoded token (community listings, unconfirmed).

Experimentos:
- E0 — baseline: llama.cpp with `-ot exps=CPU`, Sage-A1.5 Q4_K_M on an RTX 5050 8 GB.
- E1 — routing traces: record (layer, token, top-k experts) and publish as an open dataset.
- E2 — predictors offline on CPU: random, LRU, frequency, previous-layer copy, pre-gating; metric recall@k at horizons 1/2/3.
- E3 — online VRAM expert cache + async prefetch vs E0.

Perguntas: How predictable is layer L+1 from layer L on real traces? How much VRAM cache gives 90% hit rate? Does expert popularity shift by task? Can prefetch hide inter-node latency on a consumer cluster?

## Verificação

VERIFICACAO: cd /home/carl0sfelipe/src/moe-anywhere && python3 tests/check_docs.py

## Oráculo

- comando: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python tests/check_docs.py
- exit esperado: 0 — todos os arquivos, cabeçalhos, nomes e o link do patrocinador presentes, nenhuma URL fora da allowlist.

## Barra

tests/check_docs.py (gate) e este spec como fonte única dos fatos.
