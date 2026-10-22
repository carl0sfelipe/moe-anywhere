# Contribuição pro llama.cpp: fatos pra você escrever (2026-10-22)

Isto NÃO é rascunho de comentário. É a ficha de fatos. O texto público tem que ser seu.

## A regra do llama.cpp (AGENTS.md + CONTRIBUTING.md)

- Proibido: descrição de PR, commit, comentário ou resposta a revisor escrito por IA. Pena: PR fechado, possível ban.
- Proibido: agente rodar `git push` / `gh pr create` no repo deles por você.
- Feature nova começa por issue, não por PR. PR duplicado de outro aberto é fechado.
- Código feito com IA pode, se você explica cada linha sem IA e declara o uso de IA no template.
- Commit feito por agente a pedido seu: `Assisted-by:`, não `Co-authored-by:`. O patch atual usa `Co-Authored-By: Claude` e precisa trocar.
- Em #27861, voidpush já reclamou de PR "vibecoded" que o autor não defende.

## Estado dos threads (checado hoje)

| Thread | Estado | Por que importa |
|---|---|---|
| #29887 `--moe-cache-mib` (am17an) | merged 07/10 | é a base do seu patch; o dono é o mantenedor |
| #30112 multi-GPU (am17an) | merged 08/10 | seu patch é single-GPU |
| #27861 hybrid draft (csantiago78) | aberto, autor sumido desde 28/08, 61 comentários | mesma ideia que a sua (miss na CPU) |
| #30163 `--moe-cache-mib` por device | aberto | não é o seu tema |
| #29949 cache GPU-resident dentro do grafo | aberto, mexido hoje | vizinho, ler antes |
| #20757 two-tier cache | fechado | só referência |

## Seus fatos novos (de research/experiments.md)

RTX 5050 8 GB, PCIe 4.0 x8, Sage-A1.5 Q4_K_M (qwen35moe, 256 experts, top-8), `-c 8192 -fit off --load-mode none`:

- Sem cache 45.5 / upstream 5 GB 50.9 / hybrid 5 GB 55.1 tok/s.
- Cache de 2 GB: upstream 30.4 (pior que sem cache), hybrid 44.2.
- Prefill de prompt curto: upstream fica 2.7x mais lento; hybrid mantém ~90 tok/s.
- Custo do upstream medido: 10.9 ms + 0.098 ms por MiB enviado, sem overlap. Só ganha de "experts na CPU" acima de ~4.5 GB.
- Ryzen 7 7800X3D + mesma 5050: sem cache 48.4 / upstream 56.7 / hybrid 61.5.
- KLD contra sem cache: hybrid 0.0062, upstream 0.0060.
- Duas armadilhas que o thread #27861 também viu: slot zero compartilhado dá id repetido e derruba o MMQ do CUDA (precisa de `n_expert_used` slots zero distintos); um memcpy por expert trava o copy engine (chunks de 64 KiB, um por vez).
- Extra que ninguém tem: traces abertos no HF e um timeline que replica tok/s com erro de 2%.

## Estado do patch

- Rebaseado em 2026-10-22 no master `36cbf415f`: `patches/llama.cpp-36cbf415f-moe-cache-hybrid.patch` (aplica limpo, checado).
- Único conflito: `src/llama-graph.cpp`. O patch reindenta o bloco do MoE, e o master tinha adicionado `LLM_ARCH_FLINT` na condição do `build_amax_src1_f16`. Fiquei com o lado do patch e levei o FLINT junto. A mudança de Hadamard do master (`cur_mm` em `build_lora_mm_id`) entra sozinha, porque o hybrid passa todas as matmuls de expert por `build_lora_mm_id`.
- O trailer agora é `Assisted-by: Claude Opus 5.5` (regra do llama.cpp). A mensagem de commit ainda precisa ser reescrita por você antes do PR.
- Compilou só em CPU (`llama-completion`, `test-backend-ops`, sem warning nos arquivos do patch), numa máquina sem CUDA.
- **CUDA validado em 2026-10-22 (E3f)**, RTX 3090 + Ryzen 5 3500X: compila sem mudança, roda, KLD igual ao upstream no Sage-A1.5 e no Nemotron 3.5 Bolt (arquitetura nova: experts por camada, sem gate, LatentMoE). **Mas nessa máquina o hybrid ficou 2x mais lento que o upstream** e instável. Falta saber se é a máquina (PCIe 3.0 x16 rápido, CPU fraca, containers de prod) ou o rebase: rodar 0a1ef30 e 36cbf415f no mesmo dia no 7800X3D. (Resolvido em E3g.)
- **Rebase não regrediu (E3g, 2026-10-22, 7800X3D + RTX 5050)**: no mesmo dia, hyb5120 deu 62.8 tok/s em 36cbf415f e 63.0 em 0a1ef30; upstream e no cache idênticos. A perda no omarchy é da máquina (PCIe 3.0 x16 rápido contra CPU fraca e compartilhada), não do rebase.
- Bolt no 5050 (8 GB, só cabe 2 GB de cache, 12% dos experts): hybrid empata com no cache (29.7 vs 29.0), upstream perde (28.3). KLD do Bolt a 2048 MiB: hybrid 0.00161 vs upstream 0.00144 (+12%, barras de erro se sobrepõem, mas acima da régua de 10%). Medir de novo antes do PR.
- Contenção: cópias H2D de 64 MiB em outro processo derrubam experts na CPU em 58%, cópias de 1 MiB em 18%, um memcpy RAM-RAM em 21%. O efeito grande é fila no link/copy engine, não banda de RAM.
- Tamanho: 10 arquivos, +507 / -178. Revisão honesta pela regra deles: 1h por 200-400 linhas.
- Limite: só ubatch de 1 token, só 1 GPU, só CUDA testado.

## Ordem sugerida

1. Comentário em #29887 com os números (é o thread do mantenedor que vai decidir).
2. Issue "Feature Request: hybrid miss path for --moe-cache-mib" com os números e link pro patch no moe-anywhere.
3. PR só se am17an topar na issue. Rebase no master, `Assisted-by:`, declaração de IA no template, texto seu.
4. #27861: comentário curto creditando e com os dados, já que é a mesma ideia.
