# S4 — Preditores de experts + avaliador

Repo: /home/carl0sfelipe/src/moe-anywhere-surf. Depende de S3 (moe_anywhere/trace.py).

## O que fazer

1. `moe_anywhere/predictors/__init__.py` exporta `Random`, `LRU`, `Frequency`, `PrevLayerCopy`.
   Interface comum:
   - `observe(token_pos: int, layer: int, experts) -> None` — chamado em ordem do trace.
   - `predict(layer: int, horizon: int, k: int) -> set[int]` — prevê os experts da camada
     `layer + horizon` do token atual, logo após observar `layer`. No máximo k itens.
   - `Random(n_experts, seed=0)`: k experts uniformes sem repetição.
   - `LRU()`: os k experts usados mais recentemente na camada alvo (qualquer token anterior).
   - `Frequency(n_experts)`: os k mais frequentes até agora na camada alvo.
   - `PrevLayerCopy()`: os experts que acabaram de ser observados (camada `layer`).
2. `moe_anywhere/eval_predictors.py`:
   - `evaluate(trace, factory, k, horizons=(1, 2, 3)) -> dict[int, float]`: um predictor novo
     (`factory()`), percorre o trace; após observar (token, l), para cada h com l+h < L,
     compara `predict(l, h, k)` com os experts reais de (token, l+h); recall = |∩| / k;
     média por horizonte.
   - `cache_hit_rate(trace, cache_size_per_layer, policy="lru") -> float`: um cache LRU por
     camada; cada expert selecionado conta hit se já estava no cache, senão miss e entra
     (removendo o LRU se cheio). Tamanho 0 → 0.0. Retorna hits / total.
   - CLI: `python -m moe_anywhere.eval_predictors <trace.parquet> --k K [--cache-sizes 8,32,64]`
     imprime uma linha por preditor (random, lru, frequency, prev_layer_copy) com
     `recall@1`, `recall@2`, `recall@3` e, abaixo, `hit_rate` por tamanho de cache.

## Regras

Não invente métrica nova nem atalho que olhe o futuro do trace dentro de predict.
Não edite tests/test_predictors.py. Não use declare const nem valores fixos.

## Dados verificados

- Formato do trace: ver moe_anywhere/trace.py (S3). n_experts sai de `experts.max()+1`
  quando a CLI precisar.

## Verificação

VERIFICACAO: cd /home/carl0sfelipe/src/moe-anywhere-surf && .venv/bin/python -m pytest -q tests/test_predictors.py -k cli

## Oráculo

- comando: cd /home/carl0sfelipe/src/moe-anywhere-surf && .venv/bin/python -m pytest -q tests/test_predictors.py tests/test_trace.py
- exit esperado: 0

## Barra

tests/test_predictors.py.
