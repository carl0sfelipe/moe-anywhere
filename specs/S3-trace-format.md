# S3 — Formato de trace de roteamento + gerador sintético

Repo: /home/carl0sfelipe/src/moe-anywhere. Python com numpy e pyarrow (já no .venv).

## O que fazer

Criar `moe_anywhere/trace.py` com:

- `COLUMNS = ("prompt_id", "task", "token_pos", "layer", "experts", "weights")`.
- Um trace é um `dict` com exatamente essas chaves, N linhas, uma linha por (token, camada),
  em ordem token-major (linhas i*L..i*L+L-1 = token i, camadas 0..L-1):
  `prompt_id` np.int32 (N,), `task` list[str] (N), `token_pos` np.int32 (N,),
  `layer` np.int16 (N,), `experts` np.int16 (N, k) sem repetição na linha,
  `weights` np.float16 (N, k), não negativos, soma ≈ 1 por linha.
- `synth_trace(n_tokens, n_layers, n_experts, k, seed=0, zipf_s=1.1) -> dict`:
  determinístico por `seed` (numpy Generator). Popularidade desigual (Zipf com
  expoente zipf_s por camada) e correlação entre camadas: para cada token, parte
  dos experts da camada l+1 é herdada da camada l (ex.: cada expert herdado com
  prob. 0.5, o resto amostrado pela Zipf sem repetição). `task` = "synthetic",
  `prompt_id` = 0.
- `write_trace(trace, path)` grava Parquet (experts/weights como listas de tamanho fixo
  ou colunas por posição) e `read_trace(path) -> dict` devolve os mesmos dtypes e shapes.

## Regras

Não invente dependência nova além de numpy/pyarrow. Não edite tests/test_trace.py.
Não use declare const nem dados fixos para passar no teste.

## Dados verificados

- O schema acima é o que o Experimento 1 vai usar para traces reais do Sage-A1.5
  (40 camadas, 256 experts, top-8).

## Verificação

VERIFICACAO: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python -m pytest -q tests/test_trace.py -k shapes

## Oráculo

- comando: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python -m pytest -q tests/test_trace.py
- exit esperado: 0

## Barra

tests/test_trace.py.
