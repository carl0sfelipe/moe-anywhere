# S2 — tools/model_facts.py e a tabela do Sage-A1.5 em research/models.md

Repo: /home/carl0sfelipe/src/moe-anywhere. Depende de S1 (research/models.md existe).

## O que fazer

1. Criar `tools/model_facts.py` (stdlib apenas) com:
   - `facts(config: dict) -> dict`: aceita o config HF completo (usa `config["text_config"]`
     se existir) ou o text_config direto. Chaves: `num_layers` (num_hidden_layers),
     `n_experts` (num_experts), `top_k` (num_experts_per_tok), `hidden_size`,
     `moe_intermediate_size`, `shared_expert_intermediate_size`,
     `linear_attention_layers` e `full_attention_layers` (contagem de "linear_attention"
     e "full_attention" em `layer_types`).
   - `est_size_gb(params_billion: float, bpw: float) -> float` = params_billion * bpw / 8.
   - `to_markdown(facts: dict) -> str`: tabela `| key | value |`, cabeçalho
     `| key | value |` + `|---|---|`, uma linha `| <chave> | <valor> |` por chave na ordem acima.
   - CLI: `python3 tools/model_facts.py <config.json>` imprime `to_markdown(facts(...))`.
2. Rodar a CLI em `data/sage-a1.5-config.json` e colar a saída em `research/models.md`
   no lugar da linha `<!-- model_facts table -->` (seção `## Sage-A1.5`).

## Regras

Não invente número: todo valor sai do config. Não edite os testes nem
data/sage-a1.5-config.json. Não use declare const nem valores hardcoded para passar
no teste.

## Dados verificados

- data/sage-a1.5-config.json é o config.json real de InternScience/Sage-A1.5
  (chaves em `text_config`; `layer_types` lista 40 itens).

## Verificação

VERIFICACAO: cd /home/carl0sfelipe/src/moe-anywhere && python3 tools/model_facts.py data/sage-a1.5-config.json | grep -q "n_experts | 256"

## Oráculo

- comando: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python -m pytest -q tests/test_model_facts.py && .venv/bin/python tests/check_docs.py
- exit esperado: 0

## Barra

tests/test_model_facts.py.
