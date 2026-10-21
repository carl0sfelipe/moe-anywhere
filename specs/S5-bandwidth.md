# S5 — Calculadora de tok/s limitado por banda

Repo: /home/carl0sfelipe/src/moe-anywhere. Stdlib apenas.

## O que fazer

Criar `moe_anywhere/bandwidth.py` com:

- `PRESETS: dict[str, float]` em GB/s, exatamente:
  pcie3_x4=3.5, pcie3_x16=14.0, ddr4_dual=35.0, ddr4_quad=60.0, nvme_gen3=3.0,
  gbe1=0.117, ib40=4.0.
- `@dataclass Tier(name: str, bandwidth_gbs: float)`.
- `tokens_per_sec(bytes_per_token: dict[str, float], tiers: dict[str, Tier], hops=0, hop_latency_ms=0.0) -> float`:
  tempo = soma(bytes / (bandwidth_gbs * 1e9)) + hops * hop_latency_ms / 1000; retorna 1/tempo.
  Tier ausente em `tiers` → KeyError. Tempo zero (nada a fazer) → ValueError.
- CLI `python -m moe_anywhere.bandwidth --bytes <preset>=<bytes> [--bytes ...] [--hops N --hop-latency-ms X]`
  monta tiers a partir de PRESETS e imprime `tok/s: <valor com 2 casas>`.
- Adicionar em `research/bandwidth.md` (seção `## Tiers`) uma tabela com os PRESETS.

## Regras

Não invente preset além dos sete listados nem altere os valores. Não edite
tests/test_bandwidth.py. Não use declare const.

## Dados verificados

Casos calculados à mão (estão no teste):
- 6e9 bytes em ddr4_quad (60 GB/s) → 0,1 s → 10 tok/s.
- 3e9 @ 60 + 0,3e9 @ 3 → 0,05 + 0,1 = 0,15 s → 6,67 tok/s.
- 1e9 @ 100 + 3 saltos × 5 ms → 0,025 s → 40 tok/s.

## Verificação

VERIFICACAO: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python -m moe_anywhere.bandwidth --bytes ddr4_quad=6e9 | grep -q "10.00"

## Oráculo

- comando: cd /home/carl0sfelipe/src/moe-anywhere && .venv/bin/python -m pytest -q tests/test_bandwidth.py
- exit esperado: 0

## Barra

tests/test_bandwidth.py.
