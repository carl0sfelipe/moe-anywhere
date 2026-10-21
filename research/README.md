# Research

## Map

- [literature.md](literature.md) — papers on expert prefetching, offloading and placement.
- [prior-art.md](prior-art.md) — engines and projects that already do part of this, and the gap.
- [models.md](models.md) — the models we test, with facts taken from their configs.
- [bandwidth.md](bandwidth.md) — memory and link tiers, and how we estimate bandwidth-bound tok/s.
- [experiments.md](experiments.md) — planned and finished experiments.
- [open-questions.md](open-questions.md) — what we do not know yet.

## How to contribute

- **Add a paper:** one `###` entry in literature.md, under the right section, with a one-sentence summary and a link you have checked.
- **Add a predictor:** implement the predictor interface in `moe_anywhere/predictors/` and report its recall on the shared traces.
- **Submit a run:** add a row to the README results table with model, quant, hardware, technique and tok/s, and submit the same run to the bestmodel.run pool.
