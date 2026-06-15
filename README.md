# RIC-Gain (partial release)

Selected core code and results for **RIC-MG** — a measured-gain refresh policy that
decides which documents to re-embed first during dense-retrieval index migration
("measure, don't predict": measure recoverable recall gain on a bounded per-query
candidate pool, aggregate via a submodular one-pass ranking).

**This is a partial release for timestamping authorship/priority.** It contains the
core method (`experiments/ric_bridge.py`) and selected results; the full experiment
pipeline and data will be released upon publication.

- `experiments/ric_bridge.py` — core: measured-gain attribution, submodular one-pass selection, mixed old/new index serving.
- `results/multi_metric.csv` — multi-metric comparison (Recall@10 / nDCG@10 / Recall@100) over the four core settings; corresponds to the paper's **multi-metric / "Beyond Recall@10" diverse-BEIR table**.
- `results/scifact_rank_tests.csv` — RIC-MG vs nine refresh-ordering baselines on SciFact (MiniLM->BGE), paired Wilcoxon with Holm correction; corresponds to the paper's **main ranking results table**. RIC-MG (`ric_mean = 0.733`) is rank-first and Holm-significant over every baseline (e.g. vs the learned scheduler, delta = +0.053, Holm p = 6.2e-7).

All rights reserved by the authors. Please cite this work and contact the authors before reuse.
