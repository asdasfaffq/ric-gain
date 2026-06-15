# RIC-Gain (partial release)

Selected core code and results for **RIC-MG** — a measured-gain refresh policy that
decides which documents to re-embed first during dense-retrieval index migration
("measure, don't predict": measure recoverable recall gain on a bounded per-query
candidate pool, aggregate via a submodular one-pass ranking).

**This is a partial release for timestamping authorship/priority.** It contains the
core method (`experiments/ric_bridge.py`) and selected results; the full experiment
pipeline and data will be released upon publication.

- `experiments/ric_bridge.py` — core: measured-gain attribution, submodular one-pass selection, mixed old/new index serving.
- `results/` — selected result tables (multi-metric comparison; main significance test).

All rights reserved by the authors. Please cite this work and contact the authors before reuse.
