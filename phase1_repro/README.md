# Phase 1 — SAGE smoke reproduction (Mac, CPU)

SAGE clone: `../sage` at commit `405d5f5` (unmodified). Venv: `../.venv-sage` (Python 3.13.1).

```bash
cd /Users/ompatel/Downloads/CSE576/Project/phase1_repro
source ../.venv-sage/bin/activate
python probe.py                                  # which SAGE modules run on this Mac
python smoke_test.py --edge-threshold 0.8        # SAGE default threshold -> 0 edges on toy corpus
python smoke_test.py --edge-threshold 0.3        # smoke-only threshold -> 2 edges, shows expansion
```

Outputs are written to `output/smoke_results_t<threshold>.json` and `output/unique_edges_toy.json`.

This is a code-path smoke test on a 10-passage synthetic corpus (`toy_corpus.py`). It is not a
reproduction of SAGE's reported numbers. The substitutions are listed in the `smoke_test.py` docstring.
