# TANGRAM

**English** | [简体中文](README.zh-CN.md)

> **TANGRAM** — Stateful-Operator Coarse-Graining for Proteins
> （中文名：**七巧**）

Named after the tangram puzzle: a small set of fixed pieces — reusable **residue
operator primitives** (each residue = bead + orientation + amino-acid type + a
discrete internal state *s*) — assemble, through general coupling rules, into a
multi-scale dynamics model of a protein, rather than re-learning a full
potential-energy surface for every protein.

The dynamics is hybrid sampling: Langevin (BAOAB) for the continuous coordinates
and Metropolis flips for the discrete internal states. The potential is linear in
its parameters (RBF / Fourier basis + tabulated weights), fitted jointly by force
matching and a variational likelihood — a convex objective.

## Naming

| Context | Name |
|---|---|
| Paper / public / docs | **TANGRAM** |
| Chinese | 七巧 (qī qiǎo) |
| Package / directory / import | `socg` (historical working codename, kept) |

**SOCG** (Stateful Operator Coarse Graining) in older material refers to the
same project.

## Status (2026-10-01)

- Phase 1 (peptide POC: Ala10 / CLN025) code is complete: the `socg/` package
  plus 10 scripts in `scripts/`.
- Fast pytest layer: 78 passed / 4 skipped; slow layer: 1 passed / 3 skipped
  (skips are due to the optional PeptideBuilder dependency not being installed).
- The CG pipeline has been smoke-tested end to end on synthetic data on a GPU node.

## Quick Start

```bash
git clone https://github.com/Cedrus810/TANGRAM
cd TANGRAM
source env/activate.sh          # mamba/conda env `openmm_dev` + PYTHONPATH
pytest -q -m "not slow"         # fast test layer (-m slow for the GPU slow layer)
```

The full command sequence for AA sampling, fitting, CG production and gate
evaluation is available via each script's `--help` in `scripts/`
(`run_aa.py` → `build_dataset.py` → `fit_model.py` → `run_cg.py` → `evaluate.py`).

## License

[MIT](LICENSE)
