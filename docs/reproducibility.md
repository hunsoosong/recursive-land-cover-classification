# Reproduction scope

This document records which artifacts are provided in the repository and
how they map onto the figures and headline numbers reported in the paper.
The recursive training procedure runs on standard primitives (a small
MLP, Adam, early stopping, post-hoc temperature scaling); the constants
used in our runs are exposed in `src/config.py` and the four-arm
composition logic is in `src/protocol.py`.

## Verifiable from this repository

* **Every number reported in the manuscript.** All percentages, gap
  statistics, ratio values, per-class F1 / recall, predicted-prevalence
  L1 distances, confidently-wrong fractions, etc., can be recomputed
  directly from the CSV / NPZ / parquet files in `results/` (in
  particular `metrics_long.csv`, `metrics_BD_gen9_deltas.csv`, and
  `confidently_wrong_mass.csv`). The script
  `scripts/summarize_trajectories.py` performs a one-line verification
  of the abstract and Section 3.1 headline numbers.

* **Figures 2, 3, 4, 6, and Appendix Fig. S1.** The aggregated tables in
  `results/` contain all values plotted in these figures. Readers wishing
  to regenerate them can do so directly from the tables with their
  preferred plotting code. Figure 1 of the manuscript is a conceptual
  schematic of the four arms with no numerical inputs. Figure 5 (dense
  per-pixel land-cover predictions over three 0.4-degree tiles) requires
  a full pixel-level inference pass over the AlphaEarth embeddings of
  each tile and is therefore reproducible only end-to-end from the
  trained checkpoints; the published rendering is shown in the paper.

* **The four-arm experimental design.** The exact composition of the
  training set for every (arm, generation) pair is specified by
  `src/protocol.py::build_training_set`, and the outer-loop
  structure is documented in `recursive_loop_sketch` in the same file.

## Procedure documentation

Section 2 of the paper describes the sampling, splitting, training, and
calibration procedure using standard primitives. The constants used in
our runs are exposed in `src/config.py`; the four-arm training-set
composition logic is implemented in `src/protocol.py`. End-to-end re-runs
from the original data sources are possible by combining these with the
standard PyTorch / SciPy primitives cited in Section 2.4 (Adam, early
stopping on macro-F1, temperature scaling fitted on a validation pool,
ECE with 15 equal-width bins).

The expected level of numerical agreement between such a re-run and the
published values is qualitative: the same trajectories, the same
cross-arm orderings, and sub-percent numerical differences. Exact
bitwise reproduction is not generally achievable in CUDA-accelerated
PyTorch workflows because of cuDNN algorithm selection, atomic-add
reordering, and version drift, and is therefore not the target.

## External data and large artifacts

The following are obtained from their original providers rather than
redistributed here:

* **AlphaEarth Foundation embedding cube** - Google Earth Engine asset
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. See `docs/data_sources.md`.

* **Lesiv et al. (2025) reference dataset** - available with the
  original publication.

* **Trained MLP checkpoints** - approximately 24 GB across all sampling
  seeds, scales, arms, and generations; produced by the procedure
  documented above.

* **Per-pixel prediction parquets** used in intermediate analyses
  (approximately 12 million rows per sampling seed) - produced by a
  shallow inference pass over the checkpoints.
