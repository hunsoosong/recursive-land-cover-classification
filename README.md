# Recursive pseudo-label reuse in land-cover mapping with geospatial foundation embeddings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20108217.svg)](https://doi.org/10.5281/zenodo.20108217)

This repository accompanies the paper:

> Hwang, J., Jeong, D., Song, H. (2026).
> *Stable Accuracy, Drifting Reliability: Recursive Pseudo-Label Reuse in
> Land-Cover Mapping with Geospatial Foundation Embeddings.*

## Overview

Large-scale land-cover mapping increasingly reuses model-derived layers as
labels, masks, or sampling frames for subsequent models, creating recursive
map–model–map feedback loops in which previous model outputs shape later
training data. The paper isolates this feedback in a controlled four-arm
two-by-two factorial design (human versus model-generated labels × replace
versus accumulation reuse rule) on AlphaEarth Foundation embeddings and
the Lesiv et al. (2025) global land-cover validation reference dataset.

Across ten recursive generations and five sampling scales, human-label
accumulation produces sustained learning across overall accuracy,
macro-F1, worst-class recall, and calibration. Unfiltered pseudo-label
accumulation, applied to the same foundation embedding under the same
accumulation rule, behaves instead as an **error lock-in** mechanism:
aggregate accuracy stays near the initial level while calibration
deteriorates, predicted class prevalence drifts away from the balanced
reference distribution, and degradation concentrates in semantically
transitional classes such as grassland, sparse vegetation, and herbaceous
wetland. The paper recommends that recursively updated land-cover
workflows should be evaluated with periodic human anchoring and
diagnostics beyond overall accuracy — calibration, predicted class
distribution, class-boundary errors, and region-by-class reliability.

## Repository contents

- **`src/`** — Source code for the four-arm recursive training
  procedure described in Section 2 of the manuscript, including the
  manuscript-aligned hyperparameter configuration (`config.py`), the
  Lesiv 10-class harmonisation aligned with the ESA WorldCover main
  classes (`classes.py`), and the per-arm training-set composition logic
  for the human-versus-pseudo × replace-versus-accumulate factorial
  design (`protocol.py`).
- **`results/`** — Aggregated metric tables underlying every numerical
  figure and headline value reported in the paper, covering 3 spatial
  sampling seeds × 4 arms × 10 generations × 5 sampling scales: overall
  accuracy, macro-F1, worst-class recall, Expected Calibration Error
  after temperature scaling, per-class trajectories, generation-wise
  confusion matrices, predicted-prevalence diagnostics, confidently-wrong
  fractions, and 6° spatial-grid per-class accuracy.
- **`scripts/summarize_trajectories.py`** — Verification script that
  reads the aggregated tables and reproduces the Section 3.1 headline
  numbers in a single command (per-arm Generation 0 → Generation 9
  trajectories, B's sustained learning gain, C's degradation, A and D
  stability, and the D / B Expected Calibration Error ratio at
  Generation 9 annotated in Fig. 2(d)).
- **`docs/`** — External data sources (AlphaEarth, Lesiv) and
  reproduction scope.

External datasets and trained checkpoints are not redistributed in this
repository; see `docs/data_sources.md` and `docs/reproducibility.md` for
pointers.

## Public data sources used in this study

- Land-cover reference labels: Lesiv et al. (2025) global land-cover
  validation reference dataset — <https://doi.org/10.5194/essd-17-6149-2025>
- Geospatial foundation embeddings: AlphaEarth Foundations
  (Brown et al., 2025) — Google Earth Engine asset
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`

## Citation

If you use this code or the derived artifacts in `results/`, please cite
the paper above. A `CITATION.cff` file is provided for tooling support.

This repository is archived on Zenodo:
[10.5281/zenodo.20108217](https://doi.org/10.5281/zenodo.20108217).

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF)
grant funded by the Korea government (MSIT) (RS-2026-25499133).

## License

MIT — see `LICENSE`.
