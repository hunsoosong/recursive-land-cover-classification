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

The repository contains the experimental protocol of the four-arm
recursive training procedure described in Section 2 of the manuscript
(`src/`), the aggregated metric tables underlying every numerical figure
and headline value reported in the paper (`results/`), and a small
verification script that reads those tables and reproduces the Section 3.1
headline numbers (`scripts/summarize_trajectories.py`). External datasets
and trained checkpoints are not redistributed here; see
`docs/data_sources.md` and `docs/reproducibility.md` for pointers.

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
