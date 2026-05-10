# Recursive pseudo-label reuse in land-cover mapping with geospatial foundation embeddings

This repository accompanies the paper:

> Hwang, J., Jeong, D., Song, H. (2026).
> *Stable Accuracy, Drifting Reliability: Recursive Pseudo-Label Reuse in
> Land-Cover Mapping with Geospatial Foundation Embeddings.*

It provides:

1. The **experimental protocol** of the four-arm recursive training procedure
   described in Section 2 of the manuscript, expressed as documented Python
   (`src/`).
2. The **aggregated metric tables** from which every numerical figure and
   headline value in the paper was produced (`results/`). Figure 1 of the
   manuscript is a conceptual schematic of the four arms and has no
   numerical inputs.
3. A small **verification script** that reads those tables and prints the
   per-arm trajectory and the Section 3.1 headline numbers
   (`scripts/summarize_trajectories.py`).

## Scope

The recursive training procedure runs on top of two publicly available
datasets: the global land-cover validation reference dataset of Lesiv
et al. (2025) and the AlphaEarth Foundation embeddings of Brown et al.
(2025). This repository contains:

- the experimental design (four arms x ten generations x three sampling
  seeds x five sampling scales) at a level of detail sufficient for the
  reported numbers to be independently verified, and
- the aggregated outputs underlying every numerical figure and headline
  value in the paper.

External datasets and trained checkpoints can be obtained from their
original providers; see `docs/data_sources.md` and `docs/reproducibility.md`
for pointers.

## Quick start: verify the headline numbers

```bash
conda env create -f environment.yml
conda activate recursive-land-cover
python scripts/summarize_trajectories.py
```

This reads `results/metrics_long.csv` and prints, for each of the four
arms, the mean +/- standard deviation of overall accuracy, macro-F1,
worst-class recall, and ECE at every generation. It then prints the
Section 3.1 headline numbers: B's gain from Generation 0, C's drop from
Generation 0, A and D stability, and the D / B ECE ratio at Generation 9
(matching the annotation on Fig. 2(d)).

Add `--plot` to also save a small four-panel verification PNG.

## Layout

```
.
├── src/
│   ├── config.py                 Hyperparameters reported in the manuscript
│   ├── classes.py                Lesiv 10-class harmonisation (Section 2.2)
│   └── protocol.py               Four-arm recursive training protocol
│                                 (Section 2.1) as documented Python
├── scripts/
│   └── summarize_trajectories.py Verification of headline numbers
├── results/                      Aggregated metric tables (figure inputs)
├── docs/
│   ├── data_sources.md           AlphaEarth + Lesiv pointers
│   └── reproducibility.md        Reproduction scope
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── LICENSE                       MIT
```

## Citation

If you use this code or the derived artifacts in `results/`, please cite
the paper above. A `CITATION.cff` file is provided for tooling support.

This repository is also archived on Zenodo (DOI to be added upon
acceptance).

## License

MIT - see `LICENSE`.
