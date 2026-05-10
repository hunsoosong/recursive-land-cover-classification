# Aggregated metric tables

Aggregated outputs from the recursive training experiment. These tables are
the *direct inputs* to every numerical figure and headline value reported
in the manuscript. (Figure 1 of the manuscript is a conceptual schematic
of the four arms and has no numerical inputs.)

| File | Shape / contents | Manuscript reference |
|---|---|---|
| `metrics_long.csv` | 120 rows = 3 sampling seeds x 4 arms x 10 generations; per-row OA / macro-F1 / worst-class recall / ECE / per-class F1 / per-class recall / predicted prevalence | Fig. 2, Section 3.1 |
| `metrics_BD_gen9_deltas.csv` | per-seed B - D gap at generation 9 (OA, macro-F1, worst-class recall, ECE, ECE ratio) | Section 3.1 supplementary numbers |
| `per_class_recall_trajectory.csv` | per-class recall / precision / F1 trajectory across all 10 generations | Fig. 3, Section 3.2 |
| `confusion_agg.npz` | seed-aggregated 10x10 confusion matrices, per arm and per generation | Fig. 3, Section 3.2 |
| `class_confusion_amplification.csv` | per-(seed, arm, true_class, pred_class) confusion rates at generation 0 vs generation 9, with amplification category | Section 3.2 |
| `pseudo_label_drift.csv` | pseudo-label drift across generations (arms C, D) | Section 3.2 narrative |
| `confidently_wrong_mass.csv` | per-(arm, gen, seed) test-set fraction of confidently-wrong predictions (confidence >= 0.5) | Fig. 4, Section 3.3 |
| `grid_per_class_acc.parquet` | per-class accuracy on a 6 degree spatial grid (gen 0 / gen 9) | Fig. 6, Section 3.5 |
| `scale_metrics.csv` | 5 sampling scales x 4 arms x 10 generations x 3 sampling seeds | Fig. S1, Appendix A |

All CSV files are encoded UTF-8 with the standard pandas dialect; NPZ
files are NumPy archives; parquet files use the pyarrow dialect.

## Provenance

These tables were aggregated from the per-seed run outputs of the
recursive training procedure (Section 2 of the manuscript). They are the
only artifacts needed to verify the reported numbers and regenerate the
figures that depend on experimental results; the per-seed checkpoints
and per-pixel prediction parquets used to produce them are not
redistributed here. See `docs/reproducibility.md` for the full
input-output graph.
