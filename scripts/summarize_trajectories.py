#!/usr/bin/env python3
"""
Print a verification summary of the recursive training trajectories.

Reads `results/metrics_long.csv` (4 arms x 10 generations x 3 sampling
seeds) and prints, for each arm, the mean and standard deviation of the
four headline metrics (overall accuracy, macro-F1, worst-class recall,
ECE) at every generation.

Also prints the Section 3.1 headline numbers (B's gain from Generation 0,
C's drop from Generation 0, A and D stability, and the D / B ECE ratio at
Generation 9 from Fig. 2(d)).

Usage
-----
    python scripts/summarize_trajectories.py
    python scripts/summarize_trajectories.py --plot     # also save PNG

The optional --plot flag writes a four-panel trajectory PNG to
`scripts/summary_trajectory.png`. The styling is intentionally plain;
the polished publication figure is rendered separately and is not part
of this verification script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "results"

ARM_NAMES = {
    "A": "Human Replace",
    "B": "Human Accumulate",
    "C": "Pseudo Replace",
    "D": "Pseudo Accumulate",
}


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL / "metrics_long.csv")
    keep = ["sampling_seed", "arm", "gen",
            "accuracy", "macro_f1", "worst_class_recall", "ece_scaled"]
    return df[keep].copy()


def summarise_per_arm(df: pd.DataFrame) -> None:
    """Print a per-arm trajectory of mean +/- std across sampling seeds."""
    print("=" * 72)
    print("Recursive training trajectories (mean +/- std across 3 sampling seeds)")
    print("=" * 72)

    for arm in ("A", "B", "C", "D"):
        sub = df[df["arm"] == arm]
        agg = sub.groupby("gen").agg(
            oa_mean=("accuracy", "mean"), oa_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
            wcr_mean=("worst_class_recall", "mean"),
            wcr_std=("worst_class_recall", "std"),
            ece_mean=("ece_scaled", "mean"), ece_std=("ece_scaled", "std"),
        ).reset_index()

        print(f"\nArm {arm} ({ARM_NAMES[arm]}):")
        print(f"  {'gen':>3} {'OA':>16} {'macro-F1':>16} "
              f"{'worst-recall':>16} {'ECE':>14}")
        for _, r in agg.iterrows():
            print(f"  {int(r['gen']):>3d} "
                  f"  {r['oa_mean']:.4f}+/-{r['oa_std']:.4f}"
                  f"   {r['f1_mean']:.4f}+/-{r['f1_std']:.4f}"
                  f"   {r['wcr_mean']:.4f}+/-{r['wcr_std']:.4f}"
                  f"   {r['ece_mean']:.4f}+/-{r['ece_std']:.4f}")


def headline_metrics(df: pd.DataFrame) -> None:
    """Print the Section 3.1 headline numbers (Gen 0 -> Gen 9 by arm).

    All values are computed by averaging across the three sampling seeds
    first and then taking arm-level differences and ratios. The D / B ECE
    ratio in particular is reported as a ratio of seed-means, which is the
    quantity annotated on Fig. 2(d) of the manuscript.
    """
    g = (df.groupby(["arm", "gen"])
            [["accuracy", "macro_f1", "worst_class_recall", "ece_scaled"]]
            .mean())

    def _gen0_gen9(arm: str, col: str) -> tuple[float, float, float]:
        a = g.loc[(arm, 0), col]
        b = g.loc[(arm, 9), col]
        return a, b, b - a

    print("\n" + "=" * 72)
    print("Section 3.1 headline (mean across 3 sampling seeds)")
    print("=" * 72)

    # B (Human Accumulate) — sustained learning on OA / F1.
    a_oa, b_oa, d_oa = _gen0_gen9("B", "accuracy")
    a_f1, b_f1, d_f1 = _gen0_gen9("B", "macro_f1")
    a_wcr, b_wcr, d_wcr = _gen0_gen9("B", "worst_class_recall")
    print("\nB (Human Accumulate) - sustained learning:")
    print(f"  Overall accuracy   : {100*a_oa:.2f} -> {100*b_oa:.2f} pp"
          f"   (gain {100*d_oa:+.2f} pp)")
    print(f"  Macro-F1           : {100*a_f1:.2f} -> {100*b_f1:.2f} pp"
          f"   (gain {100*d_f1:+.2f} pp)")
    print(f"  Worst-class recall : {a_wcr:.2f} -> {b_wcr:.2f}"
          f"      (gain {100*d_wcr:+.2f} pp)")
    print("  paper says: '+8 to +9 percentage points (OA / F1)' "
          "and worst-class recall 'climbing from 0.22 to 0.33'")

    # C (Pseudo Replace) — degradation.
    a_oa, b_oa, d_oa = _gen0_gen9("C", "accuracy")
    a_f1, b_f1, d_f1 = _gen0_gen9("C", "macro_f1")
    _, _, d_wcr = _gen0_gen9("C", "worst_class_recall")
    print("\nC (Pseudo Replace) - degradation:")
    print(f"  Overall accuracy   : {100*a_oa:.2f} -> {100*b_oa:.2f} pp"
          f"   (drop {100*d_oa:+.2f} pp)")
    print(f"  Macro-F1           : {100*a_f1:.2f} -> {100*b_f1:.2f} pp"
          f"   (drop {100*d_f1:+.2f} pp)")
    print(f"  Worst-class recall : drop {100*d_wcr:+.2f} pp")
    print("  paper says: OA drops by ~6, macro-F1 by ~7, "
          "worst-class recall by ~8")

    # A and D — stability near Gen 0 (with D's worst-recall drift).
    a_a0, a_a9, _ = _gen0_gen9("A", "accuracy")
    d_a0, d_a9, _ = _gen0_gen9("D", "accuracy")
    _, _, d_wcr = _gen0_gen9("D", "worst_class_recall")
    print("\nA and D - stability near Generation 0:")
    print(f"  A overall accuracy : {100*a_a0:.2f} -> {100*a_a9:.2f} pp"
          f"   (delta {100*(a_a9-a_a0):+.2f} pp)")
    print(f"  D overall accuracy : {100*d_a0:.2f} -> {100*d_a9:.2f} pp"
          f"   (delta {100*(d_a9-d_a0):+.2f} pp)")
    print(f"  D worst-class recall drift : {100*d_wcr:+.2f} pp")
    print("  paper says: A and D 'remain near the Generation 0 level',"
          " D worst-recall drifts down by ~4 pp")

    # D / B ECE ratio at Gen 9 (ratio of seed-means; matches Fig. 2(d)).
    b_ece9 = g.loc[("B", 9), "ece_scaled"]
    d_ece9 = g.loc[("D", 9), "ece_scaled"]
    print(f"\nD / B ECE ratio at Generation 9 (ratio of seed-means): "
          f"{d_ece9 / b_ece9:.2f}x")
    print("  paper Fig. 2(d) annotation: '3.2x'")


def maybe_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Optional matplotlib trajectory plot (4 panels x 4 arms)."""
    import matplotlib.pyplot as plt

    metrics = [("accuracy", "Overall accuracy"),
               ("macro_f1", "Macro-F1"),
               ("worst_class_recall", "Worst-class recall"),
               ("ece_scaled", "ECE (post temperature scaling)")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (col, title) in zip(axes.flat, metrics):
        for arm in ("A", "B", "C", "D"):
            sub = df[df["arm"] == arm]
            agg = sub.groupby("gen")[col].agg(["mean", "std"]).reset_index()
            ax.plot(agg["gen"], agg["mean"], marker="o",
                    label=f"{arm}: {ARM_NAMES[arm]}")
            ax.fill_between(agg["gen"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true",
                   help="Also save a 4-panel verification PNG")
    args = p.parse_args()

    df = load_metrics()
    summarise_per_arm(df)
    headline_metrics(df)

    if args.plot:
        maybe_plot(df, Path(__file__).parent / "summary_trajectory.png")


if __name__ == "__main__":
    main()
