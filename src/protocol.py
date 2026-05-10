"""
Recursive label-reuse experimental protocol (Section 2 of the manuscript).

This module documents the four-arm recursive training procedure as code so
that the experimental design is unambiguous. It does *not* include a
detailed training-loop implementation: the classifier (a small MLP), the
optimizer (Adam), the early-stopping rule (patience on validation
macro-F1), and the post-hoc temperature scaling step are standard
primitives described in Section 2.4 of the paper and supported directly
by PyTorch and SciPy. Hyperparameter values are kept in `src/config.py`.

The function `build_training_set` below is the actual experimental
contribution: it specifies, for every (arm, generation) pair, how the
training set is composed from the per-generation human-labelled pools
and the previous-generation pseudo-labels. The four arms differ only
through this composition rule.

Sections referenced:

  Section 2.1 - Recursive self-training: four label-source x accumulation
                strategies (the protocol implemented below).
  Section 2.2 - Reference labels (Lesiv et al., 2025) and AlphaEarth
                Foundation embeddings (Brown et al., 2025); see
                `docs/data_sources.md`.
  Section 2.3 - Spatially balanced sampling on a 3 degree x 3 degree
                global grid with per-cell cap K = 180, drawing points one
                at a time and prioritising (class, cell) combinations whose
                class is currently furthest from its target count, yielding
                210,000 class-balanced points partitioned into a
                100,000-point recursive training pool, a 10,000-point
                validation pool, and a 100,000-point held-out test set.
                The recursive training pool is split into ten disjoint
                class-balanced generation pools (data1 ... data10).
  Section 2.4 - Classifier: 64-dim AlphaEarth embedding -> 256 -> 128 ->
                10-class softmax (ReLU, dropout 0.2). Training: Adam,
                learning rate 1e-3, cross-entropy loss, early stopping on
                validation macro-F1 with patience 15. Calibration:
                post-hoc temperature scaling fitted on the validation
                pool. Diagnostics: overall accuracy, macro-F1,
                worst-class recall, ECE (15 equal-width bins),
                predicted-prevalence L1 distance from balanced
                reference, share of confidently-wrong predictions
                (confidence >= 0.5).

The aggregated outputs of running this protocol across three sampling
seeds, four arms, ten generations, and five sampling scales are
included in `results/`. See `scripts/summarize_trajectories.py` for a
runnable verification of the headline numbers.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import ARMS, ARM_IDS, N_GENERATIONS, N_POOLS, MODEL_SEED


# ---------------------------------------------------------------------------
# Per-arm training-set composition (Section 2.1)
# ---------------------------------------------------------------------------

def build_training_set(arm_id: str,
                       gen: int,
                       human_pools: dict[str, dict[str, np.ndarray]],
                       pseudo_store: dict[tuple[str, str], np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray, str]:
    """Compose the training set for ``arm_id`` at generation ``gen``.

    Parameters
    ----------
    arm_id : {"A", "B", "C", "D"}
        Identifier of the recursive training arm. See `ARMS` in
        `src/config.py` for the full description; in short:

          A : human labels, replace previous batch each generation
          B : human labels, accumulate batches across generations
          C : pseudo labels, replace previous batch each generation
          D : pseudo labels, accumulate batches across generations
              (with data1 retained as a human-labelled anchor)

    gen : int in {0, 1, ..., N_GENERATIONS - 1}
        Generation index. All four arms start from the *same* generation 0
        model trained on data1 with human labels.
    human_pools : dict
        ``{"data1": {"X": ..., "y": ...}, ..., "dataN": ...}``.
        The ten disjoint, class-balanced generation pools constructed from
        the recursive training pool (Section 2.3).
    pseudo_store : dict
        ``{(arm_id, "dataK"): pseudo_y}``. Pseudo-labels are *frozen* at
        admission time: once stored under (arm_id, "dataK"), they are
        never overwritten by a later-generation classifier.

    Returns
    -------
    X, y : np.ndarray
        Training features and labels (concatenated across batches if the
        arm uses an accumulation rule).
    description : str
        Human-readable description of the resulting training set, useful
        for logging and audit trails.
    """
    arm = ARMS[arm_id]
    name = f"data{gen + 1}"

    # Generation 0: every arm starts from the same human-labelled batch.
    if gen == 0:
        return human_pools[name]["X"], human_pools[name]["y"], f"{name}(human)"

    if arm["label_source"] == "human":
        if arm["strategy"] == "replace":
            # Arm A: discard previous batch, draw a fresh one with human labels.
            return (human_pools[name]["X"], human_pools[name]["y"],
                    f"{name}(human)")
        # Arm B: append the new human-labelled batch to all previous batches.
        Xs = [human_pools[f"data{g+1}"]["X"] for g in range(gen + 1)]
        ys = [human_pools[f"data{g+1}"]["y"] for g in range(gen + 1)]
        desc = "+".join(f"data{g+1}(human)" for g in range(gen + 1))
        return np.concatenate(Xs), np.concatenate(ys), desc

    # Pseudo-label arms.
    if arm["strategy"] == "replace":
        # Arm C: discard previous batch; new batch's labels come from the
        # previous-generation classifier (argmax over the new batch).
        return (human_pools[name]["X"], pseudo_store[(arm_id, name)],
                f"pseudo({name})")

    # Arm D: data1 (human) anchor + frozen pseudo-labels for data2..dataN.
    Xs = [human_pools["data1"]["X"]]
    ys = [human_pools["data1"]["y"]]
    for g in range(1, gen + 1):
        p = f"data{g + 1}"
        Xs.append(human_pools[p]["X"])
        ys.append(pseudo_store[(arm_id, p)])
    desc = ("data1(human)+"
            + "+".join(f"pseudo(data{g+1})" for g in range(1, gen + 1)))
    return np.concatenate(Xs), np.concatenate(ys), desc


# ---------------------------------------------------------------------------
# Outer loop sketch (Section 2.1)
# ---------------------------------------------------------------------------

def recursive_loop_sketch(arms: Iterable[str] = ARM_IDS,
                          n_generations: int = N_GENERATIONS) -> str:
    """Return a textual outline of the per-seed recursive training loop.

    The full loop is:

        for gen in 0 .. n_generations - 1:
            # 1. (For arms C, D, gen >= 1) generate hard pseudo-labels for
            #    the upcoming batch using the previous-generation model:
            #        pseudo_y = previous_model(X_next).argmax(dim=1)
            #    Pseudo-labels are stored frozen in pseudo_store[(arm, batch)].
            # 2. For each arm, compose the training set via
            #    `build_training_set(arm, gen, human_pools, pseudo_store)`.
            # 3. Train a fresh MLP from scratch (no warm-start across
            #    generations) with cross-entropy + Adam (lr 1e-3) + early
            #    stopping on validation macro-F1 (patience 15). The
            #    classifier architecture is 64 -> 256 -> 128 -> 10 with
            #    ReLU and dropout 0.2.
            # 4. Fit a single-parameter temperature scalar on the
            #    validation pool by NLL minimisation.
            # 5. Evaluate the calibrated model on the held-out test set
            #    and record overall accuracy, macro-F1, worst-class
            #    recall, ECE (15 bins), per-class F1 / recall, and
            #    predicted prevalence per class.

    Across-seed structure: repeat the entire loop above for each of the
    three independent spatial sampling seeds defined in `config.py`. The
    full design also varies the per-generation batch size across five
    sampling scales (S1 ... S5); see `results/scale_metrics.csv` for
    the resulting metrics.
    """
    return recursive_loop_sketch.__doc__ or ""
