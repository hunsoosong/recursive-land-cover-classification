"""
Experiment configuration for the recursive pseudo-label reuse study.

The values below correspond to the procedure described in Section 2 of the
manuscript. Hyperparameters that are explicitly stated in the manuscript are
grouped at the top of the file; implementation defaults that the manuscript
does not commit to are grouped at the bottom.

Class definitions and the Lesiv 10-class harmonisation are kept in
`src/classes.py`.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Hyperparameters reported in the manuscript
# ---------------------------------------------------------------------------

# --- Sampling design (Section 2.3) ---
GRID_DEG = 3.0                  # 3 degree x 3 degree global grid
CELL_CAP = 180                  # max points per cell, applied across classes
N_CLASSES = 10                  # see src/classes.py
TOTAL_EXPERIMENT_SAMPLES = 210_000   # 21,000 per class

# 3-way split (recursive pool / validation / test)
TRAIN_PER_CLASS = 10_000        # recursive pool: 100,000 total
VAL_PER_CLASS = 1_000           # validation: 10,000 total
TEST_PER_CLASS = 10_000         # held-out test: 100,000 total

# Generation pools (10 disjoint, class-balanced)
N_POOLS = 10                    # data1 ... data10
N_GENERATIONS = 10              # Gen 0 ... Gen 9 (Model 1 ... Model 10)
SAMPLES_PER_CLASS_PER_POOL = TRAIN_PER_CLASS // N_POOLS    # 1,000 (S5)

# Sampling scales reported in Section 2.3 (per-generation batch sizes)
SAMPLING_SCALES = {
    "S1": 1_000,   # 100 per class
    "S2": 3_000,   # 300 per class
    "S3": 5_000,   # 500 per class
    "S4": 7_500,   # 750 per class
    "S5": 10_000,  # 1,000 per class (full pool used)
}

# --- Independent spatial sampling seeds (Section 2.3) ---
SAMPLING_SEEDS = (42, 123, 256)

# --- AlphaEarth Foundation embeddings (Section 2.2) ---
N_AEF_BANDS = 64
AEF_EPOCH_YEAR = 2017
AEF_NATIVE_RESOLUTION_M = 10

# --- Classifier (Section 2.4) ---
MLP_HIDDEN_DIMS = [256, 128]    # 64 -> 256 -> 128 -> 10
MLP_DROPOUT = 0.2
MLP_ACTIVATION = "relu"

# --- Training (Section 2.4) ---
LEARNING_RATE = 1e-3            # Adam
EARLY_STOPPING_PATIENCE = 15    # on validation macro-F1
ECE_N_BINS = 15                 # equal-width bins on max softmax probability

# --- Confidence diagnostics (Section 2.4) ---
CONFIDENTLY_WRONG_THRESHOLD = 0.5

# --- Calibration (Section 2.4) ---
USE_TEMPERATURE_SCALING = True

# ---------------------------------------------------------------------------
# 2. Implementation defaults (not central to the recursive design)
# ---------------------------------------------------------------------------
# Values used in our runs but not reported in the manuscript text.
# These can be modified without changing the qualitative conclusions of the
# study; they govern training-loop bookkeeping rather than the recursive
# label-reuse procedure itself.

BATCH_SIZE = 2048
MAX_EPOCHS = 1000               # early stopping terminates well before this
WEIGHT_DECAY = 1e-4
MODEL_SEED = 42                 # see Section 2.4 on initialisation handling

# Data partition seed for the spatial-block split (independent of the three
# sampling seeds above, which control the spatially balanced pool draw).
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# 3. Convenience: derived constants
# ---------------------------------------------------------------------------
TOTAL_PER_CLASS = TOTAL_EXPERIMENT_SAMPLES // N_CLASSES   # 21,000
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0
N_LAT_CELLS = int((LAT_MAX - LAT_MIN) / GRID_DEG)         # 60
N_LON_CELLS = int((LON_MAX - LON_MIN) / GRID_DEG)         # 120

# ---------------------------------------------------------------------------
# 4. Arm definitions (Section 2.1)
# ---------------------------------------------------------------------------
ARMS = {
    "A": {
        "name": "Human Replace",
        "label_source": "human",
        "strategy": "replace",
    },
    "B": {
        "name": "Human Accumulate",
        "label_source": "human",
        "strategy": "accumulate",
    },
    "C": {
        "name": "Pseudo Replace",
        "label_source": "pseudo",
        "strategy": "replace",
    },
    "D": {
        "name": "Pseudo Accumulate",
        "label_source": "pseudo",
        "strategy": "accumulate",
    },
}

ARM_IDS = ("A", "B", "C", "D")

# ---------------------------------------------------------------------------
# 5. Default output directory
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("runs")
