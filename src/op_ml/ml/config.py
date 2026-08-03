"""All tunable pipeline parameters."""

# ── OP Computation ───────────────────────────────────────────────────
RCUT = 4.5
CENTRAL_ATOM = "O"

# ── Feature Selection ────────────────────────────────────────────────
CUMUL_IMP_THRESHOLD = 0.90   # cumulative importance cutoff
MAX_TOP_FEATURES = 20        # hard cap for SFS tractability
MAX_SFS_FEATURES = 6         # SFS evaluates k=1..this

# ── Train/Test ───────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Hyperparameter Search ────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [5, 10, 15, 20],
    "subsample": [0.6, 0.7, 0.8],
}
N_ITER_SEARCH = 25

# ── Overfitting ──────────────────────────────────────────────────────
OVERFIT_GAP = 0.05           # train-test gap above this = overfitting
