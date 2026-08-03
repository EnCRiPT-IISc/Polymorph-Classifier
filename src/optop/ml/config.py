"""Default ML pipeline parameters (overridable from the CLI)."""

TEST_SIZE   = 0.2
SEED_BASE   = 42      # baseline split
SEED_TOP    = 123     # top-feature split
OVERFIT_GAP = 0.05
N_ITER      = 25
CV          = 5
MAX_SFS     = 6
MAX_TOP     = 20
IMP_THRESHOLD = 0.90  # cumulative-importance cutoff (cumulative mode)

PARAM_GRID = {
    "n_estimators": [20, 50, 100, 150, 200, 300, 350, 400, 500, 600],
    "learning_rate": [0.1, 0.2],
    "max_depth": [5, 10, 15, 20],
    "subsample": [0.6, 0.65, 0.7],
}
