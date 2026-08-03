"""ML core: baseline training, feature ranking, tuning, SFS."""

import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from .config import (CUMUL_IMP_THRESHOLD, MAX_TOP_FEATURES, MAX_SFS_FEATURES,
                     PARAM_GRID, N_ITER_SEARCH, RANDOM_STATE, OVERFIT_GAP)


def train_baseline(X_tr, y_tr, X_te, y_te):
    """Train full-feature model, return (importances, train_acc, test_acc)."""
    m = xgb.XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    m.fit(X_tr, y_tr)
    tr = accuracy_score(y_tr, m.predict(X_tr))
    te = accuracy_score(y_te, m.predict(X_te))
    return m.feature_importances_, tr, te


def select_top_features(importances, names, threshold=None, max_n=None):
    """Select features by cumulative importance, capped at max_n."""
    threshold = threshold or CUMUL_IMP_THRESHOLD
    max_n = max_n or MAX_TOP_FEATURES
    order = np.argsort(importances)[::-1]
    cum = np.cumsum(importances[order])
    n = min(int(np.searchsorted(cum, threshold) + 1), max_n, len(names))
    return names[order[:n]], order[:n]


def tune_hyperparams(X_tr, y_tr, param_grid=None, n_iter=None):
    """RandomizedSearchCV. Users can pass their own param_grid."""
    param_grid = param_grid or PARAM_GRID
    n_iter = n_iter or N_ITER_SEARCH
    search = RandomizedSearchCV(
        xgb.XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        param_grid, cv=5, n_jobs=-1, n_iter=n_iter,
        scoring="accuracy", random_state=RANDOM_STATE,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_


def run_sfs(best_params, top_names, X_tr, X_te, y_tr, y_te, outdir,
            max_k=None, overfit_gap=None):
    """Run SFS for k=1..max_k. Returns list of result dicts."""
    max_k = max_k or MAX_SFS_FEATURES
    overfit_gap = overfit_gap or OVERFIT_GAP
    model = xgb.XGBClassifier(**best_params, n_jobs=-1, random_state=RANDOM_STATE)
    results = []
    for k in range(1, max_k + 1):
        sfs = SFS(model, k_features=k, forward=True, floating=False,
                  scoring="accuracy", cv=5, n_jobs=-1, verbose=0)
        sfs.fit(X_tr, y_tr)
        sel = list(top_names[list(sfs.k_feature_idx_)])
        X_tr_s, X_te_s = sfs.transform(X_tr), sfs.transform(X_te)
        model.fit(X_tr_s, y_tr)
        tr = accuracy_score(y_tr, model.predict(X_tr_s))
        te = accuracy_score(y_te, model.predict(X_te_s))
        results.append(dict(
            k=k, features=sel, train_acc=tr, test_acc=te,
            cv_acc=sfs.k_score_, overfit=tr - te > overfit_gap,
            y_pred=model.predict(X_te_s),
        ))
        with open(os.path.join(outdir, f"xgb_{k}feat.pkl"), "wb") as f:
            pickle.dump(model, f)
    return results
