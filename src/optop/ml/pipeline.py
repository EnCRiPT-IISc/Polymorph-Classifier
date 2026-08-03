"""XGBoost phase-classification pipeline with automatic feature selection.

Baseline (all features) -> importance ranking -> top-feature selection (fixed
count or cumulative-importance) -> RandomizedSearchCV tuning -> Sequential
Forward Selection (k=1..max_sfs).  Confusion matrices are saved for every
stage and every k, alongside accuracies, feature lists, and all plots.
"""
from __future__ import annotations
import os, json, pickle
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
try:
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
except Exception:
    plot_sfs = None

from . import config as C


def _xgbc(n_jobs=-1, **kw):
    return xgb.XGBClassifier(tree_method="hist", n_jobs=n_jobs,
                             random_state=C.SEED_BASE, **kw)


def _select_top(importances, names, mode, max_top, threshold):
    order = np.argsort(importances)[::-1]
    if mode == "cumulative":
        cum = np.cumsum(importances[order])
        n = min(int(np.searchsorted(cum, threshold) + 1), max_top, len(names))
    else:
        n = min(max_top, len(names))
    idx = order[:n]
    return [names[i] for i in idx], idx


def _save_cm(y_true, y_pred, labels, names, stage, outdir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                 columns=[f"pred_{n}" for n in names]).to_csv(
        os.path.join(outdir, f"cm_{stage}.csv"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(f"Confusion matrix — {stage}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"cm_{stage}.png"), dpi=200)
    plt.close(fig)


def run_training(data_path, outdir, tag="run", *, feature_mode="cumulative",
                 max_top=C.MAX_TOP, imp_threshold=C.IMP_THRESHOLD,
                 param_grid=None, n_iter=C.N_ITER, cv=C.CV,
                 overfit_gap=C.OVERFIT_GAP, max_sfs=C.MAX_SFS, phases=None,
                 seed_base=C.SEED_BASE, seed_top=C.SEED_TOP):
    """Train a classifier on an OP CSV and write all results to `outdir`.

    phases: optional list of Class labels to keep (others dropped).
    """
    os.makedirs(outdir, exist_ok=True)
    param_grid = param_grid or C.PARAM_GRID
    t0 = time()
    print(f"[optop.train] data={data_path} outdir={outdir} tag={tag}\n"
          f"  mode={feature_mode} max_top={max_top} thr={imp_threshold} "
          f"n_iter={n_iter} cv={cv} max_sfs={max_sfs}", flush=True)

    df = pd.read_csv(data_path)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c or c == "mol_id"],
            inplace=True, errors="ignore")
    df = df[df["Class"].notna()].reset_index(drop=True)
    if phases:
        df = df[df["Class"].isin(phases)].reset_index(drop=True)
    X = df.drop("Class", axis=1); feat_names = np.array(X.columns)
    le = LabelEncoder(); y = le.fit_transform(df["Class"])
    names = list(le.classes_); labels = list(range(len(names)))
    json.dump({c: int(i) for c, i in zip(le.classes_, le.transform(le.classes_))},
              open(os.path.join(outdir, "class_mapping.json"), "w"), indent=2)
    print(f"  {len(df)} samples, {X.shape[1]} features, classes={names}", flush=True)

    stages = []

    # baseline (all features)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                          random_state=seed_base, stratify=y, shuffle=True)
    m1 = _xgbc(); m1.fit(Xtr, ytr)
    tr = accuracy_score(ytr, m1.predict(Xtr)); te = accuracy_score(yte, m1.predict(Xte))
    _save_cm(yte, m1.predict(Xte), labels, names, "baseline", outdir)
    pickle.dump(m1, open(os.path.join(outdir, "xgb_baseline.pkl"), "wb"))
    stages.append(dict(stage="baseline_all", n_features=int(X.shape[1]),
                       train_acc=tr, test_acc=te, gap=tr - te))
    print(f"  baseline: train={tr:.4f} test={te:.4f}", flush=True)

    importances = m1.feature_importances_
    top_names, top_idx = _select_top(importances, feat_names, feature_mode,
                                     max_top, imp_threshold)
    n_top = len(top_names)
    order = np.argsort(importances)[::-1]
    pd.DataFrame({"feature": feat_names[order], "importance": importances[order],
                  "cum_importance": np.cumsum(importances[order]),
                  "selected": [f in top_names for f in feat_names[order]]}).to_csv(
        os.path.join(outdir, "feature_importances.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * n_top)))
    ax.barh(range(n_top)[::-1], importances[top_idx])
    ax.set_yticks(range(n_top)[::-1]); ax.set_yticklabels(top_names)
    ax.set_xlabel("XGBoost gain importance"); ax.set_title(f"Top {n_top} ({feature_mode}) — {tag}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "feature_importances.png"), dpi=200)
    plt.close(fig)
    print(f"  selected {n_top} top features ({feature_mode})", flush=True)

    # top-N model (re-split)
    Xt = X[top_names]
    Xtr_t, Xte_t, ytr, yte = train_test_split(Xt, y, test_size=C.TEST_SIZE,
                                              random_state=seed_top, stratify=y, shuffle=True)
    mt = _xgbc(); mt.fit(Xtr_t, ytr)
    trt = accuracy_score(ytr, mt.predict(Xtr_t)); tet = accuracy_score(yte, mt.predict(Xte_t))
    _save_cm(yte, mt.predict(Xte_t), labels, names, "topN", outdir)
    pickle.dump(mt, open(os.path.join(outdir, "xgb_topN.pkl"), "wb"))
    stages.append(dict(stage=f"top{n_top}", n_features=n_top,
                       train_acc=trt, test_acc=tet, gap=trt - tet))
    pd.DataFrame(stages).to_csv(os.path.join(outdir, "stages.csv"), index=False)
    print(f"  top{n_top}: train={trt:.4f} test={tet:.4f}", flush=True)

    # hyperparameter tuning
    print("  RandomizedSearchCV ...", flush=True)
    search = RandomizedSearchCV(_xgbc(n_jobs=1), param_grid, cv=cv, n_jobs=-1,
                                n_iter=n_iter, scoring="accuracy", random_state=seed_base)
    search.fit(Xtr_t, ytr)
    best_params = search.best_params_
    json.dump(best_params, open(os.path.join(outdir, "best_params.json"), "w"), indent=2)
    print(f"  best_params={best_params}", flush=True)

    # SFS
    max_k = min(max_sfs, n_top)
    print(f"  SFS forward k=1..{max_k} ...", flush=True)
    sfs = SFS(xgb.XGBClassifier(tree_method="hist", n_jobs=1, random_state=seed_base, **best_params),
              k_features=max_k, forward=True, floating=False, scoring="accuracy",
              cv=cv, n_jobs=-1, verbose=0)
    sfs.fit(Xtr_t.values, ytr)
    if plot_sfs is not None:
        try:
            plot_sfs(sfs.get_metric_dict(confidence_interval=0.95), kind="std_dev")
            plt.title(f"SFS — {tag}"); plt.grid()
            plt.savefig(os.path.join(outdir, "sfs_stddev.png"), dpi=200); plt.close()
        except Exception:
            pass

    results = []
    fmodel = xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed_base, **best_params)
    for k in range(1, max_k + 1):
        idx = list(sfs.subsets_[k]["feature_idx"]); sel = [top_names[i] for i in idx]
        cv_acc = float(sfs.subsets_[k]["avg_score"])
        Xtr_s, Xte_s = Xtr_t.values[:, idx], Xte_t.values[:, idx]
        fmodel.fit(Xtr_s, ytr); ypred = fmodel.predict(Xte_s)
        tr_k = accuracy_score(ytr, fmodel.predict(Xtr_s)); te_k = accuracy_score(yte, ypred)
        _save_cm(yte, ypred, labels, names, f"{k}feat", outdir)
        pickle.dump(fmodel, open(os.path.join(outdir, f"xgb_{k}feat.pkl"), "wb"))
        results.append(dict(k=k, features=sel, train_acc=tr_k, test_acc=te_k,
                            cv_acc=cv_acc, gap=tr_k - te_k,
                            overfit=bool(tr_k - te_k > overfit_gap)))
        print(f"    k={k}: test={te_k:.4f} cv={cv_acc:.4f} gap={tr_k-te_k:.4f} {sel}", flush=True)
    pd.DataFrame(results).to_csv(os.path.join(outdir, "model_summary.csv"), index=False)

    ok = [r for r in results if not r["overfit"]] or results
    best = sorted(ok, key=lambda r: (-round(r["test_acc"], 6), r["k"]))[0]
    # classification report for the best model
    bidx = list(sfs.subsets_[best["k"]]["feature_idx"])
    fmodel.fit(Xtr_t.values[:, bidx], ytr)
    rep = classification_report(yte, fmodel.predict(Xte_t.values[:, bidx]),
                                target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(os.path.join(outdir, "best_classification_report.csv"))
    json.dump({"best_k": best["k"], "test_acc": best["test_acc"], "cv_acc": best["cv_acc"],
               "gap": best["gap"], "features": best["features"], "best_params": best_params,
               "n_top_features": n_top, "feature_mode": feature_mode, "classes": names},
              open(os.path.join(outdir, "best_model.json"), "w"), indent=2)

    # plots
    ks = [r["k"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, [r["train_acc"] for r in results], "o-", label="Train")
    ax.plot(ks, [r["test_acc"] for r in results], "s-", label="Test")
    ax.plot(ks, [r["cv_acc"] for r in results], "^--", label="CV")
    ax.set_xlabel("k (selected features)"); ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy vs k — {tag}"); ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "accuracy_curve.png"), dpi=200); plt.close(fig)

    s1 = results[0]["features"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in names:
        ax.hist(df[df["Class"] == c][s1[0]], bins=60, alpha=0.5, label=c, density=True)
    ax.set_xlabel(s1[0]); ax.set_ylabel("density"); ax.legend(); ax.set_title(f"{tag}: {s1[0]}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "hist_1feat.png"), dpi=200); plt.close(fig)
    if max_k >= 2:
        s2 = results[1]["features"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in names:
            d = df[df["Class"] == c]; ax.scatter(d[s2[0]], d[s2[1]], s=4, alpha=0.4, label=c)
        ax.set_xlabel(s2[0]); ax.set_ylabel(s2[1]); ax.legend(); ax.set_title(f"Best 2 OPs — {tag}")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "scatter_2feat.png"), dpi=200); plt.close(fig)
    if max_k >= 3:
        s3 = results[2]["features"]
        fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
        for c in names:
            d = df[df["Class"] == c]; ax.scatter(d[s3[0]], d[s3[1]], d[s3[2]], s=3, alpha=0.4, label=c)
        ax.set_xlabel(s3[0]); ax.set_ylabel(s3[1]); ax.set_zlabel(s3[2]); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "scatter_3feat.png"), dpi=200); plt.close(fig)
    for k in range(4, max_k + 1):
        sel = results[k - 1]["features"]
        Z = PCA(n_components=2).fit_transform(Xte_t[sel].values)
        fig, ax = plt.subplots(figsize=(7, 6))
        for ci, c in enumerate(names):
            mk = yte == ci; ax.scatter(Z[mk, 0], Z[mk, 1], s=5, alpha=0.4, label=c)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title(f"PCA k={k} — {tag}")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"pca_{k}feat.png"), dpi=200); plt.close(fig)

    print(f"\n  {tag}: BEST k={best['k']} test={best['test_acc']:.4f} "
          f"gap={best['gap']:.4f} (n_top={n_top}) in {(time()-t0)/60:.1f} min", flush=True)
    return {"results": results, "best": best, "best_params": best_params,
            "classes": names, "n_top": n_top}
