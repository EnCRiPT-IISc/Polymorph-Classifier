"""All visualisation functions — one file, zero state."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from .config import OVERFIT_GAP


def feature_importance(importances, names, n, outdir):
    idx = np.argsort(importances)[::-1][:n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names[idx][::-1], fontsize=10)
    ax.set_xlabel("Importance"); ax.set_title(f"Top {n} Feature Importances")
    ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_importances.png"), dpi=300); plt.close(fig)


def accuracy_curve(results, outdir):
    ks = [r["k"] for r in results]
    tr = [r["train_acc"] for r in results]
    te = [r["test_acc"] for r in results]
    gap = [t - e for t, e in zip(tr, te)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(ks, tr, "o-", label="Train"); a1.plot(ks, te, "s-", label="Test")
    a1.set_xlabel("Features"); a1.set_ylabel("Accuracy"); a1.legend()
    a1.grid(alpha=0.3); a1.set_xticks(ks)
    a2.bar(ks, gap, color=["red" if g > OVERFIT_GAP else "steelblue" for g in gap])
    a2.axhline(OVERFIT_GAP, color="red", ls="--", label=f"Threshold ({OVERFIT_GAP})")
    a2.set_xlabel("Features"); a2.set_ylabel("Train-Test Gap"); a2.legend()
    a2.grid(alpha=0.3); a2.set_xticks(ks)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "accuracy_curve.png"), dpi=300); plt.close(fig)


def conf_matrix(y_true, y_pred, le, k, outdir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix ({k} features)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"cm_{k}feat.png"), dpi=300); plt.close(fig)


def histogram_1f(df, feat, classes, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in classes:
        ax.hist(df.loc[df["Class"] == c, feat], bins=60, alpha=0.5, label=c, density=True)
    ax.set_xlabel(feat, fontsize=14); ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {feat} by Phase"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "hist_1feat.png"), dpi=300); plt.close(fig)


def scatter_2f(df, feats, classes, outdir):
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in classes:
        m = df["Class"] == c
        ax.scatter(df.loc[m, feats[0]], df.loc[m, feats[1]], alpha=0.3, s=8, label=c)
    ax.set_xlabel(feats[0], fontsize=14); ax.set_ylabel(feats[1], fontsize=14)
    ax.set_title("2D Scatter: Top 2 Features"); ax.legend(markerscale=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scatter_2feat.png"), dpi=300); plt.close(fig)


def scatter_3f(df, feats, classes, outdir):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for c in classes:
        m = df["Class"] == c
        ax.scatter(df.loc[m, feats[0]], df.loc[m, feats[1]], df.loc[m, feats[2]],
                   alpha=0.3, s=8, label=c)
    ax.set_xlabel(feats[0]); ax.set_ylabel(feats[1]); ax.set_zlabel(feats[2])
    ax.set_title("3D Scatter: Top 3 Features"); ax.legend(markerscale=3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "scatter_3feat.png"), dpi=300); plt.close(fig)


def pca_2d(X_sel, y, classes, le, k, outdir):
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X_sel)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in classes:
        m = y == le.transform([c])[0]
        ax.scatter(X2d[m, 0], X2d[m, 1], alpha=0.3, s=8, label=c)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA Projection ({k} features)"); ax.legend(markerscale=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"pca_{k}feat.png"), dpi=300); plt.close(fig)
