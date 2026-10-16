"""Metric, threshold and bootstrap helpers for the APTOS experiments (no torch needed)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, precision_score,
                             roc_auc_score)


def best_thr(y, p):
    """Threshold that maximises F1 on the validation set."""
    best_t, best_f = 0.5, -1.0
    for t in np.quantile(p, np.linspace(0.05, 0.95, 91)):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(Acc=accuracy_score(y, pred), Prec=precision_score(y, pred, zero_division=0),
                Recall=sens, Spec=spec, F1=f1_score(y, pred, zero_division=0),
                BalAcc=(sens + spec) / 2, AUROC=roc_auc_score(y, p),
                AUPRC=average_precision_score(y, p))


def grade_metrics(y, P):
    pred = P.argmax(1)
    return dict(Acc=accuracy_score(y, pred),
                MacroF1=f1_score(y, pred, average="macro", zero_division=0),
                QWK=cohen_kappa_score(y, pred, weights="quadratic"),
                BalAcc=balanced_accuracy_score(y, pred),
                AUROC_ovr=roc_auc_score(y, P, multi_class="ovr", average="macro"))


def val_score(task, y, p):
    """Score used to pick the learning rate and to early-stop (validation data only)."""
    if task == "binary":
        return average_precision_score(y, p)
    return cohen_kappa_score(y, p.argmax(1), weights="quadratic")


def bootstrap_ci(fn, y, P, n=1000, seed=0):
    """95% bootstrap CI for every metric returned by fn(y, P). Resamples test images."""
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            rows.append(fn(y[i], P[i]))
        except ValueError:
            continue
    b = pd.DataFrame(rows)
    return {k: (b[k].quantile(0.025), b[k].quantile(0.975)) for k in b.columns}


def fmt(v, ci):
    return f"{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"
