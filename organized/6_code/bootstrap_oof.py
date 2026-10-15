import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

d = pd.read_csv(sys.argv[1], dtype={"pid": str})
pt = d.groupby("pid").agg(label=("label", "max"), prob=("prob", "mean")).reset_index()
y, s = pt.label.values, pt.prob.values

fpr, tpr, thr = roc_curve(y, s)
t = thr[np.argmax(tpr - fpr)]   

def stats(y, s):
    pred = s >= t
    tp = (pred & (y == 1)).sum(); fn = (~pred & (y == 1)).sum()
    tn = (~pred & (y == 0)).sum(); fp = (pred & (y == 0)).sum()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return dict(AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s),
                Sens=sens, Spec=spec, Prec=prec, F1=f1, Acc=(tp + tn) / len(y))

point = stats(y, s)
rng, boots = np.random.default_rng(0), []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(i):
        boots.append(stats(y[i], s[i]))
b = pd.DataFrame(boots)
print(f"{len(y)} patients, {int(y.sum())} positive, threshold {t:.3f}")
for k, v in point.items():
    lo, hi = b[k].quantile([0.025, 0.975])
    print(f"{k:6s} {v:.3f}  (95% CI {lo:.3f}-{hi:.3f})")