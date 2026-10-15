import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .train import METRICS, OPTIMISERS, best_thr, fit_predict, metrics

df = pd.read_csv(C.OUT / "table.csv", dtype={"pid": str})
clin = np.load(C.OUT / "clinical.npy")[:, :-1]          # drop grade_base (it IS the label)
eff = np.load(C.OUT / "feat_effnet_b3.npy")
swin = np.load(C.OUT / "feat_swin_tiny.npy")
y = (df["grade_base"].values >= 2).astype(int)
groups = df["pid"].values
print(f"{len(df)} eyes | {df.pid.nunique()} patients | DR present {y.mean():.1%}")

skf = StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED)
rows, oof = [], {}
for fold, (tr, te) in enumerate(skf.split(eff, y, groups)):
    sub = df.iloc[tr]
    pats = pd.DataFrame({"pid": sub.pid.values, "y": y[tr]}).groupby("pid")["y"].max().reset_index()
    _, vp = train_test_split(pats, test_size=0.15, stratify=pats.y, random_state=C.SEED)
    is_val = sub.pid.isin(vp.pid).values
    va, tr = tr[is_val], tr[~is_val]

    def prep(a):
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(a[tr])
        return [p.transform(a[i]).astype(np.float32) for i in (tr, va, te)]

    E, S, K = prep(eff), prep(swin), prep(clin)
    ytr, yva, yte = y[tr], y[va], y[te]

    def record(name, opt, pv, pt):
        thr = best_thr(yva, pv)
        rows.append(dict(fold=fold, optimiser=opt, model=name,
                         Acc=accuracy_score(yte, (pt >= thr).astype(int)),
                         **metrics(yte, pt, thr)))
        oof.setdefault((name, opt), np.zeros(len(y)))[te] = pt

    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(K[0], ytr)
    record("Clinical-only LR (no grade)", "-",
           m.predict_proba(K[1])[:, 1], m.predict_proba(K[2])[:, 1])

    for opt in OPTIMISERS:
        blocks = {"EffNet-B3": [E], "Swin-T": [S],
                  "EffNet+Swin": [E, S], "EffNet+Swin+clinical": [E, S, K]}
        probs = {n: fit_predict([b[0] for b in ins], ytr, [b[1] for b in ins], yva,
                                [b[2] for b in ins], opt, 60, C.SEED + fold)
                 for n, ins in blocks.items()}
        a, b = probs["EffNet-B3"], probs["Swin-T"]
        probs["Ensemble avg(EffNet, Swin)"] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        for n, (pv, pt) in probs.items():
            record(n, opt, pv, pt)
    print(f"fold {fold + 1}/5 done")

res = pd.DataFrame(rows)
res.to_csv(C.OUT / "detect_per_fold.csv", index=False)
cols = ["Acc"] + METRICS
summ = res.groupby(["model", "optimiser"])[cols].agg(["mean", "std"]).round(3)
summ.to_csv(C.OUT / "detect_summary.csv")
print(f"\nMajority-class accuracy = {1 - y.mean():.3f}   chance AUPRC = {y.mean():.3f}")
print(res.pivot_table(index="model", columns="optimiser", values="AUROC").round(3))

# patient-level AUROC (average both eyes)
print("\nPatient-level AUROC / AUPRC:")
yy = pd.Series(y).groupby(groups).max()
for (n, o), p in oof.items():
    s = pd.Series(p).groupby(groups).mean()
    print(f"{n:30s} {o:6s} {roc_auc_score(yy, s):.3f} / {average_precision_score(yy, s):.3f}")