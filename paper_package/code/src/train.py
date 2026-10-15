import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

OPTIMISERS = {
    "AdamW": lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2),
    "Adam": lambda p: torch.optim.Adam(p, lr=1e-3),
    "SGD": lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9, weight_decay=1e-4),
}
METRICS = ["AUROC", "AUPRC", "Acc", "Sens", "Spec", "Prec",
           "F1", "BalAcc", "MCC", "Brier"]

def best_thr(y, p):
    """Threshold maximising F1 on the validation set."""
    best_t, best_f = 0.5, -1.0
    for t in np.quantile(p, np.linspace(0.05, 0.95, 91)):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
            Acc=accuracy_score(y, pred),
            Sens=sens, Spec=spec,
            Prec=precision_score(y, pred, zero_division=0),
            F1=f1_score(y, pred, zero_division=0), BalAcc=(sens + spec) / 2,
            MCC=matthews_corrcoef(y, pred), Brier=brier_score_loss(y, p))


class Net(nn.Module):
    """One small branch per input block, concatenated, then a classifier head."""

    def __init__(self, dims, h=128):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.3))
            for d in dims])
        self.head = nn.Sequential(nn.Linear(h * len(dims), 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, xs):
        return self.head(torch.cat([b(x) for b, x in zip(self.branches, xs)], 1)).squeeze(1)


def to_t(arrs):
    return [torch.tensor(a, dtype=torch.float32) for a in arrs]


def fit_predict(Xtr, ytr, Xva, yva, Xte, opt_name, epochs, seed, patience=12):
    torch.manual_seed(seed)
    net = Net([x.shape[1] for x in Xtr])
    opt = OPTIMISERS[opt_name](net.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)       # handles ~16% positives
    tr, va, te = to_t(Xtr), to_t(Xva), to_t(Xte)
    yt = torch.tensor(ytr, dtype=torch.float32)
    n, best, best_state, bad = len(yt), -1.0, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(net([x[idx] for x in tr]), yt[idx]).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            score = average_precision_score(yva, torch.sigmoid(net(va)).numpy())
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(va)).numpy(), torch.sigmoid(net(te)).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(C.OUT / "table.csv", dtype={"pid": str})
    clin = np.load(C.OUT / "clinical.npy")
    eff = np.load(C.OUT / "feat_effnet_b3.npy")
    swin = np.load(C.OUT / "feat_swin_tiny.npy")
    if args.no_grade:
        clin = clin[:, :-1]    
    y, groups = df["label"].values, df["pid"].values
    print(f"{len(df)} eye pairs | {df.pid.nunique()} patients | "
          f"positives {y.mean():.1%} | clinical features: {clin.shape[1]}")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(eff, y, groups)):
        sub = df.iloc[tr_idx]
        pats = sub.groupby("pid")["label"].max().reset_index()
        _, vp = train_test_split(pats, test_size=0.15, stratify=pats["label"],
                                 random_state=C.SEED)
        is_val = sub["pid"].isin(vp["pid"]).values
        va_idx, tr_idx = tr_idx[is_val], tr_idx[~is_val]

        def prep(arr):
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(arr[tr_idx])
            return [pipe.transform(arr[i]).astype(np.float32) for i in (tr_idx, va_idx, te_idx)]

        E, S, K = prep(eff), prep(swin), prep(clin)
        ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(K[0], ytr)
        pv, pt = lr.predict_proba(K[1])[:, 1], lr.predict_proba(K[2])[:, 1]
        rows.append(dict(fold=fold, optimiser="-", model="Clinical-only LR",
                         **metrics(yte, pt, best_thr(yva, pv))))

        for opt_name in OPTIMISERS:
            blocks = {"A: EffNet-B3 + clinical": [E, K],
                      "B: Swin-T + clinical": [S, K],
                      "Fusion: EffNet + Swin + clinical": [E, S, K]}
            probs = {}
            for name, ins in blocks.items():
                probs[name] = fit_predict([i[0] for i in ins], ytr, [i[1] for i in ins],
                                          yva, [i[2] for i in ins], opt_name,
                                          args.epochs, C.SEED + fold)
            a, b = probs["A: EffNet-B3 + clinical"], probs["B: Swin-T + clinical"]
            probs["Ensemble: avg(A, B)"] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for name, (pv, pt) in probs.items():
                rows.append(dict(fold=fold, optimiser=opt_name, model=name,
                                 **metrics(yte, pt, best_thr(yva, pv))))
        print(f"fold {fold + 1}/5 done")

    res = pd.DataFrame(rows)
    res.to_csv(C.OUT / "results_per_fold.csv", index=False)
    summ = res.groupby(["model", "optimiser"])[METRICS].agg(["mean", "std"]).round(3)
    summ.to_csv(C.OUT / "results_summary.csv")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\nChance-level AUPRC = positive rate = {y.mean():.3f}")
    print("\nMean AUPRC by model x optimiser:")
    print(res.pivot_table(index="model", columns="optimiser", values="AUPRC").round(3))
    print("\nMean AUROC by model x optimiser:")
    print(res.pivot_table(index="model", columns="optimiser", values="AUROC").round(3))
    print("\nFull summary (mean/std over 5 folds) saved to outputs/results_summary.csv")


if __name__ == "__main__":
    main()