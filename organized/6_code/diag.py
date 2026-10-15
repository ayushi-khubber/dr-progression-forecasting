import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs")
df = pd.read_csv(OUT / "table.csv", dtype={"pid": str})
feats = {"EffNet": np.load(OUT / "feat_effnet_b3.npy"), "Swin": np.load(OUT / "feat_swin_tiny.npy")}
g = df.grade_base.values
tasks = {"grade>=2 vs 1": (np.ones(len(df), bool), (g >= 2).astype(int)),
         "grade>=3 vs 1 (drop grade 2)": (g != 2, (g >= 3).astype(int))}
for tname, (mask, y) in tasks.items():
    idx = np.where(mask)[0]; yy = y[idx]; groups = df.pid.values[idx]
    for fname, X in feats.items():
        P = np.zeros(len(idx))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(idx, yy, groups):
            m = make_pipeline(StandardScaler(), PCA(32, random_state=0),
                              LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
            m.fit(X[idx][tr], yy[tr]); P[te] = m.predict_proba(X[idx][te])[:, 1]
        s = pd.Series(P).groupby(groups).mean(); lab = pd.Series(yy).groupby(groups).max()
        print(f"{tname:30s} {fname:7s} eyes={len(idx)} pos={yy.mean():.2f} AUROC={roc_auc_score(lab, s):.3f}")