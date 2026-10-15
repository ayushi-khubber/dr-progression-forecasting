import numpy as np, pandas as pd, timm, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from . import config as C
from .data import eval_tf, preprocess

SIZE = 448
df = pd.read_csv(C.OUT / "table.csv", dtype={"pid": str})
f = C.OUT / f"feat_effnet_b3_{SIZE}.npy"
if f.exists():
    X = np.load(f)
else:
    m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0).eval().to(C.DEVICE)
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), 8)):
            b = torch.stack([eval_tf(image=preprocess(C.ROOT / p, size=SIZE))["image"]
                             for p in df.baseline_path[i:i + 8]])
            out.append(m(b.to(C.DEVICE)).cpu().numpy())
    X = np.concatenate(out); np.save(f, X)

g = df.grade_base.values
tasks = {"grade>=2 vs 1": (np.ones(len(df), bool), (g >= 2).astype(int)),
         "grade>=3 vs 1": (g != 2, (g >= 3).astype(int))}
for name, (mask, y) in tasks.items():
    idx = np.where(mask)[0]; yy = y[idx]; grp = df.pid.values[idx]; P = np.zeros(len(idx))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(idx, yy, grp):
        mdl = make_pipeline(StandardScaler(), PCA(32, random_state=0),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        mdl.fit(X[idx][tr], yy[tr]); P[te] = mdl.predict_proba(X[idx][te])[:, 1]
    s = pd.Series(P).groupby(grp).max(); lab = pd.Series(yy).groupby(grp).max()
    print(f"{name:15s} size={SIZE} patient AUROC={roc_auc_score(lab, s):.3f}")