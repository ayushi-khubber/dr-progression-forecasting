import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = Path("outputs")
REPEATS, N_BOOT = 5, 1000


def norm_id(x):
    return str(x).strip().split(".")[0].lstrip("0")

df = pd.read_csv(OUT / "table.csv", dtype={"pid": str})
clin = np.load(OUT / "clinical.npy")   
eff = np.load(OUT / "feat_effnet_b3.npy")
swin = np.load(OUT / "feat_swin_tiny.npy")
y = df["label"].values.astype(int)
groups = df["pid"].values
pid_codes, _ = pd.factorize(df["pid"])

xl = pd.ExcelFile("data/Organized_Data of Patients.xlsx")
base = xl.parse([s for s in xl.sheet_names if "base" in s.lower()][0])
base["pid"] = base["ID"].map(norm_id)
drop = [c for c in base.columns
        if "grade" in str(c).lower() or str(c).lower().startswith("id") or c == "pid"]
num = base.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
num = num.loc[:, (num.isna().mean() < 0.4) & (num.nunique() > 1)]
num["pid"] = base["pid"]
ext = (df[["pid"]].merge(num.drop_duplicates("pid"), on="pid", how="left")
       .drop(columns="pid").values.astype(float))
ext = np.hstack([ext, df[["grade_base"]].values.astype(float)])
print(f"{len(df)} eyes | {df.pid.nunique()} patients | "
      f"{int(df.groupby('pid').label.max().sum())} positive patients | "
      f"{ext.shape[1]} extended clinical features")

def lr(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def clin_lr(C=1.0):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr(C))


def img_lr(k=16, C=0.1):
    return lambda: make_pipeline(StandardScaler(), PCA(k, random_state=0), lr(C))


def img_clin_lr(n_img, k=16, C=0.1):
    def make():
        ct = ColumnTransformer([
            ("img", make_pipeline(StandardScaler(), PCA(k, random_state=0)), slice(0, n_img)),
            ("rest", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             slice(n_img, None))])
        return make_pipeline(ct, lr(C))
    return make


models = {
    "Grade only":                 (clin[:, [5]], clin_lr()),
    "Glucose only":               (clin[:, [3]], clin_lr()),
    "Glucose + HbA1c":            (clin[:, [2, 3]], clin_lr()),
    "Clinical (6) LR":            (clin, clin_lr()),
    "Clinical extended LR":       (ext, clin_lr(0.05)),
    "EffNet PCA16 only":          (eff, img_lr()),
    "Swin PCA16 only":            (swin, img_lr()),
    "EffNet PCA16 + clinical(6)": (np.hstack([eff, clin]), img_clin_lr(eff.shape[1])),
    "Swin PCA16 + clinical(6)":   (np.hstack([swin, clin]), img_clin_lr(swin.shape[1])),
    "EffNet PCA16 + clin ext":    (np.hstack([eff, ext]), img_clin_lr(eff.shape[1])),
}


def oof(X, make):
    P = np.zeros((REPEATS, len(y)))
    for r in range(REPEATS):
        skf = StratifiedGroupKFold(5, shuffle=True, random_state=100 + r)
        for tr, te in skf.split(X, y, groups):
            P[r, te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return P


def patient_level(p):
    s = pd.Series(p).groupby(pid_codes).mean().values     
    yy = pd.Series(y).groupby(pid_codes).max().values
    return yy, s


def boot(yy, s, seed=0):
    rng, au, ap = np.random.default_rng(seed), [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yy), len(yy))
        if 0 < yy[i].sum() < len(i):
            au.append(roc_auc_score(yy[i], s[i]))
            ap.append(average_precision_score(yy[i], s[i]))
    return np.percentile(au, [2.5, 97.5]), np.percentile(ap, [2.5, 97.5])


def summarise(name, P):
    yy, s = patient_level(P.mean(0))
    (a_lo, a_hi), (p_lo, p_hi) = boot(yy, s)
    per_rep = [roc_auc_score(*patient_level(P[r])) for r in range(REPEATS)]
    return {"Model": name,
            "AUROC": round(roc_auc_score(yy, s), 3),
            "AUROC 95% CI": f"{a_lo:.2f}-{a_hi:.2f}",
            "AUROC across repeats": f"{np.mean(per_rep):.3f} ± {np.std(per_rep):.3f}",
            "AUPRC": round(average_precision_score(yy, s), 3),
            "AUPRC 95% CI": f"{p_lo:.2f}-{p_hi:.2f}"}, yy.mean()


preds, rows = {}, []
for name, (X, make) in models.items():
    preds[name] = oof(X, make)
    row, prev = summarise(name, preds[name])
    rows.append(row)
    print(f"done: {name}")

ens = (preds["EffNet PCA16 + clinical(6)"] + preds["Swin PCA16 + clinical(6)"]) / 2
rows.append(summarise("Ensemble avg(EffNet, Swin) + clinical(6)", ens)[0])

res = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_columns", 20)
print(f"\nPatient-level results. Chance AUROC = 0.5, chance AUPRC = {prev:.3f}\n")
print(res.to_string(index=False))
res.to_csv(OUT / "robust_results.csv", index=False)
print("\nSaved outputs/robust_results.csv")
