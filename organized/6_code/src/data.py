import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config as C

def norm_id(x):
    return str(x).strip().split(".")[0].lstrip("0")


def find_col(df, key):
    cols = [str(c) for c in df.columns]
    for c in cols:                               
        if c.strip().lower() == key.lower():
            return c
    hits = [c for c in cols if key.lower() in c.lower()]
    if not hits:
        raise KeyError(f"No column containing '{key}'. Columns: {cols}")
    return hits[0]

def _grade(df):
    worse = [c for c in df.columns if "worse" in str(c).lower()]
    if worse:
        return pd.to_numeric(df[worse[0]], errors="coerce")
    eyes = [c for c in df.columns if str(c).lower().startswith(("os grade", "od grade"))]
    if not eyes:
        raise KeyError(f"No grade column found. Columns: {list(df.columns)}")
    g = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in eyes], axis=1)
    return g.where(g <= 5).max(axis=1)

def _progression(df):
    hits = [c for c in df.columns if "progression" in str(c).lower()]
    if not hits:
        raise KeyError(f"No progression column. Columns: {list(df.columns)}")
    p = pd.to_numeric(df[hits[0]], errors="coerce")
    return p.map({1: 0.0, 2: 1.0})   # 1=No -> 0, 2=Yes -> 1, anything else -> NaN

def _clean_sheet(df, clinical=True, use_progression=False):
    out = pd.DataFrame({"pid": df[find_col(df, "id")].map(norm_id)})
    if use_progression:
        out["grade"] = _progression(df)
    else:
        out["grade"] = _grade(df)
    if clinical:
        for k in ("age", "sex", "hba1c", "glucose", "hyp"):
            out[k] = df[find_col(df, C.COL_KEYS[k])].values
    return out.drop_duplicates("pid")

def build_table():
    man = pd.read_csv(C.ROOT / "corrected_manifest.csv")
    man["pid"] = man["patient_id"].map(norm_id)

    xl = pd.ExcelFile(C.ROOT / "Organized_Data of Patients.xlsx")
    print("Sheets:", xl.sheet_names)
    base = _clean_sheet(xl.parse([s for s in xl.sheet_names if "base" in s.lower()][0]))
    fu = _clean_sheet(xl.parse([s for s in xl.sheet_names if "follow" in s.lower()][0]),
                      clinical=False, use_progression=True)

    n0 = len(man)
    df = man.merge(base, on="pid", how="inner")
    df = df.merge(fu[["pid", "grade"]].rename(columns={"grade": "label"}),
                  on="pid", how="inner")
    df = df.rename(columns={"grade": "grade_base"})
    print(f"Matched {len(df)}/{n0} manifest rows to clinical data")

    ok = df["grade_base"].between(1, 5) & df["label"].isin([0.0, 1.0])
    print(f"Dropped {(~ok).sum()} rows with grade 6/7 or missing label")
    df = df[ok].copy()

    ok_file = df["baseline_path"].map(lambda p: (C.ROOT / p).exists())
    if (~ok_file).any():
        print(f"Dropped {(~ok_file).sum()} rows with a missing baseline image file")
    df = df[ok_file].copy()

    qpath = C.OUT / "quality.csv"
    if qpath.exists():
        bad = set(pd.read_csv(qpath).query("bad")["baseline_path"])
        ok_q = ~df["baseline_path"].isin(bad)
        print(f"Excluded {(~ok_q).sum()} low-quality images")
        if (~ok_q).mean() > 0.2:
            raise ValueError(f"Quality filter would drop {(~ok_q).mean():.0%} of images - check quality.csv")
        df = df[ok_q].copy()

    print(df.groupby("grade_base")["label"].agg(["count", "mean"]))
    print(f"{len(df)} eye pairs | {df.pid.nunique()} patients | "
          f"progression rate {df.label.mean():.1%} (dataset card says 19.2%)")
    return df.reset_index(drop=True)

def make_splits(df, seed=C.SEED):
    pat = df.groupby("pid")["label"].max().reset_index()  
    tr, tmp = train_test_split(pat, test_size=0.2, stratify=pat.label, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=seed)
    m = {**{p: "train" for p in tr.pid}, **{p: "val" for p in va.pid},
         **{p: "test" for p in te.pid}}
    df = df.copy()
    df["split"] = df["pid"].map(m)
    return df

def prepare_clinical(df):
    X = pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": (pd.to_numeric(df["sex"], errors="coerce") == 1).astype(float),  
        "hba1c": pd.to_numeric(df["hba1c"], errors="coerce"),
        "glucose": pd.to_numeric(df["glucose"], errors="coerce"),
        "hyp": pd.to_numeric(df["hyp"], errors="coerce"),
    })
    if C.INCLUDE_BASELINE_GRADE:
        X["grade_base"] = df["grade_base"].astype(float)
    return X.values.astype(np.float32), list(X.columns)

def crop_black_border(img, tol=7):
    """Crop the black circular frame of a fundus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def preprocess(path, size=C.IMG_SIZE):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = crop_black_border(img, tol=7)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

_NORM = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
eval_tf = A.Compose([_NORM, ToTensorV2()])
train_tf = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(limit=15, p=0.5),
                      A.RandomBrightnessContrast(0.1, 0.1, p=0.5), _NORM, ToTensorV2()])

class DRDataset(Dataset):
    def __init__(self, df, clinical, train=False):
        self.df, self.clin = df.reset_index(drop=True), clinical
        self.tf = train_tf if train else eval_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = preprocess(C.ROOT / r["baseline_path"])
        return (self.tf(image=img)["image"], torch.from_numpy(self.clin[i]),
                torch.tensor(r["label"], dtype=torch.float32))
