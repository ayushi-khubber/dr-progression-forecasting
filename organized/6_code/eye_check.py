from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT, OUT = Path("data"), Path("outputs")

def norm_id(x):
    return str(x).strip().split(".")[0].lstrip("0")

df = pd.read_csv(OUT / "table.csv", dtype={"pid": str})
xl = pd.ExcelFile(ROOT / "Organized_Data of Patients.xlsx")


def eyes(sheet_kw, suffix):
    d = xl.parse([s for s in xl.sheet_names if sheet_kw in s.lower()][0])
    idc = [c for c in d.columns if str(c).strip().lower() == "id"][0]
    out = pd.DataFrame({"pid": d[idc].map(norm_id)})
    for eye in ("OD", "OS"):
        c = [c for c in d.columns if str(c).lower().startswith(f"{eye.lower()} grade")][0]
        g = pd.to_numeric(d[c], errors="coerce")
        out[f"{eye}_{suffix}"] = g.where(g.between(1, 5))  
    return out.drop_duplicates("pid")


e = eyes("base", "b").merge(eyes("follow", "f"), on="pid")
e = e[e.pid.isin(df.pid)].copy()
for eye in ("OD", "OS"):
    ok = e[f"{eye}_b"].notna() & e[f"{eye}_f"].notna()
    e[f"{eye}_prog"] = np.where(ok, (e[f"{eye}_f"] > e[f"{eye}_b"]).astype(float), np.nan)

both = e.dropna(subset=["OD_prog", "OS_prog"])
print(f"\n=== Per-eye outcomes ({len(e)} patients, {len(both)} with both eyes gradable) ===")
print(pd.crosstab(both.OD_prog, both.OS_prog, rownames=["OD worsened"], colnames=["OS worsened"]))
amb = both.OD_prog != both.OS_prog
print(f"Patients where exactly ONE eye worsened (eye identity matters): {int(amb.sum())}")
print(f"Patients where BOTH eyes worsened: {int(((both.OD_prog == 1) & (both.OS_prog == 1)).sum())}")
print(f"Patients where NEITHER worsened:   {int(((both.OD_prog == 0) & (both.OS_prog == 0)).sum())}")

def disc_x(path):
    """Horizontal position (0=left edge, 1=right edge of retina) of the brightest
    blob in the red channel, which is normally the optic disc."""
    img = cv2.imread(str(ROOT / path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return np.nan
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0 or xs.max() - xs.min() < 20:
        return np.nan
    x0, x1 = int(xs.min()), int(xs.max())
    k = max(3, int((x1 - x0) // 8)) | 1
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    red = cv2.GaussianBlur(img[..., 2], (0, 0), (x1 - x0) / 25).astype(np.float32)
    red[inner == 0] = 0
    cx = cv2.minMaxLoc(red)[3][0]
    return (cx - x0) / (x1 - x0)


df["disc_x_base"] = [disc_x(p) for p in tqdm(df["baseline_path"], desc="baseline images")]
df["disc_x_fu"] = [disc_x(p) for p in tqdm(df["followup_path"], desc="follow-up images")]
df[["key", "pid", "baseline_index", "disc_x_base", "disc_x_fu"]].to_csv(
    OUT / "eye_side.csv", index=False)

rows = []
for pid, g in df.groupby("pid"):
    if len(g) == 2 and g["disc_x_base"].notna().all():
        g = g.sort_values("baseline_index")
        rows.append((pid, *g["disc_x_base"].values))
p = pd.DataFrame(rows, columns=["pid", "idx0", "idx1"])
opp = (p.idx0 - .5) * (p.idx1 - .5) < 0
print(f"\n=== Laterality from disc position ({len(p)} two-image patients) ===")
print(f"Two eyes have the disc on OPPOSITE sides: {opp.mean():.1%}   (want > ~90%)")
print(f"Among those, baseline_index 0 has disc on the RIGHT: {(p[opp].idx0 > .5).mean():.1%}"
      "   (want near 0% or 100% = index maps consistently to one eye)")
v = df.disc_x_base.notna() & df.disc_x_fu.notna()
same = ((df.disc_x_base - .5) * (df.disc_x_fu - .5) > 0)[v].mean()
print(f"Baseline and paired follow-up image on the SAME side: {same:.1%}   (want > ~90%)")
print("\nSaved outputs/eye_side.csv")
