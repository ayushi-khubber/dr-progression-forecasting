from pathlib import Path
import cv2, numpy as np, pandas as pd
from tqdm import tqdm

ROOT, OUT = Path("data"), Path("outputs")
man = pd.read_csv(ROOT / "corrected_manifest.csv")
rows = []
for p in tqdm(man["baseline_path"].unique()):
    img = cv2.imread(str(ROOT / p))
    if img is None:
        rows.append(dict(baseline_path=p, blur=np.nan, bright=np.nan, retina_frac=np.nan))
        continue
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (512, 512))
    mask = gray > 15
    rows.append(dict(
        baseline_path=p,
        blur=cv2.Laplacian(gray, cv2.CV_64F)[mask].var(),   
        bright=gray[mask].mean() if mask.any() else np.nan,
        retina_frac=mask.mean()))             

q = pd.DataFrame(rows)
q["bad"] = (q.blur.isna()
            | (q.blur < q.blur.quantile(0.05))
            | (q.bright < q.bright.quantile(0.02))
            | (q.bright > q.bright.quantile(0.98))
            | (q.retina_frac < 0.40)
            | (q.retina_frac > 0.60))
q.to_csv(OUT / "quality.csv", index=False)
print(q.describe().round(2))
print(f"Flagged {q.bad.sum()} / {len(q)} images")