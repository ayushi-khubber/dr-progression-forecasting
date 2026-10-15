import numpy as np, pandas as pd, cv2
from . import config as C
from .data import preprocess

df = pd.read_csv("outputs/table.csv", dtype={"pid": str})
rows = []
for sel in (df.grade_base == 1, df.grade_base >= 3):
    s = df[sel].sample(6, random_state=0)
    rows.append(np.hstack([preprocess(C.ROOT / p) for p in s.baseline_path]))
cv2.imwrite("outputs/montage.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
print("saved outputs/montage.png")