import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import config as C
from .data import norm_id, find_col, _grade, _progression

xl = pd.ExcelFile(C.ROOT / "Organized_Data of Patients.xlsx")
base_df = xl.parse([s for s in xl.sheet_names if "base" in s.lower()][0])
fu_df = xl.parse([s for s in xl.sheet_names if "follow" in s.lower()][0])

print("Follow-up columns:", [str(c) for c in fu_df.columns])
raw = [c for c in fu_df.columns if "progression" in str(c).lower()][0]
print(f"\nRaw '{raw}' values:\n", fu_df[raw].value_counts(dropna=False))

b = pd.DataFrame({"pid": base_df[find_col(base_df, "id")].map(norm_id),
                  "g0": _grade(base_df)}).drop_duplicates("pid")
f = pd.DataFrame({"pid": fu_df[find_col(fu_df, "id")].map(norm_id),
                  "prog": _progression(fu_df)})
try:
    f["g1"] = _grade(fu_df).values
except KeyError:
    f["g1"] = float("nan")
    print("\nNo grade column in follow-up sheet: cannot compute grade-based label.")
f = f.drop_duplicates("pid")

m = b.merge(f, on="pid")
m = m.dropna(subset=["prog"])
m = m[m.g0.between(1, 5) & m.g1.between(1, 5)].copy()
m["worse"] = (m.g1 > m.g0).astype(int)

print(f"\nPatients with both grades valid: {len(m)}")
print(f"Progression=Yes rate : {m.prog.mean():.1%}")
print(f"Grade-worsened rate  : {m.worse.mean():.1%}")
print("\nProgression (rows) vs grade worsened (cols):")
print(pd.crosstab(m.prog, m.worse, margins=True))
print(f"\nCohen's kappa: {cohen_kappa_score(m.prog, m.worse):.3f}")
print("\nBaseline grade (rows) vs follow-up grade (cols):")
print(pd.crosstab(m.g0, m.g1))
