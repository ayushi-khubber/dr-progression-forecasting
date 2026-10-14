
import csv
import sys
from pathlib import Path

import pandas as pd

METRICS = [
    ("Sens", "Sensitivity"),
    ("Spec", "Specificity"),
    ("Prec", "Precision"),
    ("F1", "F1"),
    ("BalAcc", "Balanced Acc."),
    ("AUROC", "AUROC"),
    ("AUPRC", "AUPRC"),
]

DECIMALS = 3

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    metric_row = rows[0]   # AUROC, AUROC, AUPRC, AUPRC, ...
    stat_row = rows[1]     # mean, std, mean, std, ...

    # Map (metric, stat) -> column index
    col = {}
    for i, (m, s) in enumerate(zip(metric_row, stat_row)):
        if m and s:
            col[(m.strip(), s.strip())] = i

    # Data rows = rows with a model name and at least one number
    data = []
    for r in rows[2:]:
        if len(r) < 3 or not r[0].strip() or not r[2].strip():
            continue  # skips the "model,optimiser" label row and blanks
        data.append(r)
    return col, data


def build_table(col, data):
    out = []
    for r in data:
        row = {"Model": r[0].strip(), "Optimiser": r[1].strip()}
        for csv_name, report_name in METRICS:
            mean = float(r[col[(csv_name, "mean")]])
            std = float(r[col[(csv_name, "std")]])
            row[report_name] = f"{mean:.{DECIMALS}f} ± {std:.{DECIMALS}f}"
        out.append(row)
    return pd.DataFrame(out)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_summary.csv")
    if not csv_path.exists():
        sys.exit(f"Could not find {csv_path}. Put it next to this script or pass the path.")

    col, data = load(csv_path)
    table = build_table(col, data)

    print(table.to_string(index=False))

    out_dir = csv_path.parent
    table.to_csv(out_dir / "results_table.csv", index=False)

    # Markdown (no extra packages needed)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in table.values]
    (out_dir / "results_table.md").write_text("\n".join([header, sep] + body), encoding="utf-8")

    try:
        table.to_excel(out_dir / "results_table.xlsx", index=False)
        print("\nSaved: results_table.csv, results_table.md, results_table.xlsx")
    except ImportError:
        print("\nSaved: results_table.csv, results_table.md "
              "(install openpyxl for .xlsx: pip install openpyxl)")


if __name__ == "__main__":
    main()
