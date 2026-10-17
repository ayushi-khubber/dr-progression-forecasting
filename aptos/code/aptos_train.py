import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import aptos_nn
from aptos_metrics import (best_thr, binary_metrics, bootstrap_ci, fmt, grade_metrics, val_score)

OUT, SEED = Path("outputs"), 42
GRID = {"AdamW": [1e-4, 5e-4, 1e-3], "Adam": [1e-4, 5e-4, 1e-3], "SGD": [1e-3, 1e-2, 5e-2]}
BLOCKS = {"EffNet-B3": [0], "Swin-T": [1], "Fusion: EffNet+Swin": [0, 1]}


def load(split):
    return [np.load(OUT / f"feat_effnet_b3_{split}.npy"), np.load(OUT / f"feat_swin_tiny_{split}.npy")], \
        np.load(OUT / f"labels_{split}.npy").astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "grade5"], default="binary")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    data = {s: load(s) for s in ("train", "valid", "test")}
    if a.smoke:                                   # tiny, fast run just to check the pipeline works
        rng = np.random.default_rng(0)
        for s, k in (("train", 400), ("valid", 100), ("test", 100)):
            F, y = data[s]; i = rng.choice(len(y), min(k, len(y)), replace=False)
            data[s] = ([f[i] for f in F], y[i])
        a.epochs, a.boot = 3, 50
        grid = {o: l[:1] for o, l in GRID.items()}
    else:
        grid = GRID
    lab = (lambda y: (y >= 1).astype(int)) if a.task == "binary" else (lambda y: y)
    ytr, yva, yte = (lab(data[s][1]) for s in ("train", "valid", "test"))

    # scale each feature block using TRAIN statistics only
    sc = [StandardScaler().fit(f) for f in data["train"][0]]
    X = {s: [sc[j].transform(data[s][0][j]).astype(np.float32) for j in range(2)] for s in data}
    print(f"task={a.task} | train {len(ytr)} valid {len(yva)} test {len(yte)}")
    print(f"test class balance: {np.bincount(yte).tolist()} | majority-class accuracy {np.bincount(yte).max() / len(yte):.3f}")

    runs, grid_rows = {}, []
    for opt, lrs in grid.items():
        for lr in lrs:
            for name, idx in BLOCKS.items():
                pv, pt = aptos_nn.fit_predict([X["train"][i] for i in idx], ytr,
                                              [X["valid"][i] for i in idx], yva,
                                              [X["test"][i] for i in idx], opt, lr,
                                              a.task, a.epochs, SEED)
                runs[(name, opt, lr)] = (pv, pt)
                grid_rows.append(dict(model=name, optimiser=opt, lr=lr, val_score=val_score(a.task, yva, pv)))
            print(f"  done {opt} lr={lr}")
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT / f"aptos_{a.task}_grid.csv", index=False)

    chosen = {}
    for (name, opt), g in grid_df.groupby(["model", "optimiser"]):
        lr = g.loc[g.val_score.idxmax(), "lr"]
        chosen[(name, opt)] = (lr, *runs[(name, opt, lr)])
    for opt in grid:          
        e, s = chosen[("EffNet-B3", opt)], chosen[("Swin-T", opt)]
        chosen[("Ensemble: avg(EffNet, Swin)", opt)] = ("-", (e[1] + s[1]) / 2, (e[2] + s[2]) / 2)

    cat = lambda s: np.hstack(X[s])       
    lr_m = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(cat("train"), ytr)
    pv, pt = lr_m.predict_proba(cat("valid")), lr_m.predict_proba(cat("test"))
    if a.task == "binary":
        pv, pt = pv[:, 1], pt[:, 1]
    chosen[("Linear: LogReg on EffNet+Swin", "-")] = ("-", pv, pt)

    rows = []
    for (name, opt), (lr, pv, pt) in chosen.items():
        if a.task == "binary":
            thr = best_thr(yva, pv)
            fn = lambda y_, p_, thr=thr: binary_metrics(y_, p_, thr)
        else:
            fn = lambda y_, p_: grade_metrics(y_, p_)
        point, ci = fn(yte, pt), bootstrap_ci(fn, yte, pt, n=a.boot)
        row = dict(Model=name, Optimiser=opt, lr=lr)
        row.update({k: fmt(point[k], ci[k]) for k in point})
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["Model", "Optimiser"])
    res.to_csv(OUT / f"aptos_{a.task}_results.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 30)
    print(f"\nTEST results, mean (95% bootstrap CI over test images). Learning rate chosen on validation data.\n")
    print(res.to_string(index=False))
    print(f"\nSaved outputs/aptos_{a.task}_results.csv and aptos_{a.task}_grid.csv")


if __name__ == "__main__":
    main()
