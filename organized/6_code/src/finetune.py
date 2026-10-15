import argparse
import copy

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config as C
from .data import eval_tf, preprocess, train_tf

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

class DS(Dataset):
    def __init__(self, idx, imgs, clin, y, train):
        self.idx, self.imgs, self.clin, self.y = idx, imgs, clin, y
        self.tf = train_tf if train else eval_tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (self.tf(image=self.imgs[j])["image"], torch.from_numpy(self.clin[j]),
                torch.tensor(self.y[j], dtype=torch.float32))


class FTNet(nn.Module):
    def __init__(self, name, n_clin):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        self.clin = nn.Sequential(nn.Linear(n_clin, 32), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Dropout(0.3),
                                  nn.Linear(self.backbone.num_features + 32, 1))

    def forward(self, img, c):
        return self.head(torch.cat([self.backbone(img), self.clin(c)], 1)).squeeze(1)


@torch.no_grad()
def predict(net, loader):
    net.eval()
    out = []
    for img, c, _ in loader:
        out.append(torch.sigmoid(net(img.to(C.DEVICE), c.to(C.DEVICE))).cpu().numpy())
    return np.concatenate(out)


def run_fold(args, name, imgs, clin, y, tr, va, te):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(clin[tr])
    clin_f = pipe.transform(clin).astype(np.float32)
    mk = lambda idx, train: DataLoader(DS(idx, imgs, clin_f, y, train), batch_size=C.BATCH_SIZE,
                                       shuffle=train, drop_last=train, num_workers=C.NUM_WORKERS)
    ltr, lva, lte = mk(tr, True), mk(va, False), mk(te, False)

    net = FTNet(name, clin.shape[1]).to(C.DEVICE)
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": args.lr},          # small LR: pretrained
        {"params": list(net.clin.parameters()) + list(net.head.parameters()), "lr": 1e-3},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_w = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(C.DEVICE))

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        net.train()
        for img, c, t in tqdm(ltr, desc=f"ep{ep + 1}", leave=False):
            opt.zero_grad()
            loss_fn(net(img.to(C.DEVICE), c.to(C.DEVICE)), t.to(C.DEVICE)).backward()
            opt.step()
        sched.step()
        score = average_precision_score(y[va], predict(net, lva))
        print(f"  epoch {ep + 1}: val AUPRC {score:.3f}")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    return predict(net, lte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(C.BACKBONES), default="swin_tiny")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone LR")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=5)
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--task", choices=["forecast", "detect"], default="forecast")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    df = pd.read_csv(C.OUT / "table.csv", dtype={"pid": str})
    clin = np.load(C.OUT / "clinical.npy")
    if args.task == "detect":
        clin = clin[:, :-1]            
        df["label"] = (df["grade_base"] >= 2).astype(int)
    elif args.no_grade:
        clin = clin[:, :-1]
    y, groups = df["label"].values.astype(int), df["pid"].values
    agg = "max" if args.task == "detect" else "mean"   
    print("Caching preprocessed images ...")
    imgs = np.stack([preprocess(C.ROOT / p, size=args.size) for p in tqdm(df["baseline_path"])])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)  
    oof = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(df)), y, groups)):
        if fold >= args.max_folds:
            break
        torch.manual_seed(C.SEED + fold)
        sub = df.iloc[tr_idx]
        pats = sub.groupby("pid")["label"].max().reset_index()
        _, vp = train_test_split(pats, test_size=0.15, stratify=pats["label"], random_state=C.SEED)
        is_val = sub["pid"].isin(vp["pid"]).values
        va_idx, tr_idx = tr_idx[is_val], tr_idx[~is_val]

        print(f"Fold {fold + 1}")
        p = run_fold(args, C.BACKBONES[args.backbone], imgs, clin, y, tr_idx, va_idx, te_idx)
        oof.append(pd.DataFrame({"key": df["key"].values[te_idx], "pid": groups[te_idx],
                                 "fold": fold, "label": y[te_idx], "prob": p}))
        o = oof[-1]
        pt = o.groupby("pid").agg(label=("label", "max"), prob=("prob", agg))
        print(f"  TEST eye-level AUROC {roc_auc_score(o.label, o.prob):.3f} | "
              f"patient-level AUROC {roc_auc_score(pt.label, pt.prob):.3f}, "
              f"AUPRC {average_precision_score(pt.label, pt.prob):.3f}")

    res = pd.concat(oof)
    tag = f"{args.backbone}_{args.task}{args.size}{'_nograde' if args.no_grade else ''}"
    res.to_csv(C.OUT / f"oof_finetune_{tag}.csv", index=False)
    pt = res.groupby("pid").agg(label=("label", "max"), prob=("prob", agg))
    print(f"\nPooled OOF patient-level AUROC {roc_auc_score(pt.label, pt.prob):.3f} | "
          f"AUPRC {average_precision_score(pt.label, pt.prob):.3f} | chance {pt.label.mean():.3f}")


if __name__ == "__main__":
    main()
