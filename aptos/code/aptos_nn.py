"""Small MLP heads trained on frozen features (same design as train.py in the forecasting project)."""
import copy

import numpy as np
import torch
import torch.nn as nn

from aptos_metrics import val_score

OPT = {
    "AdamW": lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=1e-2),
    "Adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
    "SGD": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=1e-4),
}


class Net(nn.Module):
    """One small branch per input block, concatenated, then a classifier head."""

    def __init__(self, dims, n_out, h=128):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.3))
            for d in dims])
        self.head = nn.Sequential(nn.Linear(h * len(dims), 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, xs):
        out = self.head(torch.cat([b(x) for b, x in zip(self.branches, xs)], 1))
        return out.squeeze(1) if out.shape[1] == 1 else out


def _t(arrs):
    return [torch.tensor(a, dtype=torch.float32) for a in arrs]


def fit_predict(Xtr, ytr, Xva, yva, Xte, opt_name, lr, task, epochs, seed, patience=12):
    """Train on Xtr, early-stop on Xva, return (valid probs, test probs)."""
    torch.manual_seed(seed)
    n_out = 1 if task == "binary" else 5
    net = Net([x.shape[1] for x in Xtr], n_out)
    opt = OPT[opt_name](net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if task == "binary":
        pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        yt = torch.tensor(ytr, dtype=torch.float32)
    else:
        w = len(ytr) / (5 * np.maximum(np.bincount(ytr, minlength=5), 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        yt = torch.tensor(ytr, dtype=torch.long)
    tr, va, te = _t(Xtr), _t(Xva), _t(Xte)

    def predict(xs):
        net.eval()
        with torch.no_grad():
            out = net(xs)
            return (torch.sigmoid(out) if task == "binary" else torch.softmax(out, 1)).numpy()

    n, best, best_state, bad = len(yt), -1.0, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(net([x[idx] for x in tr]), yt[idx]).backward()
            opt.step()
        sched.step()
        score = val_score(task, yva, predict(va))
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return predict(va), predict(te)
