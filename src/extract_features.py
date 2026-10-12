"""Step 1: build table + splits, step 2: extract frozen deep features
from each backbone. Run from project root:  python -m src.extract_features
"""
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C
from .data import build_table, make_splits, prepare_clinical, DRDataset


@torch.no_grad()
def extract(model_name, loader):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(C.DEVICE)                   
    feats = []
    for img, _, _ in tqdm(loader, desc=model_name):
        feats.append(model(img.to(C.DEVICE)).cpu().numpy())
    return np.concatenate(feats)


if __name__ == "__main__":
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    df = make_splits(build_table())
    clin, clin_cols = prepare_clinical(df)
    print("Clinical features:", clin_cols)
    print(df.groupby("split")["label"].agg(["count", "mean"]))

    df.to_csv(C.OUT / "table.csv", index=False)
    np.save(C.OUT / "clinical.npy", clin)

    loader = DataLoader(DRDataset(df, clin, train=False), batch_size=C.BATCH_SIZE,
                        shuffle=False, num_workers=C.NUM_WORKERS)
    for short, name in C.BACKBONES.items():
        np.save(C.OUT / f"feat_{short}.npy", extract(name, loader))
    print("Saved features to", C.OUT)
