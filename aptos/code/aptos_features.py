import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT, OUT = Path("data"), Path("outputs")
SIZE = 224
SPLITS = {"train": ("train_1.csv", "train_images/train_images"),
          "valid": ("valid.csv", "val_images/val_images"),
          "test": ("test.csv", "test_images/test_images")}
BACKBONES = {"effnet_b3": "efficientnet_b3", "swin_tiny": "swin_tiny_patch4_window7_224"}
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
TF = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])


def crop_black_border(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def preprocess(path):
    """Same steps as the forecasting project: crop border, resize, CLAHE on LAB lightness."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = crop_black_border(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class ImgDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(image=preprocess(self.paths[i]))["image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only use the first N images per split (test run)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = OUT / "smoke" if a.limit else OUT
    out.mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE)
    models = {k: timm.create_model(v, pretrained=True, num_classes=0).eval().to(DEVICE)
              for k, v in BACKBONES.items()}
    for split, (csv, folder) in SPLITS.items():
        files = [out / f"feat_{k}_{split}.npy" for k in models] + [out / f"labels_{split}.npy"]
        if all(f.exists() for f in files):
            print(f"{split}: already done, skipping")
            continue
        df = pd.read_csv(ROOT / csv).dropna(subset=["id_code"])
        if a.limit:
            df = df.head(a.limit)
        paths = [ROOT / folder / f"{i}.png" for i in df.id_code]
        loader = DataLoader(ImgDS(paths), batch_size=a.batch, shuffle=False, num_workers=a.workers)
        feats = {k: [] for k in models}
        with torch.no_grad():
            for x in tqdm(loader, desc=split):
                x = x.to(DEVICE)
                for k, m in models.items():
                    feats[k].append(m(x).cpu().numpy())
        for k in models:
            np.save(out / f"feat_{k}_{split}.npy", np.concatenate(feats[k]))
        np.save(out / f"labels_{split}.npy", df.diagnosis.values.astype(int))
        print(f"{split}: saved {len(df)} images")


if __name__ == "__main__":
    main()
