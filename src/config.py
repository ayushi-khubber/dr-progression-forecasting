from pathlib import Path
import torch

ROOT = Path("data")                 
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

SEED = 42
IMG_SIZE = 224                    
BATCH_SIZE = 16
NUM_WORKERS = 0                     
INCLUDE_BASELINE_GRADE = True       

BACKBONES = {
    "effnet_b3":    "efficientnet_b3",
    "swin_tiny":    "swin_tiny_patch4_window7_224",   
}

COL_KEYS = {"age": "age", "sex": "sex", "hba1c": "hba1c",
            "glucose": "glu", "hyp": "blood-pressure"}

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
