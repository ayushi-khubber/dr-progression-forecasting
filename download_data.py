"""Run once: downloads the dataset (~2.4 GB) into ./data"""
from huggingface_hub import snapshot_download

snapshot_download(repo_id="usama10/retinal-dr-longitudinal",
                  repo_type="dataset", local_dir="data")
print("Done. Files are in ./data")
