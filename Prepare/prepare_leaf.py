import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "Dataset"
DATASET_DIR = ROOT
OUT_DIR = ROOT / "binary_leaf_dataset"

for sub in ["leaf", "background"]:
    os.makedirs(OUT_DIR / sub, exist_ok=True)

for root, _, files in os.walk(DATASET_DIR):
    folder = os.path.basename(root).lower()
    rel = Path(root).relative_to(DATASET_DIR)  # giữ cấu trúc thư mục

    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src = Path(root) / f
        if "background" in folder:
            dst = OUT_DIR / "background" / rel / f
        else:
            dst = OUT_DIR / "leaf" / rel / f

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

print(f"✅ Leaf-vs-Background dataset created in: {OUT_DIR}")
