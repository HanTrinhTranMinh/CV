import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "Dataset"
DATASET_DIR = ROOT / "binary_leaf_dataset" / "leaf"
OUT_DIR = ROOT / "binary_health_dataset"

for sub in ["healthy", "diseased"]:
    os.makedirs(OUT_DIR / sub, exist_ok=True)

def is_healthy(name: str) -> bool:
    name = name.lower()
    return name.endswith("__healthy") or "healthy" in name

for root, _, files in os.walk(DATASET_DIR):
    folder = os.path.basename(root)
    rel = Path(root).relative_to(DATASET_DIR)
    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src = Path(root) / f
        sub = "healthy" if is_healthy(folder) else "diseased"
        dst = OUT_DIR / sub / rel / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

print(f"✅ Healthy-vs-Diseased dataset created in: {OUT_DIR}")
