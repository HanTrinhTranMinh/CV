# Prepare/pipeline.py
import subprocess
import sys
import os
from pathlib import Path
import time

# === Luôn chạy từ thư mục chứa pipeline.py ===
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent       # thư mục Dataset/

def run(script: Path):
    print("\n" + "="*60)
    print(f"Running: {script.relative_to(ROOT)}")
    print("="*60)
    t0 = time.time()
    # Dùng Python hiện tại để chạy
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    # In log ra màn hình
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:\n" + result.stderr)
    print(f"Done in {time.time()-t0:.1f}s")

def exists_any(paths):
    return any(p.exists() for p in paths)

def main():
    # 1) Tạo dataset Leaf vs Background
    leaf_ds = ROOT / "binary_leaf_dataset"
    if not leaf_ds.exists() or not any(leaf_ds.rglob("*.*")):
        run(BASE_DIR / "prepare_leaf.py")
    else:
        print("Skip prepare_leaf.py (binary_leaf_dataset/ đã có)")

    # 2) Tạo dataset Healthy vs Diseased
    heal_ds = ROOT / "binary_health_dataset"
    if not heal_ds.exists() or not any(heal_ds.rglob("*.*")):
        run(BASE_DIR / "prepare_heal.py")
    else:
        print("Skip prepare_heal.py (binary_health_dataset/ đã có)")

    # 3) Train 2 model CNN
    leaf_h5 = ROOT / "leaf_or_background.h5"
    dise_h5 = ROOT / "healthy_or_diseased.h5"
    if not exists_any([leaf_h5, dise_h5]):
        run(BASE_DIR / "model1.py")
    else:
        print("Skip model1.py (.h5 đã tồn tại)")

    print("\nPREPARE PIPELINE FINISHED!")
    print("Tiếp tục chạy:")
    print("   1) python model.py     # train YOLO segmentation")
    print("   2) streamlit run app.py")

if __name__ == "__main__":
    main()
