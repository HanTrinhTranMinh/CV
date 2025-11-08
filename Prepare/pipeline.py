# Prepare/pipeline.py
import subprocess, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "Dataset"

def run(script: Path):
    print("\n" + "="*70)
    print(f"▶ Running: {script.name}")
    print("="*70)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=BASE_DIR)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:\n" + result.stderr)
    print(f"⏱️ Done in {time.time()-t0:.1f}s")

def exists_any(paths): return any(p.exists() for p in paths)

def main():
    leaf_ds = ROOT / "binary_leaf_dataset"
    heal_ds = ROOT / "binary_health_dataset"
    leaf_h5 = ROOT / "leaf_or_background.h5"
    dise_h5 = ROOT / "healthy_or_diseased.h5"

    if not leaf_ds.exists() or not any(leaf_ds.rglob("*.*")):
        run(BASE_DIR / "prepare_leaf.py")
    else:
        print("✅ Skip prepare_leaf (binary_leaf_dataset already exists)")

    if not heal_ds.exists() or not any(heal_ds.rglob("*.*")):
        run(BASE_DIR / "prepare_heal.py")
    else:
        print("✅ Skip prepare_heal (binary_health_dataset already exists)")

    if not exists_any([leaf_h5, dise_h5]):
        run(BASE_DIR / "model1.py")
    else:
        print("✅ Skip model1.py (.h5 models already exist)")

    print("\n🎯 PIPELINE FINISHED!")
    print("Next steps:")
    print("   1️⃣ python model.py      # Train YOLO segmentation")
    print("   2️⃣ streamlit run app.py")

if __name__ == "__main__":
    main()
