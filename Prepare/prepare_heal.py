import os, shutil

DATASET_DIR = "binary_leaf_dataset/leaf"   # ✅ chỉ ảnh lá
OUT_DIR = "binary_health_dataset"

for sub in ["healthy", "diseased"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def is_healthy(folder_name: str) -> bool:
    f = folder_name.lower()
    return f.endswith("__healthy") or "healthy" in f

for root, _, files in os.walk(DATASET_DIR):
    folder = os.path.basename(root)
    rel = os.path.relpath(root, DATASET_DIR)  # giữ cấu trúc con
    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src = os.path.join(root, f)
        sub = "healthy" if is_healthy(folder) else "diseased"
        dst = os.path.join(OUT_DIR, sub, rel, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

print("Healthy-vs-Diseased (leaf-only) dataset created!")
