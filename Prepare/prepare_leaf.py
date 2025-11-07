import os
import shutil

DATASET_DIR = "Dataset"
OUT_DIR = "binary_leaf_dataset"

for sub in ["leaf", "background"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

for root, _, files in os.walk(DATASET_DIR):
    folder = os.path.basename(root).lower()
    rel = os.path.relpath(root, DATASET_DIR)  # ✅ giữ nguyên cấu trúc con

    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src = os.path.join(root, f)
        # nếu là thư mục nền → copy vào background/giữ nguyên nhánh
        if "background" in folder:
            dst = os.path.join(OUT_DIR, "background", rel, f)
        else:
            # các thư mục bệnh/healthy… giữ nguyên nhánh dưới leaf/
            dst = os.path.join(OUT_DIR, "leaf", rel, f)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

print("Leaf-vs-Background dataset created (giữ nguyên cấu trúc thư mục)!")
