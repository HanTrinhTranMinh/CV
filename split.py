# split.py
import os
import random
import shutil
from collections import defaultdict

DATASET_DIR = "merged"     # merged/images/<group>/..., merged/labels/<group>/...
OUT_DIR     = "final"
SPLIT       = [0.7, 0.2, 0.1]   # train, val, test
SEED        = 42

# Các nhóm đúng như data.yaml (5 lớp bệnh) + tùy chọn noobj
DISEASE_CLASSES = ["blight", "scab", "spot", "rust", "mildew"]
INCLUDE_NOOBJ   = True
NOOBJ_GROUP     = "noobj"

images_root = os.path.join(DATASET_DIR, "images")
labels_root = os.path.join(DATASET_DIR, "labels")

for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# Thu thập ảnh theo group (cấp 1)
group_to_images = defaultdict(list)

all_groups = []
for g in os.listdir(images_root):
    gpath = os.path.join(images_root, g)
    if not os.path.isdir(gpath):
        continue
    # Chỉ nhận 5 nhóm bệnh trong data.yaml + noobj (nếu bật)
    if g in DISEASE_CLASSES or (INCLUDE_NOOBJ and g == NOOBJ_GROUP):
        all_groups.append(g)

# Duyệt từng group và gom ảnh (đệ quy)
for g in all_groups:
    gpath = os.path.join(images_root, g)
    for root, _, files in os.walk(gpath):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                abspath = os.path.join(root, f)
                # rel_path kể từ images_root (giữ cấu trúc con)
                rel_path = os.path.relpath(abspath, images_root)
                group_to_images[g].append(rel_path)

# Chia tỉ lệ per-group để giữ cân bằng
random.seed(SEED)
splits = {"train": [], "val": [], "test": []}

for g, rel_list in group_to_images.items():
    random.shuffle(rel_list)
    n = len(rel_list)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])

    splits["train"].extend(rel_list[:n_train])
    splits["val"].extend(rel_list[n_train:n_train + n_val])
    splits["test"].extend(rel_list[n_train + n_val:])

# Copy theo split, giữ cấu trúc con
for split, rel_list in splits.items():
    for rel in rel_list:
        # Ảnh nguồn/đích
        src_img = os.path.join(images_root, rel)
        dst_img = os.path.join(OUT_DIR, f"images/{split}", rel)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src_img, dst_img)

        # Nhãn nguồn/đích (cùng relative path, đổi .jpg -> .txt)
        lbl_rel = os.path.splitext(rel)[0] + ".txt"
        src_lbl = os.path.join(labels_root, lbl_rel)
        dst_lbl = os.path.join(OUT_DIR, f"labels/{split}", lbl_rel)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # tạo file rỗng nếu thiếu (YOLO hiểu là no-object)
            open(dst_lbl, "w", encoding="utf-8").close()

print("✅ Đã chia train/val/test theo từng nhóm, giữ cấu trúc con và đồng bộ ảnh–nhãn.")
