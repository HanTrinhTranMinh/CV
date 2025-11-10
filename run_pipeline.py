import os
import subprocess
import time
import shutil
import random
from datetime import datetime

# ==========================================
# ⚙️ CẤU HÌNH CHUNG
# ==========================================
DATASET_DIR = "Dataset"
PREPARED_DIR = "prepared"
YOLO_DIR = "yolo_dataset"
YOLO_LEAF_DIR = "yolo_leaf"

TRAIN_DIR = os.path.join(YOLO_DIR, "train")
VAL_DIR = os.path.join(YOLO_DIR, "val")
TEST_DIR = os.path.join(YOLO_DIR, "test")

DATASET_YAML = "dataset.yaml"
DATASET_LEAF_YAML = "dataset_leaf.yaml"

# Tham số YOLOv8
EPOCHS = 20
IMGSZ = 512
BATCH = 8
MODEL_SEG = "yolov8n-seg.pt"
MODEL_DET = "yolov8n.pt"

# ==========================================
# 🔹 HÀM CHIA DỮ LIỆU (train/val/test)
# ==========================================
def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.2):
    image_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")

    # Tạo thư mục chia tập
    for subset in ["train", "val", "test"]:
        for sub2 in ["images", "labels"]:
            folder = os.path.join(base_dir, subset, sub2)
            os.makedirs(folder, exist_ok=True)
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    # Duyệt từng lớp
    for cls in os.listdir(image_dir):
        cls_img = os.path.join(image_dir, cls)
        cls_lbl = os.path.join(label_dir, cls)
        if not os.path.isdir(cls_img):
            continue

        imgs = [f for f in os.listdir(cls_img) if f.endswith(".jpg")]
        if not imgs:
            continue

        random.shuffle(imgs)
        n = len(imgs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        for i, img_file in enumerate(imgs):
            src_img = os.path.join(cls_img, img_file)
            src_lbl = os.path.join(cls_lbl, img_file.replace(".jpg", ".txt"))

            if i < train_end:
                subset = "train"
            elif i < val_end:
                subset = "val"
            else:
                subset = "test"

            dst_root = os.path.join(base_dir, subset)
            shutil.copy(src_img, os.path.join(dst_root, "images", img_file))
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(dst_root, "labels", img_file.replace(".jpg", ".txt")))

    print(f"✅ Dataset tại '{base_dir}' đã được chia thành train/val/test!")

# ==========================================
# 🧩 BƯỚC 1: TẠO MASK
# ==========================================
print("🧩 [1/8] Tạo mask (prepare_dataset.py)...")
start_time = time.time()
subprocess.run(["python", "prepare_dataset.py"], check=True)
print(f"✅ Hoàn tất tạo mask trong {time.time() - start_time:.1f}s\n")

# ==========================================
# 🧩 BƯỚC 2: SINH LABEL BỆNH
# ==========================================
print("🧩 [2/8] Sinh label YOLO segmentation (generate_yolo_labels.py)...")
subprocess.run(["python", "generate_yolo_labels.py"], check=True)
print("✅ Đã sinh label segmentation!\n")

# ==========================================
# 🧩 BƯỚC 3: SINH LABEL LÁ
# ==========================================
print("🧩 [3/8] Sinh label YOLO leaf detection (yolo_label_leaf.py)...")
subprocess.run(["python", "yolo_label_leaf.py"], check=True)
print("✅ Đã sinh label leaf detection!\n")

# ==========================================
# 🧩 BƯỚC 4: CHIA TRAIN/VAL/TEST
# ==========================================
print("🧩 [4/8] Chia tập train/val/test...")
split_dataset(YOLO_DIR)
split_dataset(YOLO_LEAF_DIR)
print("✅ Hoàn tất chia tập!\n")

# ==========================================
# 🧩 BƯỚC 5: TẠO FILE YAML
# ==========================================
print("🧩 [5/8] Sinh file dataset.yaml và dataset_leaf.yaml...")

yaml_seg = f"""# YOLOv8 Segmentation Dataset
path: {os.path.abspath(YOLO_DIR).replace("\\", "/")}
train: train
val: val
test: test

names:
  0: healthy
  1: black_rot
  2: blight
  3: middew
  4: rust
  5: spot
"""

yaml_leaf = f"""# YOLOv8 Leaf Detection Dataset
path: {os.path.abspath(YOLO_LEAF_DIR).replace("\\", "/")}
train: train/images
val: val/images
test: test/images

names:
  0: background
  1: leaf
"""

with open(DATASET_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_seg)
with open(DATASET_LEAF_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_leaf)

print("✅ Đã tạo file dataset.yaml và dataset_leaf.yaml!\n")

# ==========================================
# 🧩 BƯỚC 6: TRAIN 2 MÔ HÌNH
# ==========================================
def train_yolo(model_type, model, data, name):
    print(f"🚀 Bắt đầu train {model_type}...")
    cmd = [
        "yolo",
        model_type,
        "train",
        f"model={model}",
        f"data={data}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        "verbose=True",
        f"name={name}"
    ]
    print("🔹 Lệnh:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)
    print(f"✅ Hoàn tất train {model_type}!\n")

train_yolo("segment", MODEL_SEG, DATASET_YAML, "train_seg")
train_yolo("detect", MODEL_DET, DATASET_LEAF_YAML, "train_leaf")

# ==========================================
# 🧩 BƯỚC 7: DỰ ĐOÁN KIỂM TRA
# ==========================================
print("🧩 [7/8] Dự đoán kiểm tra mô hình...\n")

test_img_dir = os.path.join(YOLO_DIR, "test", "images")
sample_imgs = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)[:3]]

for img_path in sample_imgs:
    print(f"🔸 Dự đoán trên {img_path}")
    subprocess.run([
        "yolo", "segment", "predict",
        f"model=runs/segment/train_seg/weights/best.pt",
        f"source={img_path}",
        "save=True",
        "conf=0.5"
    ], check=True)

# ==========================================
# 🧩 BƯỚC 8: ĐÁNH GIÁ (EVALUATE)
# ==========================================
print("🧩 [8/8] Chạy evaluate.py để đánh giá mô hình...\n")

if os.path.exists("evaluate.py"):
    subprocess.run(["python", "evaluate.py"], check=True)
    print("✅ Đã chạy evaluate.py thành công!\n")
else:
    print("⚠️ Không tìm thấy evaluate.py — bỏ qua bước đánh giá.\n")

total_time = int(time.time() - start_time)
print("\n🎉 Toàn bộ pipeline hoàn tất!")
print(f"🕒 Thời gian tổng: {total_time}s")
print("📂 Kết quả:")
print("  - Segmentation: runs/segment/train_seg/weights/best.pt")
print("  - Leaf detection: runs/detect/train_leaf/weights/best.pt")
