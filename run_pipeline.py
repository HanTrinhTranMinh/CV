import os
import subprocess
import time
import shutil
import random
from datetime import datetime

# ==========================================
# ⚙️ CẤU HÌNH
# ==========================================
DATASET_DIR = "Dataset"
PREPARED_DIR = "prepared"
YOLO_DIR = "yolo_dataset"
TRAIN_DIR = os.path.join(YOLO_DIR, "train")
VAL_DIR = os.path.join(YOLO_DIR, "val")
DATASET_YAML = "dataset.yaml"

# YOLOv8 tham số
EPOCHS = 50
IMGSZ = 512
MODEL = "yolov8n-seg.pt"
BATCH = 8

# ==========================================
# 🧩 BƯỚC 1: TẠO MASK
# ==========================================
print("🧩 [1/5] Bước 1: Tạo mask (prepare_dataset.py)...")
start_time = time.time()
subprocess.run(["python", "prepare_dataset.py"], check=True)
print(f"✅ Hoàn tất tạo mask trong {time.time() - start_time:.1f}s\n")

# ==========================================
# 🧩 BƯỚC 2: SINH LABEL YOLO
# ==========================================
print("🧩 [2/5] Bước 2: Sinh label YOLO (generate_yolo_labels.py)...")
subprocess.run(["python", "generate_yolo_labels.py"], check=True)
print("✅ Đã sinh xong label YOLO!\n")

# ==========================================
# 🧩 BƯỚC 3: CHIA TRAIN / VAL
# ==========================================
print("🧩 [3/5] Bước 3: Chia train/val...")

def split_dataset(base_dir=YOLO_DIR, train_ratio=0.8):
    image_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")

    # Clear và tạo thư mục lại
    for sub in ["train", "val"]:
        for sub2 in ["images", "labels"]:
            folder = os.path.join(base_dir, sub, sub2)
            os.makedirs(folder, exist_ok=True)
            # Xóa dữ liệu cũ nếu có
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    for cls in os.listdir(image_dir):
        cls_img = os.path.join(image_dir, cls)
        cls_lbl = os.path.join(label_dir, cls)
        if not os.path.isdir(cls_img):
            continue

        imgs = [f for f in os.listdir(cls_img) if f.endswith(".jpg")]
        if not imgs:
            continue

        random.shuffle(imgs)
        split_idx = int(len(imgs) * train_ratio)

        for i, img_file in enumerate(imgs):
            src_img = os.path.join(cls_img, img_file)
            src_lbl = os.path.join(cls_lbl, img_file.replace(".jpg", ".txt"))

            dst_root = TRAIN_DIR if i < split_idx else VAL_DIR
            shutil.copy(src_img, os.path.join(dst_root, "images", img_file))
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(dst_root, "labels", img_file.replace(".jpg", ".txt")))

    print("✅ Dataset đã được chia thành train/val!")

split_dataset()
print()

# ==========================================
# 🧩 BƯỚC 4: TẠO FILE DATASET.YAML
# ==========================================
print("🧩 [4/5] Sinh file dataset.yaml...")

yaml_content = f"""# YOLOv8 Segmentation Dataset
path: {os.path.abspath(YOLO_DIR).replace("\\", "/")}
train: train
val: val

names:
  0: healthy
  1: black_rot
  2: blight
  3: middew
  4: rust
  5: spot
"""
with open(DATASET_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("✅ Đã tạo xong dataset.yaml!\n")

# ==========================================
# 🧩 BƯỚC 5: TRAIN YOLO
# ==========================================
print("🧩 [5/5] Bắt đầu train YOLOv8 segmentation...\n")

# Đảm bảo có ultralytics
try:
    import ultralytics
except ImportError:
    print("📦 Cài đặt ultralytics...")
    subprocess.run(["pip", "install", "-U", "ultralytics"], check=True)

# Gọi lệnh train
cmd = [
    "yolo",
    "segment",
    "train",
    f"model={MODEL}",
    f"data={DATASET_YAML}",
    f"epochs={EPOCHS}",
    f"imgsz={IMGSZ}",
    f"batch={BATCH}",
    "verbose=True",
    "name=train"
]

print("🔹 Lệnh YOLO:", " ".join(cmd), "\n")

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(process.stdout.readline, ''):
    print(line, end='')  # In ra từng dòng epoch real-time
process.stdout.close()
process.wait()

total_time = int(time.time() - start_time)
print(f"\n✅ Toàn bộ pipeline hoàn tất!")
print(f"🕒 Thời gian tổng: {total_time}s")
print(f"📂 Kết quả YOLO: runs/segment/train/weights/best.pt")
