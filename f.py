import subprocess
import time
import os

# ==============================
# ⚙️ CẤU HÌNH
# ==============================
MODEL = "yolov8n-seg.pt"            # model gốc (tiny)
DATASET_YAML = "dataset.yaml"       # đường dẫn file cấu hình YOLO
EPOCHS = 20                         # giảm xuống 20
IMGSZ = 512                         # kích thước ảnh
BATCH = 8                           # batch size
NAME = "train_seg20"                # tên folder output

# ==============================
# 🚀 TRAIN YOLO
# ==============================
print("🚀 Bắt đầu train YOLOv8 segmentation...\n")
start = time.time()

# Đảm bảo ultralytics đã có
try:
    import ultralytics
except ImportError:
    print("📦 Cài đặt ultralytics...")
    subprocess.run(["pip", "install", "-U", "ultralytics"], check=True)

# Câu lệnh YOLO CLI
cmd = [
    "yolo",
    "segment",
    "train",
    f"model={MODEL}",
    f"data={DATASET_YAML}",
    f"epochs={EPOCHS}",
    f"imgsz={IMGSZ}",
    f"batch={BATCH}",
    f"name={NAME}",
    "verbose=True"
]

print("🔹 Lệnh YOLO:", " ".join(cmd), "\n")

# Chạy realtime
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(process.stdout.readline, ''):
    print(line, end='')  # hiển thị epoch realtime
process.stdout.close()
process.wait()

# ==============================
# ✅ Hoàn tất
# ==============================
elapsed = int(time.time() - start)
print(f"\n✅ Huấn luyện hoàn tất trong {elapsed}s!")
print(f"📂 Kết quả lưu tại: runs/segment/{NAME}/weights/best.pt")
