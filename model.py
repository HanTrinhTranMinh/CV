# model.py (train YOLOv8 segmentation theo bệnh)
import os
import sys
import subprocess
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def run_script(script_name: str):
    print(f"\n==============================")
    print(f"▶️  Đang chạy: {script_name}")
    print(f"==============================\n")
    result = subprocess.run([sys.executable, script_name],
                            capture_output=True, text=True, cwd=ROOT)
    print(result.stdout)
    if result.stderr:
        print("⚠️  Lỗi hoặc cảnh báo:")
        print(result.stderr)

# 1) Pipeline trước khi train (tạo mask, nhãn đa lớp, gộp, chia)
pipeline_scripts = [
    "mask.py",          # tạo leaf_masks/ + disease_masks/ (healthy = no-object)
    "yolo_label.py",    # xuất labels/ đa lớp: blight/scab/spot/rust/mildew
    "group_label.py",   # gộp theo nhóm bệnh vào merged/
    "split.py"          # chia merged/ -> final/
]

for script in pipeline_scripts:
    if (ROOT / script).exists():
        run_script(script)
    else:
        print(f"⚠️  Không tìm thấy file: {script}")

# 2) data.yaml (ghi đúng 5 lớp bệnh)
data_yaml = ROOT / "data.yaml"
data_yaml.write_text(
    "path: final\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "nc: 5\n"
    "names: [blight, scab, spot, rust, mildew]\n",
    encoding="utf-8"
)
print("ℹ️  Đã ghi data.yaml (5 lớp: blight, scab, spot, rust, mildew).")

# 3) Train YOLOv8 segmentation
print("\n==============================")
print("🏋️  BẮT ĐẦU HUẤN LUYỆN YOLOv8-SEG (đa lớp bệnh)")
print("==============================\n")

model = YOLO("yolov8n-seg.pt")  # dùng 'yolov8s-seg.pt' nếu GPU mạnh

results = model.train(
    data=str(data_yaml),   # dùng data.yaml mới tạo
    epochs=50,
    imgsz=640,
    batch=8,
    name="plant_disease_seg_multiclass",
    pretrained=True,
    device="cpu",          # đổi thành "cuda" nếu có GPU
    workers=0              # khuyến nghị trên Windows
)

# 4) Đánh giá
print("\n==============================")
print("📊  ĐÁNH GIÁ MÔ HÌNH")
print("==============================\n")
metrics = model.val(data=str(data_yaml))
print(metrics)

# 5) Dự đoán thử
print("\n==============================")
print("🔍  DỰ ĐOÁN THỬ TRÊN ẢNH TEST")
print("==============================\n")
model.predict(source=str(ROOT / "final/images/test"), conf=0.5, save=True)
