import os
import subprocess
import time

# ==========================================
# ⚙️ CẤU HÌNH
# ==========================================
DATASET_YAML = "dataset.yaml"           # segmentation dataset
DATASET_LEAF_YAML = "dataset_leaf.yaml" # leaf detection dataset
SEG_MODEL = "runs/segment/train_seg/weights/best.pt"
DET_MODEL = "runs/detect/train_leaf/weights/best.pt"

# ==========================================
# 🧩 ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
# ==========================================
def evaluate_model():
    start_time = time.time()
    print("🧩 [EVALUATE] Bắt đầu đánh giá mô hình trên tập test...\n")

    # Kiểm tra file tồn tại
    if not os.path.exists(SEG_MODEL):
        print(f"⚠️ Không tìm thấy model segmentation: {SEG_MODEL}")
        return
    if not os.path.exists(DET_MODEL):
        print(f"⚠️ Không tìm thấy model detection: {DET_MODEL}")
        return
    if not os.path.exists(DATASET_YAML) or not os.path.exists(DATASET_LEAF_YAML):
        print("⚠️ Không tìm thấy file dataset YAML!")
        return

    # -----------------------------
    # 1️⃣ Đánh giá segmentation
    # -----------------------------
    print("📊 Đang đánh giá mô hình **Segmentation (Bệnh lá)** ...")
    subprocess.run([
        "yolo", "segment", "val",
        f"model={SEG_MODEL}",
        f"data={DATASET_YAML}",
        "split=test",
        "save_json=True",
        "project=runs/evaluate",
        "name=seg_test_eval"
    ], check=True)
    print("✅ Hoàn tất đánh giá segmentation!\n")

    # -----------------------------
    # 2️⃣ Đánh giá detection (lá)
    # -----------------------------
    print("📊 Đang đánh giá mô hình **Leaf Detection** ...")
    subprocess.run([
        "yolo", "detect", "val",
        f"model={DET_MODEL}",
        f"data={DATASET_LEAF_YAML}",
        "split=test",
        "save_json=True",
        "project=runs/evaluate",
        "name=leaf_test_eval"
    ], check=True)
    print("✅ Hoàn tất đánh giá detection!\n")

    # -----------------------------
    # 3️⃣ Tổng kết thời gian
    # -----------------------------
    total_time = time.time() - start_time
    print("🎯 Đã đánh giá xong cả hai mô hình!")
    print(f"🕒 Thời gian tổng: {total_time:.1f}s\n")
    print("📂 Kết quả lưu tại:")
    print("  - Segmentation: runs/evaluate/seg_test_eval/")
    print("  - Detection: runs/evaluate/leaf_test_eval/")

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    evaluate_model()
