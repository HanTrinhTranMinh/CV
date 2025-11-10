import os
import cv2
import numpy as np

# ========================
# ⚙️ CẤU HÌNH
# ========================
PREPARED_DIR = "prepared"
OUTPUT_IMAGE_DIR = "yolo_leaf/images"
OUTPUT_LABEL_DIR = "yolo_leaf/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Lớp dùng cho YOLO detect
CLASS_MAP = {
    "background": 0,
    "leaf": 1
}


# ========================
# 📦 HÀM CHUYỂN MASK → LABEL YOLO (bbox)
# ========================
def mask_to_yolo_bbox(mask_path, output_txt, cls_id=1):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Không đọc được mask: {mask_path}")
        return

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(output_txt, "w") as f:
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # bỏ vùng nhỏ
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Chuyển sang YOLO format (tâm_x, tâm_y, width, height)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            w_norm = bw / w
            h_norm = bh / h

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


# ========================
# 🎨 HÀM TẠO ẢNH TỪ MASK LÁ
# ========================
def synthesize_leaf_image(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask > 0] = (60, 180, 60)  # màu xanh lá
    return img


# ========================
# 🚀 DUYỆT TOÀN BỘ THƯ MỤC prepared/
# ========================
for cls_name in os.listdir(PREPARED_DIR):
    cls_dir = os.path.join(PREPARED_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        continue

    print(f"🔹 Đang xử lý lớp: {cls_name}")

    for f in os.listdir(cls_dir):
        if not f.endswith("_leaf.png"):
            continue

        leaf_path = os.path.join(cls_dir, f)
        base_name = f.replace("_leaf.png", ".jpg")
        out_img_path = os.path.join(OUTPUT_IMAGE_DIR, base_name)
        out_txt_path = os.path.join(OUTPUT_LABEL_DIR, base_name.replace(".jpg", ".txt"))

        # 1️⃣ Tạo ảnh từ mask
        leaf_img = synthesize_leaf_image(leaf_path)
        if leaf_img is None:
            continue
        cv2.imwrite(out_img_path, leaf_img)

        # 2️⃣ Sinh nhãn YOLO bbox (class = leaf)
        mask_to_yolo_bbox(leaf_path, out_txt_path, cls_id=CLASS_MAP["leaf"])

print("✅ generate_yolo_labels_leaf.py: Đã tạo ảnh + nhãn YOLO detection cho lá thành công!")
