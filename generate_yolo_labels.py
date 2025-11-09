import os
import cv2
import numpy as np

# ========================
# ⚙️ CẤU HÌNH
# ========================
PREPARED_DIR = "prepared"
OUTPUT_IMAGE_DIR = "yolo_dataset/images"
OUTPUT_LABEL_DIR = "yolo_dataset/labels"

CLASS_MAP = {
    "healthy": 0,
    "black_rot": 1,
    "blight": 2,
    "middew": 3,
    "rust": 4,
    "spot": 5
}


# ========================
# 🧩 CHUYỂN MASK → LABEL YOLO
# ========================
def mask_to_yolo(mask_path, cls_name, output_txt):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Không đọc được mask: {mask_path}")
        return
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(output_txt, "w") as f:
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.squeeze(1)
            norm_points = []
            for x, y in contour:
                nx = x / w
                ny = y / h
                norm_points.extend([nx, ny])
            cls_id = CLASS_MAP.get(cls_name, 0)
            line = f"{cls_id} " + " ".join([f"{p:.6f}" for p in norm_points])
            f.write(line + "\n")


# ========================
# 🎨 TẠO ẢNH GIẢ MÀU TỪ MASK
# ========================
def synthesize_image(leaf_path, disease_path=None):
    leaf = cv2.imread(leaf_path, cv2.IMREAD_GRAYSCALE)
    if leaf is None:
        return None

    h, w = leaf.shape
    leaf_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # màu nền lá (xanh lá cây)
    leaf_rgb[leaf > 0] = (50, 180, 50)

    # nếu có mask bệnh thì tô vùng bệnh (đỏ)
    if disease_path and os.path.exists(disease_path):
        disease = cv2.imread(disease_path, cv2.IMREAD_GRAYSCALE)
        leaf_rgb[disease > 0] = (0, 0, 255)

    return leaf_rgb


# ========================
# 🚀 XỬ LÝ TỪNG LỚP
# ========================
for cls_name in CLASS_MAP.keys():
    cls_dir = os.path.join(PREPARED_DIR, cls_name)
    if not os.path.exists(cls_dir):
        print(f"⚠️ Bỏ qua lớp {cls_name} (không tồn tại thư mục).")
        continue

    out_img_dir = os.path.join(OUTPUT_IMAGE_DIR, cls_name)
    out_lbl_dir = os.path.join(OUTPUT_LABEL_DIR, cls_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for f in os.listdir(cls_dir):
        if not f.endswith("_leaf.png"):
            continue

        leaf_path = os.path.join(cls_dir, f)
        disease_path = leaf_path.replace("_leaf.png", "_disease.png")
        base_name = f.replace("_leaf.png", ".jpg")
        out_img_path = os.path.join(out_img_dir, base_name)
        out_txt_path = os.path.join(out_lbl_dir, base_name.replace(".jpg", ".txt"))

        # === 1️⃣ Tạo ảnh tổng hợp từ leaf + disease
        synth_img = synthesize_image(leaf_path, disease_path if os.path.exists(disease_path) else None)
        if synth_img is not None:
            cv2.imwrite(out_img_path, synth_img)

        # === 2️⃣ Sinh nhãn YOLO từ mask bệnh (nếu có) hoặc từ leaf nếu healthy
        target_mask = disease_path if os.path.exists(disease_path) else leaf_path
        mask_to_yolo(target_mask, cls_name, out_txt_path)

print("✅ generate_yolo_labels.py: Đã tạo ảnh + nhãn YOLO segmentation thành công!")
