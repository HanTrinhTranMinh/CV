import cv2
import os
import numpy as np

# ====== CẤU HÌNH ======
IMG_DIR = "binary_health_dataset"     # healthy/diseased/<ClassFolder>/img.jpg
DISEASE_MASK_DIR = "disease_masks"    # mask bệnh (đã resize về size gốc)
LABEL_OUT_DIR = "labels"              # nơi xuất .txt YOLO
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

# ====== LỚP BỆNH → ID (khớp data.yaml) ======
DISEASE_CLASSES = ["blight", "scab", "spot", "rust", "mildew"]
KEYWORDS = {
    "blight":  ["blight"],
    "scab":    ["scab"],
    "spot":    ["spot"],
    "rust":    ["rust"],
    "mildew":  ["mildew"],
}

def infer_disease_class_id(rel_dir_parts):
    """Trả về class_id (0..4) nếu là ảnh bệnh; None nếu healthy/không match."""
    if any("healthy" in p.lower() for p in rel_dir_parts):
        return None
    folder = rel_dir_parts[-1].lower() if rel_dir_parts else ""
    for idx, name in enumerate(DISEASE_CLASSES):
        if any(k in folder for k in KEYWORDS[name]):
            return idx
    return None

def contours_to_yolo_lines(mask, class_id, img_h, img_w, min_area=50, simplify_eps_ratio=0.002):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = simplify_eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = []
        for p in approx:
            x, y = p[0]
            pts += [x / img_w, y / img_h]
        if len(pts) >= 6:
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in pts))
    return lines

def mask_path(rel_dir, file_name):
    return os.path.join(DISEASE_MASK_DIR, rel_dir, file_name)

# ====== DUYỆT ẢNH & GHI NHÃN ======
for root, _, files in os.walk(IMG_DIR):
    rel_dir = os.path.relpath(root, IMG_DIR)
    rel_parts = [] if rel_dir == "." else rel_dir.split(os.sep)

    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Suy class theo bệnh
        class_id = infer_disease_class_id(rel_parts)

        # Đường dẫn IO
        image_path = os.path.join(root, file)
        out_dir = os.path.join(LABEL_OUT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        label_path = os.path.join(out_dir, os.path.splitext(file)[0] + ".txt")

        # Healthy/không xác định loại bệnh → no-object
        if class_id is None:
            open(label_path, "w", encoding="utf-8").close()
            continue

        # Đọc ảnh & mask bệnh
        img = cv2.imread(image_path)
        if img is None:
            open(label_path, "w", encoding="utf-8").close()
            continue
        h, w = img.shape[:2]

        mpath = mask_path(rel_dir, file)
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) if os.path.exists(mpath) else None
        if mask is None:
            # không có mask → coi như no-object
            open(label_path, "w", encoding="utf-8").close()
            continue

        # Tạo các dòng polygon theo lớp bệnh
        lines = contours_to_yolo_lines(mask, class_id, h, w)

        # Ghi file nhãn (có thể rỗng nếu không tìm thấy contour đủ lớn)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

print("✅ Đã tạo nhãn YOLO segmentation đa lớp theo bệnh (labels/)")
