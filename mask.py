import cv2
import numpy as np
import os
from pathlib import Path

# =====================================================
# ⚙️ CẤU HÌNH
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "Dataset"

INPUT_DIR = ROOT / "binary_health_dataset"   # ✅ tập đã lọc chỉ còn lá
OUTPUT_LEAF = ROOT / "leaf_masks"
OUTPUT_DISEASE = ROOT / "disease_masks"
IMG_SIZE = (512, 512)

OUTPUT_LEAF.mkdir(parents=True, exist_ok=True)
OUTPUT_DISEASE.mkdir(parents=True, exist_ok=True)


# =====================================================
# 🧩 HÀM HỖ TRỢ
# =====================================================
def ensure_dir(path: Path):
    """Tạo thư mục cha nếu chưa có"""
    path.parent.mkdir(parents=True, exist_ok=True)


# =====================================================
# 🌿 TÁCH LÁ
# =====================================================
def segment_leaf(img_bgr):
    """Trả về leaf_mask (0/255). Dùng HSV + morphology để ổn định hơn."""
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Hỗ trợ xanh lá nhạt -> đậm
    lower = np.array([25, 30, 30], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology: loại nhiễu và liền mạch
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))

    # Giữ vùng lớn nhất (lá chính)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        keep = np.zeros_like(mask)
        cv2.drawContours(keep, [biggest], -1, 255, thickness=cv2.FILLED)
        mask = keep

    return mask


# =====================================================
# 🍂 PHÁT HIỆN BỆNH (cho ảnh diseased)
# =====================================================
def disease_region_diseased(img_bgr, leaf_mask):
    """Tìm vùng bệnh trong ảnh diseased: dùng Otsu threshold trên vùng lá"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Chỉ xét trong lá
    gray_in_leaf = gray.copy()
    gray_in_leaf[leaf_mask == 0] = 255

    # Otsu threshold để tìm vùng tối hơn
    _, otsu = cv2.threshold(gray_in_leaf, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Giới hạn trong lá
    disease = cv2.bitwise_and(otsu, leaf_mask)

    # Morphology làm mịn
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    disease = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Lọc vùng nhỏ
    cnts, _ = cv2.findContours(disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(disease)
    for c in cnts:
        if cv2.contourArea(c) >= 80:
            cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
    return keep


# =====================================================
# 🌱 ẢNH HEALTHY (không có bệnh)
# =====================================================
def disease_region_healthy(leaf_mask):
    """Ảnh healthy: không có vùng bệnh."""
    return np.zeros_like(leaf_mask, dtype=np.uint8)


# =====================================================
# 🚀 XỬ LÝ TOÀN BỘ DATASET
# =====================================================
count = 0
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = Path(root) / file
        rel_path = img_path.relative_to(INPUT_DIR)
        top_class = rel_path.parts[0].lower()  # healthy / diseased

        leaf_out = OUTPUT_LEAF / rel_path
        disease_out = OUTPUT_DISEASE / rel_path
        ensure_dir(leaf_out)
        ensure_dir(disease_out)

        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue
        H0, W0 = img0.shape[:2]

        # Resize nhỏ để xử lý ổn định hơn
        img = cv2.resize(img0, IMG_SIZE)
        leaf_mask_small = segment_leaf(img)

        if top_class == "healthy":
            disease_mask_small = disease_region_healthy(leaf_mask_small)
        else:
            disease_mask_small = disease_region_diseased(img, leaf_mask_small)

        # Resize lại về kích thước gốc
        leaf_mask = cv2.resize(leaf_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
        disease_mask = cv2.resize(disease_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(leaf_out), leaf_mask)
        cv2.imwrite(str(disease_out), disease_mask)

        count += 1
        if count % 100 == 0:
            print(f"🟢 Đã xử lý {count} ảnh...")

print(f"\n✅ Hoàn tất! Đã tạo {count} mask:")
print(f"   🌿 {OUTPUT_LEAF}")
print(f"   🍂 {OUTPUT_DISEASE}")
