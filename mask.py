import cv2
import numpy as np
import os

# ====== CẤU HÌNH ======
INPUT_DIR = "binary_health_dataset"      # ✅ chạy trên tập đã lọc chỉ còn lá (healthy/diseased)
OUTPUT_LEAF = "leaf_masks"
OUTPUT_DISEASE = "disease_masks"
IMG_SIZE = (512, 512)

os.makedirs(OUTPUT_LEAF, exist_ok=True)
os.makedirs(OUTPUT_DISEASE, exist_ok=True)

# ====== UTIL ======
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def segment_leaf(img_bgr):
    """Trả về leaf_mask (0/255). Dùng HSV + morphology để ổn định hơn."""
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Hỗ trợ cả xanh nhạt và xanh đậm
    lower1 = np.array([25, 30, 30], dtype=np.uint8)
    upper1 = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)

    # Đóng - Mở để liền mạch & loại nhiễu
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))

    # (tuỳ chọn) Giữ vùng lớn nhất để tránh dính nền xanh gần đó
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        areas = [cv2.contourArea(c) for c in cnts]
        biggest = cnts[int(np.argmax(areas))]
        keep = np.zeros_like(mask)
        cv2.drawContours(keep, [biggest], -1, 255, thickness=cv2.FILLED)
        mask = keep

    return mask

def disease_region_diseased(img_bgr, leaf_mask):
    """Tìm vùng bệnh trong ảnh diseased: Otsu trên gray (chỉ bên trong lá) + morphology + lọc area nhỏ."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Chỉ xét trong lá; ngoài lá set = 255 (trắng) để không bị threshold nhầm
    gray_in_leaf = gray.copy()
    gray_in_leaf[leaf_mask == 0] = 255

    # Otsu cho nhị phân vùng tối → bệnh hay nằm ở vùng tối/khác màu
    _, otsu = cv2.threshold(gray_in_leaf, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Chỉ giữ trong lá
    disease = cv2.bitwise_and(otsu, leaf_mask)

    # Mở - Đóng để sạch biên
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    disease = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Lọc nhiễu nhỏ theo diện tích
    cnts, _ = cv2.findContours(disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(disease)
    for c in cnts:
        if cv2.contourArea(c) >= 80:  # ngưỡng tuỳ chỉnh
            cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
    return keep

def disease_region_healthy(leaf_mask):
    """Ảnh healthy: KHÔNG tô đầy vùng bệnh → chỉ vẽ viền lá mỏng (morphological gradient)."""
    edge = cv2.morphologyEx(leaf_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    # (tuỳ chọn) làm mảnh viền
    edge = cv2.morphologyEx(edge, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=1)
    return edge  # nếu muốn hoàn toàn rỗng thì return np.zeros_like(leaf_mask)

# ====== XỬ LÝ TOÀN BỘ FOLDER ======
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(root, file)
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        top_class = rel_path.split(os.sep)[0].lower()  # 'healthy' hoặc 'diseased'

        leaf_out = os.path.join(OUTPUT_LEAF, rel_path)
        disease_out = os.path.join(OUTPUT_DISEASE, rel_path)
        ensure_dir(leaf_out)
        ensure_dir(disease_out)

        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue
        H0, W0 = img0.shape[:2]

        # xử lý trên ảnh thu nhỏ cho ổn định
        img = cv2.resize(img0, IMG_SIZE)
        leaf_mask_small = segment_leaf(img)

        if top_class == "healthy":
            # 👉 ĐỂ TRAIN THEO BỆNH: healthy = NO-OBJECT → disease mask = zeros
            disease_mask_small = np.zeros_like(leaf_mask_small, dtype=np.uint8)
        else:
            disease_mask_small = disease_region_diseased(img, leaf_mask_small)

        # ⬅️ Resize mask về kích thước gốc để khớp với ảnh gốc
        leaf_mask    = cv2.resize(leaf_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
        disease_mask = cv2.resize(disease_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(leaf_out, leaf_mask)
        cv2.imwrite(disease_out, disease_mask)


print("✅ Hoàn tất tạo leaf_masks/ và disease_masks/ từ binary_health_dataset/")
