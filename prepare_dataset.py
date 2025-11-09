import os
import cv2
import numpy as np

# ========================
# 🔧 CẤU HÌNH
# ========================
INPUT_DIR = "Dataset"      # Gốc chứa healthy / disease
OUTPUT_DIR = "prepared"    # Nơi lưu mask đã xử lý
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# 🌞 HÀM TĂNG SÁNG / CẢI THIỆN ẢNH
# ========================

def enhance_brightness(img_bgr):
    """Tăng tương phản cục bộ bằng CLAHE"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def adjust_gamma(image, gamma=1.3):
    """Điều chỉnh độ sáng toàn cục"""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ========================
# 🌿 HÀM TẠO MASK
# ========================

def create_mask(img_path, is_healthy):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Không đọc được ảnh: {img_path}")
        return None, None

    # ======== TĂNG SÁNG + GIẢM BÓNG ========
    img_bgr = enhance_brightness(img_bgr)
    img_bgr = adjust_gamma(img_bgr, gamma=1.2)

    # Cân bằng màu để giảm ám vàng / tối
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ======== TRÍCH XUẤT VÙNG LÁ ========
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 20, 20], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask_leaf = cv2.inRange(hsv, lower, upper)

    # Loại bỏ nhiễu sáng bằng adaptive blur + morphology
    mask_leaf = cv2.GaussianBlur(mask_leaf, (5, 5), 0)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    mask_leaf = cv2.medianBlur(mask_leaf, 7)

    # Giữ vùng lớn nhất (lá chính)
    contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask_leaf)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_clean, [largest], -1, 255, thickness=-1)
    mask_leaf = mask_clean

    # Làm mượt viền
    mask_leaf = cv2.GaussianBlur(mask_leaf, (9,9), 0)
    _, mask_leaf = cv2.threshold(mask_leaf, 127, 255, cv2.THRESH_BINARY)

    # ======== TẠO MASK BỆNH ========
    if is_healthy:
        mask_disease = np.zeros_like(mask_leaf)
    else:
        # ======== Phát hiện vùng bệnh ========
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Làm mượt, loại nhiễu ánh sáng mạnh
        gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
        diff = cv2.absdiff(gray_blur, cv2.medianBlur(gray_blur, 15))

        # Tăng tương phản vùng bệnh
        _, mask_disease = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask_disease = cv2.bitwise_and(mask_disease, mask_leaf)

        # ======== Loại bỏ nhiễu nhỏ và vùng sai ========
        contours, _ = cv2.findContours(mask_disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask_disease)

        h, w = mask_disease.shape[:2]
        min_area = (h * w) * 0.001   # chỉ giữ đốm >0.1% diện tích lá
        max_area = (h * w) * 0.25    # loại vùng quá lớn (bóng)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                cv2.drawContours(mask_clean, [cnt], -1, 255, -1)

        # Làm mượt lần cuối
        mask_disease = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))


    return mask_leaf, mask_disease

# ========================
# 📂 HÀM DUYỆT THƯ MỤC
# ========================

def process_folder(folder, label, is_healthy=False):
    out_label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for i, f in enumerate(os.listdir(folder)):
        path = os.path.join(folder, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        leaf_mask, disease_mask = create_mask(path, is_healthy)
        if leaf_mask is None:
            continue

        base = os.path.splitext(f)[0]
        cv2.imwrite(os.path.join(out_label_dir, f"{base}_leaf.png"), leaf_mask)
        if not is_healthy:
            cv2.imwrite(os.path.join(out_label_dir, f"{base}_disease.png"), disease_mask)

# ========================
# 🚀 CHẠY XỬ LÝ
# ========================

# Healthy leaves
process_folder(os.path.join(INPUT_DIR, "healthy"), "healthy", True)

# Diseased leaves
for cls in os.listdir(os.path.join(INPUT_DIR, "disease")):
    path = os.path.join(INPUT_DIR, "disease", cls)
    if not os.path.isdir(path):
        continue
    process_folder(path, cls, False)

print("✅ Hoàn tất xử lý mask! Kết quả nằm trong thư mục 'prepared/'")
