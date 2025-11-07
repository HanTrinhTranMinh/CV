import os
import shutil

# =======================
# CẤU HÌNH
# =======================
IMG_DIR    = "binary_health_dataset"   # Ảnh nguồn (healthy/diseased/...)
LABEL_DIR  = "labels"                  # Nhãn YOLO (polygon) sinh từ masks
OUT_IMG    = "merged/images"
OUT_LABEL  = "merged/labels"

# Có copy ảnh "no-object" (healthy/other) không?
INCLUDE_NOOBJ = True
NOOBJ_GROUP   = "noobj"                # thư mục chứa no-object (ngoài 5 lớp)

# =======================
# KHỚP VỚI data.yaml
# data.yaml:
# nc: 5
# names: [blight, scab, spot, rust, mildew]
# =======================
DISEASE_CLASSES = ["blight", "scab", "spot", "rust", "mildew"]
KEYWORDS = {
    "blight":  ["blight"],
    "scab":    ["scab"],
    "spot":    ["spot"],
    "rust":    ["rust"],
    "mildew":  ["mildew"],
}

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

def infer_group_from_folder(folder_name: str) -> str | None:
    """
    Trả về tên group (blight/scab/spot/rust/mildew) nếu khớp keyword.
    Healthy/other → None (no-object).
    """
    n = folder_name.lower()
    if "healthy" in n or "background" in n:
        return None
    for g, kws in KEYWORDS.items():
        if any(k in n for k in kws):
            return g
    return None  # không khớp 5 lớp

def safe_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

# =======================
# DUYỆT VÀ GỘP
# =======================
for root, _, files in os.walk(IMG_DIR):
    # ví dụ: root = binary_health_dataset/diseased/Apple__Apple_scab
    rel_dir_from_img = os.path.relpath(root, IMG_DIR)        # diseased/Apple__Apple_scab
    base_folder = os.path.basename(root)                      # Apple__Apple_scab
    group = infer_group_from_folder(base_folder)              # 1 trong 5 lớp hoặc None (no-object)

    # Bỏ qua cây nền còn sót (cực hiếm)
    if "background" in base_folder.lower():
        continue

    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_img   = os.path.join(root, file)
        src_label = os.path.join(LABEL_DIR, rel_dir_from_img,
                                 os.path.splitext(file)[0] + ".txt")

        # ======== NHÓM ĐÍCH ========
        if group is None:
            # no-object (healthy/other)
            if not INCLUDE_NOOBJ:
                # không copy mẫu nền
                continue
            out_group = NOOBJ_GROUP
        else:
            out_group = group  # blight/scab/spot/rust/mildew

        # Tạo đích GIỮ CẤU TRÚC CON để tránh trùng tên
        dst_img_dir   = os.path.join(OUT_IMG, out_group, rel_dir_from_img)
        dst_label_dir = os.path.join(OUT_LABEL, out_group, rel_dir_from_img)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)

        # Đường dẫn đích
        dst_img   = os.path.join(dst_img_dir, file)
        dst_label = os.path.join(dst_label_dir, os.path.splitext(file)[0] + ".txt")

        # ======== COPY ẢNH ========
        safe_copy(src_img, dst_img)

        # ======== COPY NHÃN ========
        if os.path.exists(src_label):
            safe_copy(src_label, dst_label)
        else:
            # tạo txt rỗng (YOLO hiểu là không có object)
            open(dst_label, "w", encoding="utf-8").close()

print("✅ Đã gộp theo nhóm bệnh trùng khớp data.yaml (blight/scab/spot/rust/mildew).")
print(f"ℹ️  Ảnh no-object (healthy/other) {'được copy vào '+NOOBJ_GROUP if INCLUDE_NOOBJ else 'đã bị bỏ qua'}.")
