import os

def rename_images(folder_path, prefix="image", start=1, end=None):
    """
    Đổi tên toàn bộ ảnh trong thư mục theo dạng:
        <prefix>_<stt>.<đuôi gốc>

    Tham số:
        folder_path: Đường dẫn thư mục chứa ảnh
        prefix: Tên mới (vd: 'cat', 'leaf', 'house')
        start: Số thứ tự bắt đầu
        end: Số thứ tự kết thúc (nếu None -> đổi hết)
    """
    # Các định dạng ảnh phổ biến
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    # Lấy toàn bộ file ảnh trong thư mục
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    files.sort()  # sắp xếp để có thứ tự ổn định

    total = len(files)
    if end is None or end > total + start - 1:
        end = total + start - 1

    print(f"📂 Thư mục: {folder_path}")
    print(f"🔤 Prefix: {prefix}")
    print(f"🔢 Từ {start} đến {end}")

    for i, filename in enumerate(files, start=start):
        if i > end:
            break
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1]
        new_filename = f"{prefix}_{i}{ext}"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"✅ {filename} → {new_filename}")

    print(f"\n🎉 Hoàn tất đổi tên {min(end - start + 1, total)} ảnh!")

# === Ví dụ chạy ===
rename_images(r"C:\Users\Admin\Desktop\Dataset\Tomato___Early_blight", prefix="blight", start=1001, end=2000)
