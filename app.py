import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

st.set_page_config(page_title="🌿 Leaf Disease Detection", layout="wide")
st.title("🌿 Leaf Disease Detection (Real-Time)")

# ============================
# 🔹 Load Models
# ============================
lleaf_cls = YOLO("runs/detect/train_leaf/weights/best.pt")
disease_seg = YOLO("runs/segment/train_seg/weights/best.pt")


# ============================
# 🔹 Start webcam
# ============================
FRAME_WINDOW = st.image([])  # nơi hiển thị video
camera = cv2.VideoCapture(0)  # 0 = webcam mặc định

st.sidebar.header("⚙️ Cài đặt")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
enable_crop = st.sidebar.checkbox("✂️ Tự động cắt lá để detect bệnh", value=True)
st.sidebar.info("Nhấn **Stop** để dừng camera.")

stop_button = st.sidebar.button("⛔ Stop camera")

# ============================
# 🔹 Loop đọc từng khung hình
# ============================
while camera.isOpened() and not stop_button:
    success, frame = camera.read()
    if not success:
        st.warning("Không thể truy cập camera!")
        break

    # Lật ảnh để hiển thị tự nhiên hơn
    frame = cv2.flip(frame, 1)

    # -----------------------------
    # 1️⃣ Phát hiện lá bằng model phân loại
    # -----------------------------
    res_cls = leaf_cls(frame, conf=conf_threshold)
    boxes = res_cls[0].boxes.xyxy.cpu().numpy() if res_cls[0].boxes is not None else []

    # -----------------------------
    # 2️⃣ Duyệt qua từng bounding box
    # -----------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if enable_crop and crop.size > 0:
            # 3️⃣ Phát hiện bệnh trong vùng lá đã cắt
            seg_result = disease_seg(crop, conf=conf_threshold)
            seg_img = seg_result[0].plot()

            # Ghép kết quả trở lại vào frame
            frame[y1:y2, x1:x2] = cv2.resize(seg_img, (x2 - x1, y2 - y1))

        # 4️⃣ Vẽ khung lá
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Leaf", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -----------------------------
    # 5️⃣ Hiển thị lên Streamlit
    # -----------------------------
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    time.sleep(0.05)  # làm mượt stream

camera.release()
st.success("Camera đã dừng 🎉")


