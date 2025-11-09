import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("🌿 Leaf Disease Detection")

leaf_cls = YOLO("runs/classify/train/weights/best.pt")
disease_seg = YOLO("runs/segment/train/weights/best.pt")

camera = st.camera_input("Chụp ảnh hoặc tải ảnh lá")

if camera:
    img = cv2.imdecode(np.frombuffer(camera.read(), np.uint8), cv2.IMREAD_COLOR)
    res = leaf_cls(img)
    label = res[0].names[int(res[0].probs.top1)]

    if label == "background":
        st.warning("🚫 Không phải lá cây.")
    else:
        result = disease_seg(img)
        result[0].show()
        st.image(result[0].plot(), caption="Kết quả phân loại bệnh", use_column_width=True)
