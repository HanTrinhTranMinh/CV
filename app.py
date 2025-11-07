import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import cv2, os, tempfile
import numpy as np
from PIL import Image

# ========== 1️⃣ UI ==========
st.set_page_config(page_title="🌿 Smart Plant Health Detection", layout="centered")
st.title("🌿 Smart Plant Disease Detection (Hybrid YOLO + CNN)")
st.caption("Stage 1: Leaf/Background → Stage 2: Healthy/Diseased → YOLOv8 Seg (multi-disease).")

# ========== 2️⃣ PATH & LOAD ==========
# 🧠 Các đường dẫn phổ biến cho YOLO đa lớp (theo model.py đã sửa)
YOLO_CANDIDATES = [
    "runs/segment/plant_disease_seg_multiclass/weights/best.pt",   # tên run đề xuất
    "runs/segment/plant_disease_seg/weights/best.pt",              # nếu bạn vẫn dùng tên cũ
    "best.pt"                                                      # fallback nếu bạn để chung thư mục
]

def _find_yolo_weight():
    for p in YOLO_CANDIDATES:
        if os.path.exists(p):
            return p
    return YOLO_CANDIDATES[0]  # cứ trỏ về path chuẩn (sẽ báo lỗi nếu chưa train)

@st.cache_resource
def load_models():
    # 2 CNN nhị phân (giữ đường dẫn cũ của bạn)
    leaf_model   = tf.keras.models.load_model("leaf_or_background.h5")
    health_model = tf.keras.models.load_model("healthy_or_diseased.h5")

    # YOLO đa lớp bệnh
    yolo_path = _find_yolo_weight()
    yolo_model = YOLO(yolo_path)
    return leaf_model, health_model, yolo_model, yolo_path

leaf_model, health_model, yolo_model, yolo_path = load_models()
st.sidebar.success(f"✅ YOLO weights: {yolo_path}")

# Lớp bệnh theo data.yaml (thứ tự phải khớp khi train)
DISEASE_NAMES = ["blight", "scab", "spot", "rust", "mildew"]

# ========== 3️⃣ UTIL ==========
def preprocess_image(image_bgr):
    img = cv2.resize(image_bgr, (224, 224)).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def draw_tag(img, color, label):
    h, w, _ = img.shape
    out = img.copy()
    cv2.rectangle(out, (5, 5), (w - 5, h - 5), color, 4)
    cv2.putText(out, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out

def run_yolo(image_path, conf=0.4):
    res = yolo_model.predict(source=image_path, conf=conf, verbose=False)
    return res[0]  # ultralytics Result

def format_detections(result):
    """Trả về list text mô tả mask theo lớp bệnh + score."""
    infos = []
    if result.masks is None:
        return infos
    # result.boxes.cls/ conf có thể None với seg nặng, fallback bằng probs
    if result.probs is not None:
        # phân loại toàn ảnh (hiếm dùng cho seg multi object) – không dùng ở đây
        pass
    boxes = result.boxes
    if boxes is None:
        return infos
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(cls))
    for c, s in zip(cls, conf):
        name = DISEASE_NAMES[c] if 0 <= c < len(DISEASE_NAMES) else f"class_{c}"
        infos.append(f"• {name}: {s:.2f}")
    return infos

# ========== 4️⃣ PIPELINE 1 ẢNH ==========
def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Invalid image."

    # Stage 1: Leaf vs Background
    leaf_prob = float(leaf_model.predict(preprocess_image(img), verbose=0)[0][0])
    if leaf_prob < 0.5:
        frame = draw_tag(img, (0, 255, 255), "Not a Leaf")
        return frame, "🪨 Background / non-leaf."

    # Stage 2: Healthy vs Diseased
    health_prob = float(health_model.predict(preprocess_image(img), verbose=0)[0][0])
    if health_prob > 0.5:
        frame = draw_tag(img, (0, 200, 0), "Healthy Leaf")
        return frame, "🍃 Healthy leaf."

    # Stage 3: YOLOv8 Seg (đa lớp bệnh)
    result = run_yolo(image_path, conf=0.4)
    frame = result.plot()  # vẽ masks/contours lên ảnh
    tags = format_detections(result)
    tag_text = "⚠️ Diseased leaf detected!\n" + ("\n".join(tags) if tags else "No visible disease mask.")
    frame = draw_tag(frame, (0, 0, 255), "Diseased Leaf")
    return frame, tag_text

# ========== 5️⃣ UI ==========
tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Realtime Camera"])

with tab1:
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.image(uploaded_file, caption="🖼️ Uploaded", use_column_width=True)
        with st.spinner("🔍 Analyzing..."):
            frame, text = analyze_image(temp_path)
        if frame is not None:
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            c2.success(text)
        else:
            st.error(text)
        os.remove(temp_path)
    else:
        st.info("👆 Upload a leaf image to start detection.")

with tab2:
    st.info("🔴 Realtime detection with webcam (CPU may be slow).")
    run = st.checkbox("Start camera")
    if run:
        cam = cv2.VideoCapture(0)
        stframe = st.empty()
        try:
            while run:
                ok, frame = cam.read()
                if not ok:
                    st.error("Camera not detected!")
                    break

                lp = float(leaf_model.predict(preprocess_image(frame), verbose=0)[0][0])
                if lp < 0.5:
                    disp = draw_tag(frame, (0, 255, 255), "Not a Leaf")
                else:
                    hp = float(health_model.predict(preprocess_image(frame), verbose=0)[0][0])
                    if hp > 0.5:
                        disp = draw_tag(frame, (0, 200, 0), "Healthy")
                    else:
                        # để realtime nhanh: chỉ vẽ tag, không chạy YOLO mỗi frame
                        disp = draw_tag(frame, (0, 0, 255), "Diseased")
                stframe.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        finally:
            cam.release()
