# 🌿 Leaf Disease Segmentation Pipeline (YOLOv8)

This repository provides a **complete automated pipeline** for detecting and segmenting **leaf diseases** using YOLOv8 segmentation.  
It includes **image enhancement**, **mask generation**, **label creation**, **train/val splitting**, and **YOLO training**, all runnable with a single command.

---

## 🧠 Overview

### 🌳 Dataset structure (before running)
Dataset/
├── background/ # non-leaf images
├── healthy/ # healthy leaves
└── disease/
├── blight/
├── rust/
├── middew/
├── black_rot/
└── spot/

markdown
Copy code

### 🧩 Processing stages
1. **Image enhancement + mask creation**  
   - CLAHE (local contrast)
   - Gamma correction (brightness)
   - HSV filtering + morphology → clean leaf contour  
   → Output: leaf masks & disease masks  

2. **Label generation for YOLO segmentation**  
   - Each mask is converted into YOLO polygon label (`.txt`)  
   - Coordinates normalized to [0,1]  

3. **Train/Val splitting** (default 80/20)

4. **YOLOv8 segmentation training**  
   - Real-time epoch logging in console  
   - Results saved to `runs/segment/train/weights/best.pt`

---

## 🧰 Requirements

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
🚀 Quick Start
1️⃣ Prepare dataset
Place your dataset inside a folder named Dataset/ as shown above.

2️⃣ Run full pipeline
bash
Copy code
python run_pipeline.py
The script will:

Run prepare_dataset.py → generate masks

Run generate_yolo_labels.py → generate YOLO labels

Split train/val

Create dataset.yaml

Train YOLOv8 segmentation and print each epoch in real-time

3️⃣ Check training progress
During training, you’ll see live logs like:

python-repl
Copy code
Epoch    GPU_mem   box_loss   cls_loss   mask_loss   val/meanIoU
1/50     2.1G      0.411      0.184      0.089       0.712
2/50     2.2G      0.385      0.167      0.075       0.726
...
50/50    2.3G      0.105      0.041      0.023       0.912
4️⃣ Outputs
After running, your structure becomes:

swift
Copy code
prepared/
├── healthy/
│   ├── leaf_mask.png
│   └── ...
├── blight/
│   ├── *_leaf.png
│   ├── *_disease.png
│   └── ...
yolo_dataset/
├── images/train/...
├── images/val/...
├── labels/train/...
└── labels/val/...
dataset.yaml
runs/segment/train/weights/best.pt
🖥️ Optional: Run Streamlit app
Once training finishes, you can visualize detection results in real-time using your webcam:

bash
Copy code
streamlit run app.py
Example app usage:

Opens camera → detect if leaf or background (Stage 1)

If leaf → segment disease regions and classify type