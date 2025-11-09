# fix_ultralytics_path.py
import json, os
cfg_path = os.path.expanduser("~\\AppData\\Roaming\\Ultralytics\\settings.json")
cfg = {
    "datasets_dir": "C:/Users/Admin/Desktop/CV",
    "weights_dir": "C:/Users/Admin/Desktop/CV/runs",
    "runs_dir": "C:/Users/Admin/Desktop/CV/runs",
    "uuid": "",
    "sync": False
}
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
print("✅ settings.json fixed successfully at", cfg_path)
