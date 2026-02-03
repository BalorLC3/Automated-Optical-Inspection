from pathlib import Path
import os
import yaml
import torch
from ultralytics import YOLO

# -----------------------------
# Fix working directory
# -----------------------------
ROOT = Path(__file__).resolve().parents[3]  # metal_yolov9
os.chdir(ROOT)

# -----------------------------
# Paths
# -----------------------------
TRAIN_CFG = "services/inference/config/train.yaml"
DATASET_YAML = "services/inference/config/dataset.yaml"
MODEL_PATH = "services/inference/models/yolo26n.pt"

# -----------------------------
# Utils
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

# -----------------------------
# Train
# -----------------------------
def train() -> None:
    cfg = load_config(TRAIN_CFG)

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATASET_YAML,                 # ✅ PATH, not dict
        device=get_device(cfg["device"]),
        **cfg["training"],
        **cfg["run"],
    )

if __name__ == "__main__":
    train()
