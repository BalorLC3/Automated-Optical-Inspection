from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "services" / "inference" / "config" / "train.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train() -> None:
    cfg = load_config(CONFIG_PATH)

    model_path = BASE_DIR / cfg["model"]
    data_yaml = BASE_DIR / cfg["data"]

    if not model_path.exists():
        print(f"Pretrained model not found. Downloading to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        YOLO("yolov26n.pt").save(str(model_path))
        
    assert data_yaml.exists()

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        device=get_device(cfg["device"]),
        **cfg["training"],
        **cfg["run"],
    )


if __name__ == "__main__":
    train()
