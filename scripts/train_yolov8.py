import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for helmet detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", default="helmet-yolov8", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "dataset.yaml"

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project=project_root / "outputs",
    )


if __name__ == "__main__":
    main()


