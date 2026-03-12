from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # GPU ile eğitilen son model
    weights = project_root / "outputs" / "helmet-yolov8n-gpu2" / "weights" / "best.pt"
    # Test verisinde deneme
    source = project_root / "data" / "raw" / "test" / "images"
    output_dir = project_root / "outputs" / "predictions"

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        save=True,
        project=output_dir,
        name="test",
        imgsz=640,
    )


if __name__ == "__main__":
    main()