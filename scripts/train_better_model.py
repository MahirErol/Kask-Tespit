"""
Daha iyi performans için YOLOv8s veya YOLOv8m modeli eğitir.
GTX 1650 Ti için batch size ve image size ayarlanmış.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train better YOLOv8 model for helmet detection")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        choices=["yolov8s.pt", "yolov8m.pt"],
        help="Model size: yolov8s (small) veya yolov8m (medium)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epoch sayısı (daha fazla = daha iyi)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (1650 Ti için 4 önerilir)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", default=None, help="Run name (otomatik oluşturulur)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "dataset.yaml"

    # Otomatik isim oluştur
    if args.name is None:
        model_name = args.model.replace(".pt", "")
        args.name = f"helmet-{model_name}-improved"

    print(f"Eğitim başlıyor: {args.model}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Image Size: {args.imgsz}")
    print(f"Run name: {args.name}")
    print("-" * 50)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project=project_root / "outputs",
        patience=20,  # Early stopping için
        save=True,
        plots=True,
    )

    print("\n" + "=" * 50)
    print("EĞİTİM TAMAMLANDI!")
    print(f"Model kaydedildi: outputs/{args.name}/weights/best.pt")
    print("=" * 50)


if __name__ == "__main__":
    main()

