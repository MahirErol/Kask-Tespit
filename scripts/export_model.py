"""
Eğitilmiş modeli farklı formatlara export eder (ONNX, TensorRT, CoreML vb.)
Production deployment için kullanışlı.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLOv8 model")
    parser.add_argument(
        "--weights",
        default="outputs/helmet-yolov8n-gpu2/weights/best.pt",
        help="Model weights path (relative to project root)",
    )
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "torchscript", "tensorrt", "coreml", "openvino"],
        help="Export format",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / args.weights

    if not weights_path.exists():
        print(f"Model bulunamadı: {weights_path}")
        return

    print(f"Model export ediliyor: {args.format.upper()}")
    print(f"Kaynak: {weights_path}")
    print("-" * 50)

    model = YOLO(str(weights_path))
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        simplify=True,
    )

    print("\n" + "=" * 50)
    print(f"EXPORT TAMAMLANDI: {args.format.upper()}")
    print(f"Export edilen dosya: {weights_path.parent / f'best.{args.format}'}")
    print("=" * 50)


if __name__ == "__main__":
    main()

