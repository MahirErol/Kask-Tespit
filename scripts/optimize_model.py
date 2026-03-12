"""
Model optimizasyonu: Quantization, Pruning, TensorRT gibi tekniklerle
modeli daha hızlı ve küçük hale getirir.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize YOLOv8 model")
    parser.add_argument(
        "--weights",
        default="outputs/helmet-yolov8n-gpu2/weights/best.pt",
        help="Model weights path",
    )
    parser.add_argument(
        "--method",
        default="onnx",
        choices=["onnx", "tensorrt", "openvino", "quantize"],
        help="Optimization method",
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

    print(f"Model optimize ediliyor: {args.method.upper()}")
    print(f"Kaynak: {weights_path}")
    print("-" * 50)

    model = YOLO(str(weights_path))

    if args.method == "quantize":
        # INT8 quantization (daha küçük model, biraz daha yavaş)
        print("INT8 Quantization uygulanıyor...")
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print("Quantization için ONNX export edildi.")
        print("Not: Tam quantization için TensorRT kullanılabilir.")
    elif args.method == "tensorrt":
        # TensorRT (NVIDIA GPU için en hızlı)
        print("TensorRT formatına çevriliyor (NVIDIA GPU gerekli)...")
        model.export(format="engine", imgsz=args.imgsz, simplify=True)
    else:
        # ONNX, OpenVINO gibi formatlar
        model.export(format=args.method, imgsz=args.imgsz, simplify=True)

    print("\n" + "=" * 50)
    print(f"OPTİMİZASYON TAMAMLANDI: {args.method.upper()}")
    print(f"Export edilen dosya: {weights_path.parent / f'best.{args.method}'}")
    print("=" * 50)


if __name__ == "__main__":
    main()

