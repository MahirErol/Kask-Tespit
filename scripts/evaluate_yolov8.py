from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights = project_root / "outputs" / "helmet-yolov8n-gpu2" / "weights" / "best.pt"
    data_yaml = project_root / "data" / "dataset.yaml"

    model = YOLO(str(weights))

    # Test setinde detaylı değerlendirme
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        plots=True,  # Confusion matrix ve PR curve çizdir
        save_json=True,  # JSON formatında kaydet
    )

    print("\n" + "=" * 50)
    print("MODEL PERFORMANS ÖZETİ")
    print("=" * 50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 50)

    print(f"\nDetaylı sonuçlar kaydedildi: {project_root / 'outputs' / 'helmet-yolov8n-gpu2'}")


if __name__ == "__main__":
    main()

