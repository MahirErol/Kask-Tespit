from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights = project_root / "outputs" / "helmet-yolov8n-gpu2" / "weights" / "best.pt"

    # Burayı kendi video yoluna göre değiştir
    # Örnek: C:/Users/Mahir/Desktop/kask-tespit/data/raw/video/deneme.mp4
    video_path = project_root / "data" / "raw" / "video.mp4"

    if not video_path.exists():
        print(f"Video bulunamadı: {video_path}")
        print("Lütfen videonu bu yola kopyala ya da script içindeki yolu güncelle.")
        return

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Video açılamadı, yolun doğru olduğundan emin ol.")
        return

    window_name = "Helmet Detection - Video (q ile çık)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # conf=0.2 ile daha fazla tespit yakala (daha hassas)
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.2,
            verbose=False,
        )

        annotated = results[0].plot()

        # Görüntüyü ekranda daha "uzaktan" görmek için küçült (0.7 oranını istersen değiştir)
        scale = 0.7
        resized = cv2.resize(
            annotated,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imshow(window_name, resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


