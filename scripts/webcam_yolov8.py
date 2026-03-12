from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights = project_root / "outputs" / "helmet-yolov8n-gpu2" / "weights" / "best.pt"

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı. Doğru kameranın takılı ve boş olduğundan emin ol.")
        return

    # Pencere ismi
    window_name = "Helmet Detection - Webcam (q ile çık)"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare okunamadı, döngü sonlandırılıyor.")
            break

        # YOLO ile tahmin (conf=0.2 ile daha fazla tespit)
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.2,
            verbose=False,
        )

        # İlk sonucu al ve üzerine çiz
        annotated_frame = results[0].plot()

        # Ekranda daha geniş açı görmek için görüntüyü küçült (0.7 oranını istersen değiştir)
        scale = 0.7
        resized = cv2.resize(
            annotated_frame,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imshow(window_name, resized)

        # q'ya basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


