"""
Gerçek zamanlı video stream işleme (RTSP, HTTP stream vb.)
Birden fazla kaynaktan gelen videoları paralel işler.
"""
import argparse
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO


def process_stream(stream_url: str, model: YOLO, stream_id: int, conf: float = 0.25):
    """Tek bir stream'i işle"""
    cap = cv2.VideoCapture(stream_url)
    window_name = f"Stream {stream_id} - Helmet Detection (q ile çık)"

    if not cap.isOpened():
        print(f"Stream {stream_id} açılamadı: {stream_url}")
        return

    print(f"Stream {stream_id} başlatıldı: {stream_url}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream {stream_id} kare okunamadı, yeniden deneniyor...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            continue

        # Tahmin yap
        results = model.predict(source=frame, imgsz=640, conf=conf, verbose=False)
        annotated = results[0].plot()

        # İstatistikleri göster
        with_helmet = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
        without_helmet = sum(1 for box in results[0].boxes if int(box.cls[0]) == 1)

        cv2.putText(
            annotated,
            f"With Helmet: {with_helmet} | Without: {without_helmet}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Process multiple video streams")
    parser.add_argument("--streams", nargs="+", required=True, help="Stream URL'leri (RTSP, HTTP vb.)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--weights",
        default="outputs/helmet-yolov8n-gpu2/weights/best.pt",
        help="Model weights path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / args.weights

    if not weights_path.exists():
        print(f"Model bulunamadı: {weights_path}")
        return

    model = YOLO(str(weights_path))

    # Her stream için thread oluştur
    threads = []
    for i, stream_url in enumerate(args.streams):
        thread = threading.Thread(
            target=process_stream,
            args=(stream_url, model, i + 1, args.conf),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    print(f"{len(args.streams)} stream işleniyor...")
    print("Çıkmak için herhangi bir pencerede 'q' tuşuna bas")

    # Thread'leri bekle
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

