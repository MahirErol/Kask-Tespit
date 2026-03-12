"""
Tespit sonuçlarını veritabanına kaydet ve alert sistemi ekle.
SQLite kullanır (basit ve taşınabilir).
"""
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


class DetectionLogger:
    def __init__(self, db_path: str = "detections.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Veritabanı tablosunu oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                with_helmet INTEGER DEFAULT 0,
                without_helmet INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                alert_sent INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()
        print(f"Veritabanı hazır: {self.db_path}")

    def log_detection(
        self,
        with_helmet: int,
        without_helmet: int,
        image_path: str = None,
        alert_threshold: int = 1,
    ):
        """Tespit sonuçlarını kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        total = with_helmet + without_helmet
        alert_sent = 1 if without_helmet >= alert_threshold else 0

        cursor.execute(
            """
            INSERT INTO detections 
            (timestamp, image_path, with_helmet, without_helmet, total_detections, alert_sent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, image_path, with_helmet, without_helmet, total, alert_sent),
        )

        conn.commit()
        conn.close()

        # Alert gönder (kasksız tespit edildiyse)
        if alert_sent:
            self.send_alert(without_helmet, timestamp)

    def send_alert(self, count: int, timestamp: str):
        """Alert gönder (kasksız tespit edildiğinde)"""
        print(f"⚠️  ALERT: {count} kasksız kişi tespit edildi! ({timestamp})")
        # Buraya email, SMS, webhook vb. eklenebilir

    def get_statistics(self, hours: int = 24):
        """Son N saatteki istatistikleri getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_detections,
                SUM(with_helmet) as total_with_helmet,
                SUM(without_helmet) as total_without_helmet,
                SUM(alert_sent) as total_alerts
            FROM detections
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """,
            (hours,),
        )

        result = cursor.fetchone()
        conn.close()

        return {
            "total_detections": result[0] or 0,
            "total_with_helmet": result[1] or 0,
            "total_without_helmet": result[2] or 0,
            "total_alerts": result[3] or 0,
        }


def process_with_logging(
    source: str,
    model: YOLO,
    logger: DetectionLogger,
    save_images: bool = False,
    output_dir: Path = None,
):
    """Görsel/video işle ve veritabanına kaydet"""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source) if source.isdigit() or source.endswith((".mp4", ".avi")) else None

    frame_count = 0
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Tek görsel
            frame = cv2.imread(source)
            if frame is None:
                print(f"Görsel okunamadı: {source}")
                return

        # Tahmin yap
        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
        result = results[0]

        # İstatistikleri hesapla
        with_helmet = sum(1 for box in result.boxes if int(box.cls[0]) == 0)
        without_helmet = sum(1 for box in result.boxes if int(box.cls[0]) == 1)

        # Veritabanına kaydet
        image_path = None
        if save_images and output_dir:
            image_path = str(output_dir / f"detection_{frame_count:06d}.jpg")
            annotated = result.plot()
            cv2.imwrite(image_path, annotated)

        logger.log_detection(with_helmet, without_helmet, image_path)

        frame_count += 1

        if not cap:
            break  # Tek görsel için döngüden çık

    if cap:
        cap.release()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Model ile tespit yap ve veritabanına kaydet")
    parser.add_argument("--source", required=True, help="Görsel/video yolu veya kamera index")
    parser.add_argument("--weights", default="outputs/helmet-yolov8n-gpu2/weights/best.pt")
    parser.add_argument("--db", default="detections.db", help="Veritabanı dosyası")
    parser.add_argument("--save-images", action="store_true", help="Tespit edilen görselleri kaydet")
    parser.add_argument("--output-dir", default="outputs/detections", help="Kayıt klasörü")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / args.weights

    if not weights_path.exists():
        print(f"Model bulunamadı: {weights_path}")
        return

    model = YOLO(str(weights_path))
    logger = DetectionLogger(args.db)

    output_dir = project_root / args.output_dir if args.save_images else None

    print("İşleme başlıyor...")
    process_with_logging(args.source, model, logger, args.save_images, output_dir)

    # İstatistikleri göster
    stats = logger.get_statistics(24)
    print("\n" + "=" * 50)
    print("SON 24 SAAT İSTATİSTİKLERİ")
    print("=" * 50)
    print(f"Toplam Tespit: {stats['total_detections']}")
    print(f"Kasklı: {stats['total_with_helmet']}")
    print(f"Kasksız: {stats['total_without_helmet']}")
    print(f"Alert Sayısı: {stats['total_alerts']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

