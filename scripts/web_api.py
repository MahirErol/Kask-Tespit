"""
Basit Flask web API - Modeli HTTP üzerinden kullanmak için.
POST isteği ile görsel gönderip sonuç alabilirsin.
"""
from pathlib import Path

import cv2
from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# Model yükleme
project_root = Path(__file__).resolve().parents[1]
weights = project_root / "outputs" / "helmet-yolov8n-gpu2" / "weights" / "best.pt"
model = YOLO(str(weights))


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Kask Tespiti API",
        "endpoints": {
            "/predict": "POST - Görsel gönder, tespit sonuçlarını al",
            "/health": "GET - API durumu"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "loaded"})


@app.route("/predict", methods=["POST"])
def predict():
    """Görsel gönderip kask tespiti yap"""
    if "file" not in request.files:
        return jsonify({"error": "Görsel dosyası bulunamadı"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    # Görseli oku
    import numpy as np
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Görsel okunamadı"}), 400

    # Tahmin yap
    results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    result = results[0]

    # Sonuçları formatla
    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": result.names[cls],
            "confidence": round(conf, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })

    # İstatistikler
    with_helmet = sum(1 for d in detections if d["class"] == "With Helmet")
    without_helmet = sum(1 for d in detections if d["class"] == "Without Helmet")

    return jsonify({
        "detections": detections,
        "summary": {
            "total": len(detections),
            "with_helmet": with_helmet,
            "without_helmet": without_helmet,
        },
    })


if __name__ == "__main__":
    print("API başlatılıyor...")
    print("Kullanım: http://localhost:5000/predict")
    print("Örnek curl komutu:")
    print('curl -X POST -F "file=@image.jpg" http://localhost:5000/predict')
    app.run(host="0.0.0.0", port=5000, debug=True)

