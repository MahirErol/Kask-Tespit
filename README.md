# Kask Tespiti (YOLOv8)

Bu proje motosiklet üzerindeki kişilerin kask takıp takmadığını YOLOv8 ile tespit etmek için hazırlandı.

## 📋 İçindekiler
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Eğitim](#eğitim)
- [Tahmin ve Test](#tahmin-ve-test)
- [Gerçek Zamanlı Kullanım](#gerçek-zamanlı-kullanım)
- [Model İyileştirme](#model-iyileştirme)
- [Sonraki Adımlar](#sonraki-adımlar)

## 🚀 Kurulum

```bash
pip install -r requirements.txt
```

**Not:** GTX 1650 Ti için CUDA'lı PyTorch kurulu olmalı:
```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"  # NumPy uyumluluğu için
```

## ⚡ Hızlı Başlangıç

### 1. Veri Yapısı
- `data/raw/train|valid|test/images` ve `labels` (YOLO formatı)
- `data/dataset.yaml` yol tanımları hazır

### 2. Model Eğitimi
```bash
python scripts/train_yolov8.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640
```

### 3. Test Setinde Tahmin
```bash
python scripts/predict_yolov8.py
```

### 4. Video Üzerinde Test
```bash
# Videoyu data/raw/video.mp4 olarak koy
python scripts/video_yolov8.py
```

### 5. Webcam ile Gerçek Zamanlı
```bash
python scripts/webcam_yolov8.py
```

## 🎓 Eğitim

### Temel Eğitim (YOLOv8n - Hızlı)
```bash
python scripts/train_yolov8.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --name helmet-yolov8n
```

### Daha İyi Model (YOLOv8s - Önerilen)
```bash
python scripts/train_better_model.py --model yolov8s.pt --epochs 100 --batch 4
```

### En İyi Model (YOLOv8m - En Güçlü)
```bash
python scripts/train_better_model.py --model yolov8m.pt --epochs 100 --batch 2
```

**Çıktılar:** `outputs/[run-name]/weights/best.pt`

## 🔍 Tahmin ve Test

### Test Setinde Tahmin
```bash
python scripts/predict_yolov8.py
```
Sonuçlar: `outputs/predictions/test/`

### Model Performans Analizi
```bash
python scripts/evaluate_yolov8.py
```
Confusion matrix, PR curve ve detaylı metrikler üretir.

## 📹 Gerçek Zamanlı Kullanım

### Video Üzerinde
```bash
# Videoyu data/raw/video.mp4 olarak koy
python scripts/video_yolov8.py
```

### Webcam ile
```bash
python scripts/webcam_yolov8.py
# Çıkmak için 'q' tuşuna bas
```

**Not:** Confidence threshold `0.2` olarak ayarlanmış (daha fazla tespit için). Script içinden değiştirebilirsin.

## 🚀 Model İyileştirme

### Daha İyi Performans İçin
1. **Daha büyük model eğit:**
   ```bash
   python scripts/train_better_model.py --model yolov8s.pt --epochs 100
   ```

2. **Model export (production için):**
   ```bash
   python scripts/export_model.py --weights outputs/helmet-yolov8s-improved/weights/best.pt --format onnx
   ```

3. **Hyperparameter tuning:** `scripts/train_yolov8.py` içindeki parametreleri dene.

## 📈 Sonraki Adımlar

### ✅ Tamamlananlar
- [x] Dataset yapısı oluşturuldu
- [x] YOLOv8n modeli eğitildi (mAP50: ~0.84)
- [x] Test setinde tahmin yapıldı
- [x] Video ve webcam scriptleri hazır
- [x] Confidence threshold optimize edildi

### 🔄 Yapılacaklar (Öneriler)

1. **Model İyileştirme:**
   - [ ] YOLOv8s/m modeli eğit (daha iyi performans)
   - [ ] Data augmentation artır
   - [ ] Hyperparameter tuning

2. **Deployment:**
   - [ ] Model export (ONNX, TensorRT)
   - [ ] Web API (Flask/FastAPI)
   - [ ] Mobil uygulama entegrasyonu

3. **Gelişmiş Özellikler:**
   - [ ] Çoklu kamera desteği
   - [ ] Veritabanı entegrasyonu (tespit kayıtları)
   - [ ] Alert sistemi (kasksız tespit edildiğinde)

4. **Dokümantasyon:**
   - [ ] API dokümantasyonu
   - [ ] Deployment rehberi
   - [ ] Kullanım örnekleri

## 📝 Notlar

- **Sınıflar:** `["With Helmet", "Without Helmet"]`, `nc: 2`
- **Girdi boyutu:** Varsayılan 640 (GTX 1650 Ti için uygun)
- **GPU:** CUDA 12.1 ile PyTorch kurulu olmalı
- **NumPy:** `<2.0` versiyonu kullanılmalı (uyumluluk için)

## 🐛 Sorun Giderme

- **Video açılmıyor:** Video H.264 formatında olmalı, gerekirse ffmpeg ile convert et
- **Kamera açılmıyor:** Başka uygulama kullanıyor olabilir, index'i değiştir (0→1)
- **CUDA hatası:** PyTorch CUDA versiyonunu kontrol et
- **NumPy hatası:** `pip install "numpy<2"` ile düzelt

## 📄 Lisans



