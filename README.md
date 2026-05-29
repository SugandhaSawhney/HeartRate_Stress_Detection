# 🫀 Non-Invasive Heart Rate & Stress Monitoring using rPPG

> Real-time heart rate and stress level estimation from facial video using Remote Photoplethysmography (rPPG) — no wearables, no physical contact required.

---

## 📌 Overview

This project implements a **contactless physiological monitoring system** that detects heart rate (BPM) and stress levels from a standard webcam feed. It uses subtle skin color variations caused by blood flow — invisible to the naked eye but detectable through computer vision and signal processing.

---

## ✨ Key Features

- 🎯 **Multi-region ROI extraction** — Forehead + left cheek + right cheek for higher accuracy
- **3 rPPG algorithms** — GREEN, CHROM, POS (selectable)
- **Dual BPM estimation** — FFT + Peak detection, averaged for robustness
- **HRV-based stress analysis** — SDNN, RMSSD, pNN50 features
- **Real-time signal quality scoring** — SNR-based quality metric
- **Wellness feedback** — Actionable advice based on stress level

---

## 🏗️ Project Architecture

```
rPPG-Monitor/
├── face_detector.py       # MediaPipe face mesh → multi-region ROI extraction
├── rppg_extractor.py      # RGB signal buffering + GREEN / CHROM / POS algorithms
├── signal_processor.py    # Bandpass filter + FFT + peak detection → BPM
├── stress_analyzer.py     # HRV features (SDNN, RMSSD, pNN50) → stress level
└── test_accuracy.py       # UBFC-rPPG dataset evaluation pipeline
```

---

## ⚙️ How It Works

```
Webcam Frame
     ↓
FaceDetector  →  MediaPipe Face Mesh (478 landmarks)
     ↓              Forehead ROI  +  Left Cheek  +  Right Cheek
RPPGExtractor →  Mean R, G, B per frame  →  Circular buffer (300 frames)
     ↓              Apply: GREEN / CHROM / POS algorithm
SignalProcessor → Bandpass filter (0.7–3.0 Hz = 42–180 BPM)
     ↓              FFT peak frequency × 60 = BPM
     ↓              Peak detection → inter-beat intervals → BPM
     ↓              Ensemble: average both if |diff| < 15 BPM
StressAnalyzer →  BPM history → RR intervals → HRV features
     ↓              SDNN + RMSSD + mean BPM → Stress Score (0–100)
Output        →  BPM  |  Signal Quality %  |  Stress Level  |  Advice
```

---

## rPPG Algorithms Implemented

| Algorithm | Method | Strength |
|-----------|--------|----------|
| **GREEN** | Raw green channel mean | Simple baseline |
| **CHROM** | Chrominance-based (de Haan 2013) | Robust to motion |
| **POS** | Plane-orthogonal-to-skin (Wang 2017) | Best accuracy |

**CHROM formula:**
```
X = 3R - 2G
Y = 1.5R + G - 1.5B
signal = X - α·Y    where α = std(X)/std(Y)
```

**POS formula:**
```
S1 = R_norm - G_norm
S2 = R_norm + G_norm - 2·B_norm
signal = S1 + α·S2    where α = std(S1)/std(S2)
```

---

## Stress Analysis — HRV Features

| Feature | Description | Low Stress | High Stress |
|---------|-------------|-----------|------------|
| **RMSSD** | Root mean square of successive RR differences | > 40 ms | < 15 ms |
| **SDNN** | Standard deviation of RR intervals | > 40 ms | < 20 ms |
| **pNN50** | % of successive RR diffs > 50ms | High | Low |
| **Mean BPM** | Average heart rate | 60–70 | > 100 |

Stress Score (0–100) is computed as a weighted combination of the above features.

---

### Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| OpenCV | ≥ 4.5 |
| MediaPipe | ≥ 0.10 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |

---

## Usage

### Real-time Webcam

```python
from face_detector import FaceDetector
from rppg_extractor import RPPGExtractor
from signal_processor import SignalProcessor
from stress_analyzer import StressAnalyzer
import cv2

detector = FaceDetector()
extractor = RPPGExtractor(buffer_size=300, fps=30)
processor = SignalProcessor(fps=30)
stress = StressAnalyzer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi, annotated = detector.get_forehead_roi(frame)

    if roi is not None:
        extractor.add_frame(roi)

    if extractor.is_ready():
        sig = extractor.get_signal(method='CHROM')  # or 'POS', 'GREEN'
        bpm = processor.get_bpm(sig)
        quality = processor.get_signal_quality(sig)

        if bpm:
            stress.add_bpm(bpm)
            level, score, features = stress.get_stress_level()
            print(f"BPM: {bpm} | Quality: {quality}% | Stress: {level} ({score})")

cap.release()
```

### Dataset Evaluation (UBFC-rPPG) (ONGOING)

```bash
# Place dataset at UBFC-rPPG/DATASET_2/
python test_accuracy.py
```

---

## Signal Processing Pipeline

```
Raw RGB signal  →  Bandpass Filter (Butterworth 4th order, 0.7–3.0 Hz)
                →  FFT  →  Peak frequency  →  BPM₁
                →  Peak detection (min_distance=fps×0.4)  →  BPM₂
                →  Ensemble: |BPM₁ - BPM₂| < 15  →  average
                                                else  →  BPM₁ (FFT)
```

**Why bandpass 0.7–3.0 Hz?**
- 0.7 Hz = 42 BPM (resting minimum)
- 3.0 Hz = 180 BPM (exercise maximum)
- Removes motion artifacts, lighting flicker, breathing noise

---

## Clinical Relevance

This project demonstrates the feasibility of **non-invasive physiological monitoring** — a core challenge in modern healthcare AI:

- **Remote Patient Monitoring** — ICU/home care without wearables
- **Stress Detection** — Mental health, workplace wellness
- **HRV Analysis** — Cardiovascular risk assessment
- **Blood Oxygen Estimation** — Extension toward SpO2 (ratiometric R/IR model)

The rPPG signal is the foundational signal for non-invasive blood analysis — the same photoplethysmographic principle used in pulse oximeters, extended here to work at a distance.

---

## File Reference

| File | Description |
|------|-------------|
| `face_detector.py` | MediaPipe face mesh, forehead + cheek ROI with masking |
| `rppg_extractor.py` | Frame buffer, RGB mean extraction, GREEN/CHROM/POS |
| `signal_processor.py` | Butterworth filter, FFT BPM, peak detection BPM, SNR quality |
| `stress_analyzer.py` | HRV features (SDNN, RMSSD, pNN50), stress scoring, wellness advice |
| `test_accuracy.py` | UBFC-rPPG Dataset 2 evaluation — MAE, RMSE, within-5BPM% |

---

## Future Work

- [ ] Deep learning rPPG model (PhysNet / EfficientPhys)
- [ ] SpO2 estimation using R/IR ratio
- [ ] Real-time dashboard with live signal plot
- [ ] Mobile deployment (TFLite / CoreML)
- [ ] Multi-person simultaneous monitoring

---

## 📚 References

1. de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering.*
2. Wang, W., den Brinker, A.C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering.*
3. Bobbia, S., et al. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters.* (UBFC-rPPG Dataset)

---
