import numpy as np
import cv2
import os
import json
from face_detector import FaceDetector
from rppg_extractor import RPPGExtractor
from signal_processor import SignalProcessor


def read_ground_truth_dataset2(gt_path):
    """
    UBFC Dataset2 ground_truth.txt format:
    Line 1: PPG signal values
    Line 2: HR values
    Line 3: Timestamps
    """
    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print(f"  Unexpected format in {gt_path}")
            return None

        hr_values = []
        for val in lines[1].strip().split():
            try:
                hr = float(val)
                if 42 <= hr <= 180:
                    hr_values.append(hr)
            except ValueError:
                continue

        if hr_values:
            return np.mean(hr_values)
        return None

    except Exception as e:
        print(f"  Error reading {gt_path}: {e}")
        return None


def process_video(video_path, method='CHROM'):
    """
    Video se predicted BPM nikalo
    """
    detector = FaceDetector()
    extractor = RPPGExtractor(buffer_size=300, fps=30)
    processor = SignalProcessor(fps=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")

    frame_count = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 2 != 0:  # Har 2nd frame
            continue

        roi, _ = detector.get_forehead_roi(frame)
        if roi is not None:
            extractor.add_frame(roi)
            success_count += 1

    cap.release()
    print(f"  Face detected: {success_count}/{frame_count//2} frames")

    if not extractor.is_ready():
        print(f"  Not enough data")
        return None

    sig = extractor.get_signal(method=method)
    bpm = processor.get_bpm(sig)
    quality = processor.get_signal_quality(sig)
    print(f"  Signal quality: {quality}%")
    return bpm


def calculate_metrics(errors):
    errors = np.array(errors)
    abs_errors = np.abs(errors)
    return {
        'MAE': round(np.mean(abs_errors), 2),
        'RMSE': round(np.sqrt(np.mean(errors**2)), 2),
        'STD': round(np.std(abs_errors), 2),
        'within_5bpm_%': round(np.sum(abs_errors <= 5) / len(abs_errors) * 100, 1),
        'within_10bpm_%': round(np.sum(abs_errors <= 10) / len(abs_errors) * 100, 1),
        'num_subjects': len(errors)
    }


def run_evaluation(dataset_path, max_subjects=None, method='CHROM'):
    print("=" * 60)
    print(f"UBFC-rPPG Evaluation | Method: {method}")
    print("=" * 60)

    subjects = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if max_subjects:
        subjects = subjects[:max_subjects]

    results = []
    errors = []

    for i, subject in enumerate(subjects):
        subject_path = os.path.join(dataset_path, subject)
        video_path = os.path.join(subject_path, 'vid.avi')
        gt_path = os.path.join(subject_path, 'ground_truth.txt')

        if not os.path.exists(video_path) or not os.path.exists(gt_path):
            print(f"\n[{i+1}] {subject}: Files missing, skipping")
            continue

        print(f"\n[{i+1}/{len(subjects)}] {subject}")

        gt_bpm = read_ground_truth_dataset2(gt_path)
        if gt_bpm is None:
            print("  Ground truth failed, skipping")
            continue
        print(f"  GT BPM:        {gt_bpm:.1f}")

        pred_bpm = process_video(video_path, method=method)
        if pred_bpm is None:
            print("  Prediction failed, skipping")
            continue
        print(f"  Predicted BPM: {pred_bpm:.1f}")

        error = pred_bpm - gt_bpm
        print(f"  Error:         {error:+.1f} BPM")

        results.append({
            'subject': subject,
            'ground_truth': round(gt_bpm, 1),
            'predicted': round(pred_bpm, 1),
            'error': round(error, 1),
            'abs_error': round(abs(error), 1)
        })
        errors.append(error)

    if not errors:
        print("\nNo valid results!")
        return

    metrics = calculate_metrics(errors)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Subjects: {metrics['num_subjects']}")
    print(f"MAE: {metrics['MAE']} BPM")
    print(f"RMSE: {metrics['RMSE']} BPM")
    print(f"Within ±5 BPM: {metrics['within_5bpm_%']}%")
    print(f"Within ±10 BPM: {metrics['within_10bpm_%']}%")
    print("=" * 60)

    with open(f'results_{method}.json', 'w') as f:
        json.dump({'method': method, 'metrics': metrics,
                   'per_subject': results}, f, indent=2)

    print(f"\nSaved: results_{method}.json")


if __name__ == "__main__":
    DATASET_PATH = "UBFC-rPPG/DATASET_2"

    run_evaluation(DATASET_PATH, max_subjects=5, method='CHROM')
