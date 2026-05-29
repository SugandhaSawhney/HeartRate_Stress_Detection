import numpy as np
from collections import deque


class StressAnalyzer:
    def __init__(self, history_size=30):
        self.bpm_history = deque(maxlen=history_size)

    def add_bpm(self, bpm):
        if bpm and 42 <= bpm <= 180:
            self.bpm_history.append(bpm)

    def calculate_hrv_features(self):
        if len(self.bpm_history) < 5:
            return None

        bpm_array = np.array(self.bpm_history)
        rr_intervals = (60000.0 / bpm_array)

        sdnn = np.std(rr_intervals)

        if len(rr_intervals) > 1:
            successive_diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        else:
            rmssd = 0

        if len(rr_intervals) > 1:
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            pnn50 = (nn50 / len(np.diff(rr_intervals))) * 100
        else:
            pnn50 = 0

        return {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'mean_bpm': np.mean(bpm_array),
            'bpm_std': np.std(bpm_array)
        }

    def get_stress_level(self):
        features = self.calculate_hrv_features()

        if features is None:
            return "Analyzing...", 0, {}

        rmssd = features['rmssd']
        sdnn = features['sdnn']
        mean_bpm = features['mean_bpm']

        stress_score = 0

        if rmssd < 15:
            stress_score += 50
        elif rmssd < 25:
            stress_score += 35
        elif rmssd < 40:
            stress_score += 20
        else:
            stress_score += 5

        if mean_bpm > 100:
            stress_score += 30
        elif mean_bpm > 85:
            stress_score += 20
        elif mean_bpm > 70:
            stress_score += 10
        else:
            stress_score += 0

        if sdnn < 20:
            stress_score += 20
        elif sdnn < 40:
            stress_score += 10
        else:
            stress_score += 0

        stress_score = min(100, stress_score)

        if stress_score < 30:
            level = "Low"
        elif stress_score < 60:
            level = "Medium"
        else:
            level = "High"

        return level, stress_score, features

    def get_wellness_advice(self, stress_level):
        advice_map = {
            "Low": "Great! Your body is relaxed. Keep it up!",
            "Medium": "Moderate stress detected. Try deep breathing.",
            "High": "High stress! Take a break, breathe slowly."
        }
        return advice_map.get(stress_level, "Keep still for better reading...")

    def is_ready(self):
        return len(self.bpm_history) >= 5
