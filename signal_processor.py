import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


class SignalProcessor:
    def __init__(self, fps=30):
        self.fps = fps
        self.MIN_HR = 42
        self.MAX_HR = 180

    def bandpass_filter(self, sig, lowcut=0.7, highcut=3.0):
        """
        Bandpass filter - sirf heartbeat frequency range rakhta hai
        0.7 Hz = 42 BPM (minimum normal heart rate)
        3.0 Hz = 180 BPM (maximum during exercise)
        """
        nyquist = self.fps / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist

        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))

        if low >= high:
            return sig

        b, a = signal.butter(4, [low, high], btype='band')

        try:
            filtered = signal.filtfilt(b, a, sig)  # Zero phase filtering
            return filtered
        except Exception:
            return sig

    def calculate_bpm_fft(self, sig):
        """
        FFT se heart rate calculate karo
        Peak frequency × 60 = BPM
        """
        if sig is None or len(sig) < 30:
            return None

        filtered_sig = self.bandpass_filter(sig)

        N = len(filtered_sig)
        fft_result = np.abs(fft(filtered_sig))
        frequencies = fftfreq(N, d=1.0/self.fps)

        pos_mask = frequencies > 0
        frequencies = frequencies[pos_mask]
        fft_result = fft_result[pos_mask]

        valid_mask = (frequencies >= 0.7) & (frequencies <= 3.0)

        if not np.any(valid_mask):
            return None

        valid_freqs = frequencies[valid_mask]
        valid_fft = fft_result[valid_mask]

        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]

        bpm = peak_freq * 60

        if self.MIN_HR <= bpm <= self.MAX_HR:
            return round(bpm, 1)
        return None

    def calculate_bpm_peak(self, sig):
        """
        Peak detection se BPM - alternative method
        """
        if sig is None or len(sig) < 30:
            return None

        filtered_sig = self.bandpass_filter(sig)

        mean_val = np.mean(filtered_sig)
        std_val = np.std(filtered_sig) + 1e-8
        normalized = (filtered_sig - mean_val) / std_val

        min_distance = int(self.fps * 0.4)
        peaks, properties = signal.find_peaks(
            normalized,
            distance=min_distance,
            height=0.3,
            prominence=0.5
        )

        if len(peaks) < 2:
            return None

        rr_intervals = np.diff(peaks) / self.fps
        mean_rr = np.mean(rr_intervals)

        if mean_rr <= 0:
            return None

        bpm = 60.0 / mean_rr

        if self.MIN_HR <= bpm <= self.MAX_HR:
            return round(bpm, 1)
        return None

    def get_bpm(self, sig):
        """
        Both methods try karo, average lo
        """
        bpm_fft = self.calculate_bpm_fft(sig)
        bpm_peak = self.calculate_bpm_peak(sig)

        if bpm_fft and bpm_peak:
            # Dono close hain toh average
            if abs(bpm_fft - bpm_peak) < 15:
                return round((bpm_fft + bpm_peak) / 2, 1)
            else:
                return bpm_fft  # FFT more reliable
        elif bpm_fft:
            return bpm_fft
        elif bpm_peak:
            return bpm_peak
        return None

    def get_signal_quality(self, sig):
        """
        Signal kitna clean hai (0-100%)
        """
        if sig is None or len(sig) < 30:
            return 0

        filtered = self.bandpass_filter(sig)

        signal_power = np.var(filtered)
        noise = sig - filtered
        noise_power = np.var(noise)

        if noise_power == 0:
            return 100

        snr = signal_power / (noise_power + 1e-8)
        quality = min(100, int(snr * 20))
        return quality
