import numpy as np
from collections import deque


class RPPGExtractor:
    def __init__(self, buffer_size=300, fps=30):
        """
        buffer_size: kitne frames store karne hain
                     (300 frames = 10 sec at 30fps)
        fps: webcam frame rate
        """
        self.buffer_size = buffer_size
        self.fps = fps

        self.r_buffer = deque(maxlen=buffer_size)
        self.g_buffer = deque(maxlen=buffer_size)
        self.b_buffer = deque(maxlen=buffer_size)

    def add_frame(self, roi):
        if roi is None or roi.size == 0:
            return False

        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])

        self.r_buffer.append(mean_r)
        self.g_buffer.append(mean_g)
        self.b_buffer.append(mean_b)

        return True

    def get_signal(self, method='GREEN'):
        if len(self.g_buffer) < 60:
            return None

        r = np.array(self.r_buffer)
        g = np.array(self.g_buffer)
        b = np.array(self.b_buffer)

        if method == 'GREEN':
            signal = g - np.mean(g)
            return signal

        elif method == 'CHROM':
            r_norm = r / (np.mean(r) + 1e-8)
            g_norm = g / (np.mean(g) + 1e-8)
            b_norm = b / (np.mean(b) + 1e-8)

            X = 3 * r_norm - 2 * g_norm
            Y = 1.5 * r_norm + g_norm - 1.5 * b_norm

            alpha = np.std(X) / (np.std(Y) + 1e-8)
            signal = X - alpha * Y
            return signal

        elif method == 'POS':
            r_norm = r / (np.mean(r) + 1e-8)
            g_norm = g / (np.mean(g) + 1e-8)
            b_norm = b / (np.mean(b) + 1e-8)

            S1 = r_norm - g_norm
            S2 = r_norm + g_norm - 2 * b_norm

            alpha = np.std(S1) / (np.std(S2) + 1e-8)
            H = S1 + alpha * S2
            signal = H - np.mean(H)
            return signal

        return None

    def get_buffer_length(self):
        return len(self.g_buffer)

    def is_ready(self):
        return len(self.g_buffer) >= 60

    def clear(self):
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
