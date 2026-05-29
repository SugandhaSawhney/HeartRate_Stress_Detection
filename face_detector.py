import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Forehead points
        self.FOREHEAD_POINTS = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378
        ]
        # Left cheek points
        self.LEFT_CHEEK = [116, 117, 118, 119, 120, 121, 128, 198]
        # Right cheek points
        self.RIGHT_CHEEK = [345, 346, 347, 348, 349, 350, 357, 422]

    def get_forehead_roi(self, frame):
        """
        Forehead + both cheeks ka combined ROI — better accuracy
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        def get_coords(points):
            coords = []
            for idx in points:
                lm = face_landmarks.landmark[idx]
                coords.append([int(lm.x * w), int(lm.y * h)])
            return np.array(coords)

        forehead = get_coords(self.FOREHEAD_POINTS)
        left_cheek = get_coords(self.LEFT_CHEEK)
        right_cheek = get_coords(self.RIGHT_CHEEK)

        # Combine all regions
        all_points = np.vstack([forehead, left_cheek, right_cheek])

        x_min = max(0, np.min(all_points[:, 0]) - 5)
        x_max = min(w, np.max(all_points[:, 0]) + 5)
        y_min = max(0, np.min(all_points[:, 1]) - 5)
        y_max = min(h, np.max(all_points[:, 1]) + 5)

        # Create mask — skin only, ignore eyes/mouth
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [forehead], 255)
        cv2.fillPoly(mask, [left_cheek], 255)
        cv2.fillPoly(mask, [right_cheek], 255)

        # Extract masked pixels
        masked_frame = frame.copy()
        masked_frame[mask == 0] = 0
        roi = masked_frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None, None

        # Draw on annotated frame
        annotated_frame = frame.copy()
        cv2.polylines(
            annotated_frame, [forehead], True, (0, 255, 100), 1
        )
        cv2.polylines(
            annotated_frame, [left_cheek], True, (0, 255, 100), 1
        )
        cv2.polylines(
            annotated_frame, [right_cheek], True, (0, 255, 100), 1
        )
        cv2.putText(
            annotated_frame, "ROI Active",
            (x_min, y_min - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 100), 1
        )

        return roi, annotated_frame

    def get_cheek_roi(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        CHEEK_POINTS = [
            116, 117, 118, 119, 120, 121, 128,
            198, 217, 174, 177, 215
        ]
        cheek_coords = []
        for idx in CHEEK_POINTS:
            lm = face_landmarks.landmark[idx]
            cheek_coords.append([int(lm.x * w), int(lm.y * h)])

        cheek_coords = np.array(cheek_coords)
        x_min = max(0, np.min(cheek_coords[:, 0]))
        x_max = min(w, np.max(cheek_coords[:, 0]))
        y_min = max(0, np.min(cheek_coords[:, 1]))
        y_max = min(h, np.max(cheek_coords[:, 1]))

        return frame[y_min:y_max, x_min:x_max]
