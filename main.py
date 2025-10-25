import sys
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class HandDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Counter with MediaPipe")
        self.setGeometry(100, 100, 800, 600)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support 2 hands for counting
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

        self.cap = cv2.VideoCapture(1)

        # print camera resolution
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {width} x {height}")

        # FPS tracking
        self.prev_frame_time = 0
        self.fps = 0
        self.fps_history = deque(maxlen=300)  # Store last 10 seconds at 30fps
        self.fps_times = deque(maxlen=300)  # Store timestamps

        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (~33 fps)

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create label to display video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        # Create stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_capture)
        layout.addWidget(self.stop_button)

    def draw_fps_chart(self, frame):
        """Draw FPS chart on the frame"""
        if len(self.fps_history) < 2:
            return

        h, w, c = frame.shape

        # Chart dimensions
        chart_width = 300
        chart_height = 100
        chart_x = w - chart_width - 20
        chart_y = 60

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (chart_x - 10, chart_y - 10),
            (chart_x + chart_width + 10, chart_y + chart_height + 10),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (chart_x, chart_y),
            (chart_x + chart_width, chart_y + chart_height),
            (255, 255, 255),
            2,
        )

        # Get FPS data
        fps_data = list(self.fps_history)
        if not fps_data:
            return

        max_fps = max(max(fps_data), 60)  # At least 60 for scale
        min_fps = 0

        # Draw grid lines
        for i in range(0, int(max_fps) + 1, 20):
            y = chart_y + chart_height - int((i / max_fps) * chart_height)
            cv2.line(
                frame, (chart_x, y), (chart_x + chart_width, y), (100, 100, 100), 1
            )
            cv2.putText(
                frame,
                str(i),
                (chart_x - 30, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        # Draw FPS line
        points = []
        for i, fps_val in enumerate(fps_data):
            x = chart_x + int((i / len(fps_data)) * chart_width)
            y = chart_y + chart_height - int((fps_val / max_fps) * chart_height)
            points.append((x, y))

        # Draw the line
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

        # Draw current FPS point
        if points:
            cv2.circle(frame, points[-1], 4, (0, 255, 255), -1)

    def is_finger_extended(self, landmarks, finger_id):
        """Check if a finger is extended"""
        if finger_id == 0:  # Thumb
            # Thumb is extended if tip is further from wrist than IP joint
            return landmarks[self.tip_ids[0]].x < landmarks[self.tip_ids[0] - 1].x
        else:
            # Other fingers: extended if tip is above PIP joint
            return (
                landmarks[self.tip_ids[finger_id]].y
                < landmarks[self.tip_ids[finger_id] - 2].y
            )

    def detect_taiwanese_number(self, hand_landmarks):
        """
        Detect Taiwanese hand counting (1-9):
        1 = index finger only
        2 = index + middle fingers
        3 = index + middle + ring fingers
        4 = index + middle + ring + pinky (no thumb)
        5 = all five fingers
        6 = thumb + pinky (hang loose sign)
        7 = thumb + index finger (like pinching)
        8 = thumb + index + middle fingers
        9 = thumb + index + middle + ring (no pinky)
        """
        landmarks = hand_landmarks.landmark

        # Check which fingers are extended
        fingers_extended = [self.is_finger_extended(landmarks, i) for i in range(5)]
        thumb, index, middle, ring, pinky = fingers_extended

        # Check for each number pattern
        if thumb and index and middle and ring and pinky:
            return 5  # All fingers
        elif not thumb and index and middle and ring and pinky:
            return 4  # All except thumb
        elif not thumb and index and middle and ring and not pinky:
            return 3  # Index, middle, ring
        elif not thumb and index and middle and not ring and not pinky:
            return 2  # Index and middle
        elif not thumb and index and not middle and not ring and not pinky:
            return 1  # Index only
        elif thumb and not index and not middle and not ring and pinky:
            return 6  # Thumb and pinky (hang loose)
        elif thumb and index and not middle and not ring and not pinky:
            return 7  # Thumb and index
        elif thumb and index and middle and not ring and not pinky:
            return 8  # Thumb, index, middle
        elif thumb and index and middle and ring and not pinky:
            return 9  # Thumb, index, middle, ring (no pinky)

        return None  # Unknown gesture

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Calculate FPS
        current_frame_time = time.time()
        if self.prev_frame_time != 0:
            self.fps = 1 / (current_frame_time - self.prev_frame_time)
            self.fps_history.append(self.fps)
            self.fps_times.append(current_frame_time)
        self.prev_frame_time = current_frame_time

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Variable to store the detected number
        detected_number = None

        # Draw hand landmarks and detect number
        if results.multi_hand_landmarks and results.multi_handedness:
            # Single hand: detect numbers 1-9
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    rgb_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                # Detect Taiwanese number (1-9)
                detected_number = self.detect_taiwanese_number(hand_landmarks)

            # Two hands: check for number 10 (all fingers of two hands extended)
            elif len(results.multi_hand_landmarks) == 2:
                hand1_landmarks = results.multi_hand_landmarks[0]
                hand2_landmarks = results.multi_hand_landmarks[1]

                # Draw landmarks for both hands
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )

                # Check if both hands show 5 fingers
                fingers_hand1 = [
                    self.is_finger_extended(hand1_landmarks.landmark, i)
                    for i in range(5)
                ]
                fingers_hand2 = [
                    self.is_finger_extended(hand2_landmarks.landmark, i)
                    for i in range(5)
                ]
                if all(fingers_hand1) and all(fingers_hand2):
                    detected_number = 10

        # Display the detected number (large and centered)
        if detected_number is not None:
            h, w, c = rgb_frame.shape
            cv2.putText(
                rgb_frame,
                str(detected_number),
                (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                6,
                (0, 255, 255),
                12,
            )
            cv2.putText(
                rgb_frame,
                f"Number: {detected_number}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 0),
                4,
            )

        # Display FPS
        h, w, c = rgb_frame.shape
        cv2.putText(
            rgb_frame,
            f"FPS: {int(self.fps)}",
            (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw FPS chart
        self.draw_fps_chart(rgb_frame)

        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale and display
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def stop_capture(self):
        self.timer.stop()
        self.cap.release()
        self.close()

    def closeEvent(self, event):
        # Clean up when window is closed
        self.timer.stop()
        self.cap.release()
        self.hands.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = HandDetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
