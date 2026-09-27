
import sys
import cv2
import pickle
import time
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from attendance import login_person, logout_person


# ============================================================
# CONFIGURATION
# ============================================================

YUNET_MODEL = "models/face_detection_yunet_2026may.onnx"
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
DATABASE = "face_database.pkl"

# YOUR CURRENT CAMERA SETUP
# Camera 1 = desktop/laptop webcam -> ENTRY / LOGIN
ENTRY_CAMERA = "http://10.21.57.73:8080/video"

# Camera 2 = mobile IP camera -> EXIT / LOGOUT
EXIT_CAMERA = "http://192.0.0.4:8080/video"

COSINE_THRESHOLD = 0.363

# Run recognition every N frames to reduce CPU usage.
RECOGNITION_INTERVAL = 3

# Minimum time between database attempts for the same person/camera.
DB_COOLDOWN = 8.0


# ============================================================
# LOAD FACE DATABASE
# ============================================================

try:
    with open(DATABASE, "rb") as file:
        database = pickle.load(file)

except FileNotFoundError:
    raise SystemExit(
        "face_database.pkl not found. Run: python face_train.py"
    )

if not database:
    raise SystemExit("face_database.pkl is empty. Run: python face_train.py")

print("\nRegistered People:")
for name in database:
    print("  ->", name)


# ============================================================
# FACE ENGINE
# ============================================================

def create_face_engine():
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )

    recognizer = cv2.FaceRecognizerSF.create(
        SFACE_MODEL,
        "",
    )

    return detector, recognizer


def recognize_faces(frame, detector, recognizer):
    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    try:
        _, faces = detector.detect(frame)
    except cv2.error:
        return []

    results = []

    if faces is None:
        return results

    for face in faces:

        x, y, w, h = face[:4].astype(int)

        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        try:
            aligned_face = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned_face)
        except Exception:
            continue

        feature = feature / (np.linalg.norm(feature) + 1e-8)

        best_name = "Unknown"
        best_score = -1.0

        for name, stored_feature in database.items():

            stored_feature = np.asarray(
                stored_feature,
                dtype=np.float32
            ).reshape(1, -1)

            try:
                score = recognizer.match(
                    feature,
                    stored_feature,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= COSINE_THRESHOLD:
            name = best_name
        else:
            name = "Unknown"

        results.append(
            {
                "name": name,
                "score": float(best_score),
                "confidence": float(face[-1]),
                "box": (x, y, w, h),
            }
        )

    return results


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(QThread):

    frame_ready = Signal(object, object)
    person_event = Signal(str, str, str)
    camera_status = Signal(str)
    error = Signal(str)

    def __init__(self, source, camera_type):
        super().__init__()

        self.source = source
        self.camera_type = camera_type

        self.running = True
        self.frame_count = 0

        self.last_db_action = {}

    def run(self):

        try:
            detector, recognizer = create_face_engine()
        except Exception as exc:
            self.error.emit(
                f"{self.camera_type}: AI engine error - {exc}"
            )
            return

        self.camera_status.emit(
            f"{self.camera_type}: Connecting..."
        )

        cap = cv2.VideoCapture(self.source)

        # Reduce buffering where supported.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.camera_status.emit(
                f"{self.camera_type}: Connection failed"
            )
            cap.release()
            return

        self.camera_status.emit(
            f"{self.camera_type}: Camera connected"
        )

        results = []

        while self.running:

            ret, frame = cap.read()

            if not ret or frame is None:
                self.camera_status.emit(
                    f"{self.camera_type}: Frame unavailable"
                )
                time.sleep(0.1)
                continue

            self.frame_count += 1

            # Recognition is intentionally not run on every frame.
            if self.frame_count % RECOGNITION_INTERVAL == 0:

                results = recognize_faces(
                    frame,
                    detector,
                    recognizer,
                )

                # Console diagnostics: useful during initial deployment.
                print(
                    f"{self.camera_type}: {len(results)} face(s) detected",
                    flush=True,
                )
                for result in results:
                    print(
                        f"  {result['name']} | "
                        f"similarity={result['score']:.3f} | "
                        f"detection={result['confidence']:.2f}",
                        flush=True,
                    )

                self.process_attendance(results)

            # Draw recognition results on the frame.
            display_frame = frame.copy()

            self.draw_results(
                display_frame,
                results,
            )

            self.frame_ready.emit(
                display_frame,
                results,
            )

        cap.release()

        self.camera_status.emit(
            f"{self.camera_type}: Stopped"
        )

    def process_attendance(self, results):

        now = time.monotonic()

        for result in results:

            name = result["name"]

            # Unknown faces are displayed but never written to DB.
            if name == "Unknown":
                continue

            key = name

            last_time = self.last_db_action.get(key, 0)

            if now - last_time < DB_COOLDOWN:
                continue

            try:

                if self.camera_type == "ENTRY":

                    success, message = login_person(name)

                    if success:
                        self.person_event.emit(
                            name,
                            "LOGIN",
                            "Entry attendance recorded",
                        )
                    else:
                        self.person_event.emit(
                            name,
                            message,
                            "Entry detected",
                        )

                else:

                    success, message = logout_person(name)

                    if success:
                        self.person_event.emit(
                            name,
                            "LOGOUT",
                            "Exit attendance recorded",
                        )
                    else:
                        self.person_event.emit(
                            name,
                            message,
                            "Exit detected",
                        )

                self.last_db_action[key] = now

            except Exception as exc:
                self.error.emit(
                    f"Database error for {name}: {exc}"
                )

    def draw_results(self, frame, results):

        title = (
            "ENTRY / LOGIN"
            if self.camera_type == "ENTRY"
            else "EXIT / LOGOUT"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 42),
            (15, 23, 42),
            -1,
        )

        cv2.putText(
            frame,
            title,
            (15, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        for result in results:

            name = result["name"]
            score = result["score"]
            x, y, w, h = result["box"]

            if name == "Unknown":

                color = (0, 0, 255)
                label = "UNKNOWN"

            else:

                color = (0, 220, 80)
                label = name

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

            cv2.rectangle(
                frame,
                (x, max(45, y - 30)),
                (x + max(w, 150), y),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x + 6, max(67, y - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Similarity: {score:.3f}",
                (x, y + h + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def stop(self):
        self.running = False


# ============================================================
# CAMERA CARD
# ============================================================

class CameraCard(QFrame):

    def __init__(self, title, mode):
        super().__init__()

        self.setObjectName("cameraCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setObjectName("cameraTitle")

        badge = QLabel(mode)
        badge.setObjectName("modeBadge")

        header.addWidget(title_label)
        header.addStretch()
        header.addWidget(badge)

        layout.addLayout(header)

        self.camera_view = QLabel("CAMERA OFFLINE")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(400, 280)
        self.camera_view.setObjectName("cameraView")

        layout.addWidget(self.camera_view)

        info = QHBoxLayout()

        self.person_label = QLabel("Person: —")
        self.action_label = QLabel("Status: Waiting")

        self.person_label.setObjectName("infoLabel")
        self.action_label.setObjectName("infoLabel")

        info.addWidget(self.person_label)
        info.addStretch()
        info.addWidget(self.action_label)

        layout.addLayout(info)


# ============================================================
# MAIN APPLICATION
# ============================================================

class AttendanceApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.entry_worker = None
        self.exit_worker = None
        self.system_running = False

        self.setWindowTitle(
            "Smart Attendance Monitoring System"
        )

        self.resize(1280, 820)

        self.setup_ui()
        self.apply_style()

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main = QVBoxLayout(central)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(16)

        # HEADER
        header = QHBoxLayout()

        title_box = QVBoxLayout()

        title = QLabel("Smart Attendance Monitoring")
        title.setObjectName("mainTitle")

        subtitle = QLabel(
            "AI-powered face recognition • Entry & Exit monitoring"
        )
        subtitle.setObjectName("subtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addLayout(title_box)
        header.addStretch()

        self.system_status = QLabel("●  SYSTEM OFF")
        self.system_status.setObjectName("statusOff")

        header.addWidget(self.system_status)

        main.addLayout(header)

        # CAMERAS
        cameras = QHBoxLayout()
        cameras.setSpacing(16)

        self.entry_card = CameraCard(
            "Camera 1 — Entry",
            "LOGIN",
        )

        self.exit_card = CameraCard(
            "Camera 2 — Exit",
            "LOGOUT",
        )

        cameras.addWidget(self.entry_card)
        cameras.addWidget(self.exit_card)

        main.addLayout(cameras)

        # CONTROL
        control = QFrame()
        control.setObjectName("controlPanel")

        control_layout = QHBoxLayout(control)

        self.system_message = QLabel(
            "System is OFF. Press TURN ON to start attendance."
        )
        self.system_message.setObjectName("systemMessage")

        control_layout.addWidget(self.system_message)
        control_layout.addStretch()

        self.toggle_button = QPushButton("TURN ON")
        self.toggle_button.setObjectName("onButton")
        self.toggle_button.setMinimumSize(160, 48)

        self.toggle_button.clicked.connect(
            self.toggle_system
        )

        control_layout.addWidget(self.toggle_button)

        main.addWidget(control)

        # FOOTER
        footer = QLabel(
            "YuNet • SFace • MySQL"
        )

        footer.setAlignment(Qt.AlignCenter)
        footer.setObjectName("footer")

        main.addWidget(footer)

    # ========================================================
    # ON / OFF
    # ========================================================

    def toggle_system(self):

        if self.system_running:
            self.stop_system()
        else:
            self.start_system()

    def start_system(self):

        self.system_running = True

        self.system_status.setText(
            "●  SYSTEM ON"
        )
        self.system_status.setObjectName(
            "statusOn"
        )

        self.toggle_button.setText(
            "TURN OFF"
        )
        self.toggle_button.setObjectName(
            "offButton"
        )

        self.system_message.setText(
            "Starting Entry and Exit cameras..."
        )

        self.apply_style()

        # ENTRY = desktop webcam
        self.entry_worker = CameraWorker(
            ENTRY_CAMERA,
            "ENTRY",
        )

        self.entry_worker.frame_ready.connect(
            self.update_entry_frame
        )

        self.entry_worker.person_event.connect(
            self.entry_event
        )

        self.entry_worker.camera_status.connect(
            self.entry_status
        )

        self.entry_worker.error.connect(
            self.show_error
        )

        # EXIT = mobile IP camera
        self.exit_worker = CameraWorker(
            EXIT_CAMERA,
            "EXIT",
        )

        self.exit_worker.frame_ready.connect(
            self.update_exit_frame
        )

        self.exit_worker.person_event.connect(
            self.exit_event
        )

        self.exit_worker.camera_status.connect(
            self.exit_status
        )

        self.exit_worker.error.connect(
            self.show_error
        )

        self.entry_worker.start()
        self.exit_worker.start()

    def stop_system(self):

        self.system_running = False

        if self.entry_worker is not None:

            self.entry_worker.stop()
            self.entry_worker.wait(3000)
            self.entry_worker = None

        if self.exit_worker is not None:

            self.exit_worker.stop()
            self.exit_worker.wait(3000)
            self.exit_worker = None

        self.system_status.setText(
            "●  SYSTEM OFF"
        )
        self.system_status.setObjectName(
            "statusOff"
        )

        self.toggle_button.setText(
            "TURN ON"
        )
        self.toggle_button.setObjectName(
            "onButton"
        )

        self.entry_card.camera_view.clear()
        self.entry_card.camera_view.setText(
            "CAMERA OFFLINE"
        )

        self.exit_card.camera_view.clear()
        self.exit_card.camera_view.setText(
            "CAMERA OFFLINE"
        )

        self.entry_card.person_label.setText(
            "Person: —"
        )

        self.exit_card.person_label.setText(
            "Person: —"
        )

        self.entry_card.action_label.setText(
            "Status: Waiting"
        )

        self.exit_card.action_label.setText(
            "Status: Waiting"
        )

        self.system_message.setText(
            "System is OFF. Press TURN ON to start attendance."
        )

        self.apply_style()

    # ========================================================
    # FRAME DISPLAY
    # ========================================================

    def display_frame(self, frame, label):

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        height, width, channels = frame_rgb.shape

        bytes_per_line = channels * width

        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        label.setPixmap(pixmap)

    # ========================================================
    # ENTRY GUI
    # ========================================================

    def update_entry_frame(self, frame, results):

        self.display_frame(
            frame,
            self.entry_card.camera_view,
        )

        known = [
            r["name"]
            for r in results
            if r["name"] != "Unknown"
        ]

        if known:
            self.entry_card.person_label.setText(
                "Person: " + ", ".join(known)
            )
        else:
            self.entry_card.person_label.setText(
                "Person: —"
            )

        self.entry_card.action_label.setText(
            "Status: Monitoring"
        )

    def entry_event(self, name, action, message):

        self.entry_card.person_label.setText(
            f"Person: {name}"
        )

        self.entry_card.action_label.setText(
            f"{action}"
        )

        self.system_message.setText(
            f"ENTRY: {name} → {action}"
        )

    def entry_status(self, message):

        self.entry_card.action_label.setText(
            message.replace("ENTRY: ", "")
        )

    # ========================================================
    # EXIT GUI
    # ========================================================

    def update_exit_frame(self, frame, results):

        self.display_frame(
            frame,
            self.exit_card.camera_view,
        )

        known = [
            r["name"]
            for r in results
            if r["name"] != "Unknown"
        ]

        if known:
            self.exit_card.person_label.setText(
                "Person: " + ", ".join(known)
            )
        else:
            self.exit_card.person_label.setText(
                "Person: —"
            )

        self.exit_card.action_label.setText(
            "Status: Monitoring"
        )

    def exit_event(self, name, action, message):

        self.exit_card.person_label.setText(
            f"Person: {name}"
        )

        self.exit_card.action_label.setText(
            f"{action}"
        )

        self.system_message.setText(
            f"EXIT: {name} → {action}"
        )

    def exit_status(self, message):

        self.exit_card.action_label.setText(
            message.replace("EXIT: ", "")
        )

    # ========================================================
    # ERRORS
    # ========================================================

    def show_error(self, message):

        print("ERROR:", message)

        self.system_message.setText(
            message
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        if self.system_running:
            self.stop_system()

        event.accept()

    # ========================================================
    # STYLE
    # ========================================================

    def apply_style(self):

        self.setStyleSheet("""
            QMainWindow {
                background: #0b1120;
            }

            QWidget {
                color: #e5e7eb;
                font-family: "Segoe UI";
            }

            #mainTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
            }

            #subtitle {
                font-size: 13px;
                color: #94a3b8;
            }

            #statusOff {
                color: #94a3b8;
                font-size: 14px;
                font-weight: 700;
                padding: 10px 16px;
                border: 1px solid #334155;
                border-radius: 20px;
            }

            #statusOn {
                color: #4ade80;
                font-size: 14px;
                font-weight: 700;
                padding: 10px 16px;
                border: 1px solid #166534;
                border-radius: 20px;
            }

            #cameraCard {
                background: #111827;
                border: 1px solid #263244;
                border-radius: 16px;
            }

            #cameraTitle {
                font-size: 17px;
                font-weight: 700;
                color: #f8fafc;
            }

            #modeBadge {
                background: #1e293b;
                color: #cbd5e1;
                padding: 5px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 700;
            }

            #cameraView {
                background: #020617;
                border: 1px solid #1e293b;
                border-radius: 12px;
                color: #64748b;
                font-size: 15px;
                font-weight: 600;
            }

            #infoLabel {
                color: #94a3b8;
                font-size: 12px;
            }

            #controlPanel {
                background: #111827;
                border: 1px solid #263244;
                border-radius: 14px;
            }

            #systemMessage {
                color: #94a3b8;
                font-size: 12px;
            }

            QPushButton {
                border-radius: 10px;
                padding: 11px 20px;
                font-size: 13px;
                font-weight: 700;
            }

            #onButton {
                background: #22c55e;
                color: #052e16;
                border: none;
            }

            #onButton:hover {
                background: #4ade80;
            }

            #offButton {
                background: #ef4444;
                color: white;
                border: none;
            }

            #offButton:hover {
                background: #f87171;
            }

            #footer {
                color: #64748b;
                font-size: 11px;
            }
        """)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AttendanceApp()
    window.show()

    sys.exit(app.exec())
