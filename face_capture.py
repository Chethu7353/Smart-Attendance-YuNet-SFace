import cv2
import os
import time

# ============================================
# FACE DATASET CAPTURE - YuNet
# ============================================

MODEL = "models/face_detection_yunet_2026may.onnx"

TARGET_IMAGES = 150

name = input("Enter person name: ").strip()

if not name:
    print("❌ Name cannot be empty.")
    exit()

folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Start numbering after existing images
existing = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

count = len(existing)

# --------------------------------------------
# Load YuNet
# --------------------------------------------

detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# --------------------------------------------
# Laptop webcam
# --------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open laptop webcam")
    exit()

print("\n========================================")
print(f"Registering: {name}")
print(f"Target images: {TARGET_IMAGES}")
print("========================================")

print("""
Move your face slowly:

  LEFT ↔ RIGHT
       ↑
       |
     FACE
       |
       ↓
     UP/DOWN

Also slightly change your distance and expression.

Press Q to stop.
""")

last_capture = 0

while count < TARGET_IMAGES:

    ret, frame = cap.read()

    if not ret:
        print("❌ Could not read webcam")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        # Capture only when exactly one face is visible
        if len(faces) == 1:

            face = faces[0]

            x, y, w, h = face[:4].astype(int)

            # Keep coordinates valid
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            # Add a small margin around face
            margin = int(0.15 * min(w, h))

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)

            face_crop = frame[y1:y2, x1:x2]

            current_time = time.time()

            # Avoid saving identical frames too quickly
            if current_time - last_capture >= 0.10:

                count += 1

                filename = os.path.join(
                    folder,
                    f"{count:03d}.jpg"
                )

                cv2.imwrite(filename, face_crop)

                last_capture = current_time

            # Draw detection
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name}: {count}/{TARGET_IMAGES}",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Only ONE face should be visible",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "Face not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "SFace Dataset Capture",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print("✅ Capture completed")
print(f"Person: {name}")
print(f"Images: {count}")
print(f"Folder: {folder}")
print("========================================")