import cv2

MODEL = "models/face_detection_yunet_2026may.onnx"

# Load YuNet neural-network face detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("✅ YuNet Neural Network loaded")

# Laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open laptop webcam")
    exit()

print("✅ Laptop webcam connected")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Could not read webcam")
        break

    height, width = frame.shape[:2]

    # Match detector input size to webcam frame
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            confidence = face[-1]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Face: {confidence * 100:.1f}%",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw 5 facial landmarks
            landmarks = face[4:14].reshape(5, 2).astype(int)

            for lx, ly in landmarks:

                cv2.circle(
                    frame,
                    (lx, ly),
                    2,
                    (255, 0, 0),
                    -1
                )

    cv2.imshow(
        "YuNet NN Face Detection - Laptop Webcam",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("🛑 Detection stopped")