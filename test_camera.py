import cv2
import threading

ENTRY_CAMERA = "http://192.0.0.4:8080/video"
EXIT_CAMERA = "http://10.21.57.87:8080/video"


def test_camera(url, window_name):

    print(f"Connecting: {window_name}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"❌ {window_name} connection failed")
        return

    print(f"✅ {window_name} connected")

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"❌ {window_name} frame read failed")
            break

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(window_name)


t1 = threading.Thread(
    target=test_camera,
    args=(ENTRY_CAMERA, "ENTRY CAMERA")
)

t2 = threading.Thread(
    target=test_camera,
    args=(EXIT_CAMERA, "EXIT CAMERA")
)

t1.start()
t2.start()

t1.join()
t2.join()

cv2.destroyAllWindows()