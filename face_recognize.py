import cv2
import pickle
import numpy as np
import threading
import time

from attendance import login_person, logout_person


# ============================================================
# CONFIGURATION
# ============================================================

YUNET_MODEL = "models/face_detection_yunet_2026may.onnx"
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
DATABASE = "face_database.pkl"

# Camera 1 → ENTRY
ENTRY_CAMERA = "http://192.0.0.4:8080/video"

# Camera 2 → EXIT
EXIT_CAMERA = 0

COSINE_THRESHOLD = 0.363

# Display size of EACH camera
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

# Combined window size
WINDOW_WIDTH = DISPLAY_WIDTH * 2
WINDOW_HEIGHT = DISPLAY_HEIGHT


# ============================================================
# LOAD FACE DATABASE
# ============================================================

try:

    with open(DATABASE, "rb") as file:
        database = pickle.load(file)

except FileNotFoundError:

    print("❌ face_database.pkl not found")
    print("Run: python face_train.py")
    exit()


if not database:

    print("❌ Face database is empty")
    exit()


print("\n👥 Registered People:")

for name in database:

    print(f"   → {name}")


# ============================================================
# CREATE FACE ENGINE
# ============================================================

def create_face_engine():

    try:

        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        recognizer = cv2.FaceRecognizerSF.create(
            SFACE_MODEL,
            ""
        )

        return detector, recognizer

    except Exception as e:

        print("❌ Could not create face engine")
        print(e)

        return None, None


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(frame, detector, recognizer):

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    try:

        _, faces = detector.detect(frame)

    except cv2.error as e:

        print("❌ YuNet detection error:")
        print(e)

        return []


    results = []


    if faces is None:

        return results


    # Process every detected face
    for face in faces:

        x, y, w, h = face[:4].astype(int)

        x = max(0, x)
        y = max(0, y)

        w = min(w, width - x)
        h = min(h, height - y)


        # ----------------------------------------------------
        # ALIGN FACE
        # ----------------------------------------------------

        try:

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

        except Exception:

            continue


        # ----------------------------------------------------
        # EXTRACT SFACE FEATURE
        # ----------------------------------------------------

        try:

            feature = recognizer.feature(
                aligned_face
            )

        except Exception:

            continue


        feature = feature / (
            np.linalg.norm(feature) + 1e-8
        )


        # ----------------------------------------------------
        # MATCH WITH REGISTERED DATABASE
        # ----------------------------------------------------

        best_name = "Unknown"
        best_score = -1


        for name, stored_feature in database.items():

            stored_feature = np.asarray(
                stored_feature,
                dtype=np.float32
            ).reshape(1, -1)


            score = recognizer.match(
                feature,
                stored_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )


            if score > best_score:

                best_score = score
                best_name = name


        # ----------------------------------------------------
        # APPLY THRESHOLD
        # ----------------------------------------------------

        if best_score >= COSINE_THRESHOLD:

            name = best_name

        else:

            name = "Unknown"


        detection_confidence = float(
            face[-1]
        )


        results.append({

            "name": name,

            "score": float(best_score),

            "confidence": detection_confidence,

            "box": (x, y, w, h)

        })


    return results


# ============================================================
# CAMERA PROCESSOR
# ============================================================

class CameraProcessor:

    def __init__(self, url, camera_type):

        self.url = url

        self.camera_type = camera_type

        self.frame = None

        self.results = []

        self.running = True

        self.lock = threading.Lock()

        self.processed = set()

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )


    # --------------------------------------------------------
    # START CAMERA
    # --------------------------------------------------------

    def start(self):

        self.thread.start()


    # --------------------------------------------------------
    # CAMERA LOOP
    # --------------------------------------------------------

    def run(self):

        print(
            f"\n📷 Connecting to "
            f"{self.camera_type} camera..."
        )


        # Each camera has its own detector + recognizer
        detector, recognizer = create_face_engine()


        if detector is None:

            self.running = False

            return


        cap = cv2.VideoCapture(
            self.url
        )


        if not cap.isOpened():

            print(
                f"❌ {self.camera_type} camera "
                f"connection failed"
            )

            self.running = False

            return


        print(
            f"✅ {self.camera_type} camera connected"
        )


        while self.running:

            ret, frame = cap.read()


            if not ret or frame is None:

                print(
                    f"❌ {self.camera_type} "
                    f"camera frame failed"
                )

                time.sleep(0.1)

                continue


            # ------------------------------------------------
            # FACE RECOGNITION
            # ------------------------------------------------

            results = recognize_face(
                frame,
                detector,
                recognizer
            )


            # ------------------------------------------------
            # ATTENDANCE
            # ------------------------------------------------

            for result in results:

                name = result["name"]


                # Ignore unknown faces
                if name == "Unknown":

                    continue


                # Prevent repeated DB calls
                if name in self.processed:

                    continue


                try:

                    # ========================================
                    # ENTRY → LOGIN
                    # ========================================

                    if self.camera_type == "ENTRY":

                        success, message = login_person(
                            name
                        )


                        if success:

                            print(
                                f"🟢 {name} → LOGIN"
                            )

                        else:

                            print(
                                f"ℹ️ {name} → {message}"
                            )


                    # ========================================
                    # EXIT → LOGOUT
                    # ========================================

                    else:

                        success, message = logout_person(
                            name
                        )


                        if success:

                            print(
                                f"🔴 {name} → LOGOUT"
                            )

                        else:

                            print(
                                f"ℹ️ {name} → {message}"
                            )


                    self.processed.add(name)


                except Exception as e:

                    print(
                        f"❌ Database error "
                        f"for {name}"
                    )

                    print(e)


            # ------------------------------------------------
            # STORE FRAME
            # ------------------------------------------------

            with self.lock:

                self.frame = frame.copy()

                self.results = results


        cap.release()


    # --------------------------------------------------------
    # GET LATEST FRAME
    # --------------------------------------------------------

    def get_data(self):

        with self.lock:

            if self.frame is None:

                return None, []

            return (
                self.frame.copy(),
                list(self.results)
            )


    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        self.running = False


# ============================================================
# DRAW UI
# ============================================================

def draw_results(
    frame,
    results,
    camera_type
):

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if camera_type == "ENTRY":

        title = "CAMERA 1 - ENTRY / LOGIN"

    else:

        title = "CAMERA 2 - EXIT / LOGOUT"


    cv2.rectangle(

        frame,

        (0, 0),

        (frame.shape[1], 45),

        (30, 30, 30),

        -1

    )


    cv2.putText(

        frame,

        title,

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------------
    # Draw detected faces
    # --------------------------------------------------------

    for result in results:

        name = result["name"]

        score = result["score"]

        confidence = result["confidence"]

        x, y, w, h = result["box"]


        # ----------------------------------------------------
        # Known / Unknown color
        # ----------------------------------------------------

        if name == "Unknown":

            color = (0, 0, 255)

            status = "UNKNOWN"

        else:

            color = (0, 255, 0)

            if camera_type == "ENTRY":

                status = "ENTRY / LOGIN"

            else:

                status = "EXIT / LOGOUT"


        # ----------------------------------------------------
        # Face rectangle
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            color,

            2

        )


        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        cv2.putText(

            frame,

            name,

            (x, max(70, y - 30)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            color,

            2

        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        cv2.putText(

            frame,

            status,

            (x, max(95, y - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )


        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Similarity: {score:.3f}",

            (x, y + h + 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # Detection confidence
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Detection: {confidence:.2f}",

            (x, y + h + 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1

        )


    return frame


# ============================================================
# MAIN
# ============================================================

print("\n==============================================")
print("     SMART ATTENDANCE - TWO CAMERA SYSTEM")
print("==============================================")

print("Camera 1 : ENTRY → LOGIN")
print("Camera 2 : EXIT  → LOGOUT")

print("Detection : YuNet")
print("Recognition: SFace")
print("Database  : MySQL")

print("\nPress Q to quit")

print("==============================================\n")


# ============================================================
# CREATE CAMERAS
# ============================================================

entry_camera = CameraProcessor(

    ENTRY_CAMERA,

    "ENTRY"

)


exit_camera = CameraProcessor(

    EXIT_CAMERA,

    "EXIT"

)


# ============================================================
# START BOTH CAMERAS
# ============================================================

entry_camera.start()

exit_camera.start()


# ============================================================
# CREATE ONE COMBINED WINDOW
# ============================================================

WINDOW_NAME = (
    "SMART ATTENDANCE - "
    "TWO CAMERA MONITOR"
)


cv2.namedWindow(

    WINDOW_NAME,

    cv2.WINDOW_NORMAL

)


cv2.resizeWindow(

    WINDOW_NAME,

    WINDOW_WIDTH,

    WINDOW_HEIGHT

)


# ============================================================
# MAIN DISPLAY LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Get Entry frame
    # --------------------------------------------------------

    entry_frame, entry_results = (
        entry_camera.get_data()
    )


    # --------------------------------------------------------
    # Get Exit frame
    # --------------------------------------------------------

    exit_frame, exit_results = (
        exit_camera.get_data()
    )


    # ========================================================
    # ENTRY FRAME
    # ========================================================

    if entry_frame is not None:

        entry_frame = draw_results(

            entry_frame,

            entry_results,

            "ENTRY"

        )


        entry_frame = cv2.resize(

            entry_frame,

            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )

        )

    else:

        entry_frame = np.zeros(

            (
                DISPLAY_HEIGHT,
                DISPLAY_WIDTH,
                3
            ),

            dtype=np.uint8

        )


        cv2.putText(

            entry_frame,

            "ENTRY CAMERA",

            (190, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            2

        )


    # ========================================================
    # EXIT FRAME
    # ========================================================

    if exit_frame is not None:

        exit_frame = draw_results(

            exit_frame,

            exit_results,

            "EXIT"

        )


        exit_frame = cv2.resize(

            exit_frame,

            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )

        )

    else:

        exit_frame = np.zeros(

            (
                DISPLAY_HEIGHT,
                DISPLAY_WIDTH,
                3
            ),

            dtype=np.uint8

        )


        cv2.putText(

            exit_frame,

            "EXIT CAMERA",

            (190, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            2

        )


    # ========================================================
    # JOIN TWO EQUAL FRAMES
    # ========================================================

    combined_frame = np.hstack(

        (
            entry_frame,
            exit_frame
        )

    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        combined_frame

    )


    # ========================================================
    # Q = QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# STOP CAMERAS
# ============================================================

entry_camera.stop()

exit_camera.stop()


entry_camera.thread.join(
    timeout=2
)

exit_camera.thread.join(
    timeout=2
)


cv2.destroyAllWindows()


print("\n==============================================")
print("🛑 Two-camera recognition stopped")
print("==============================================")