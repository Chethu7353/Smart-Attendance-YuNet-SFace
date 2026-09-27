import cv2
import os
import pickle
import numpy as np

# ============================================
# SFace Face Embedding Database
# ============================================

MODEL = "models/face_recognition_sface_2021dec.onnx"
DATASET = "dataset"
OUTPUT = "face_database.pkl"

# Load SFace
recognizer = cv2.FaceRecognizerSF.create(
    MODEL,
    ""
)

print("✅ SFace model loaded")

database = {}

people = [
    name for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
]

if not people:
    print("❌ No people found in dataset")
    exit()

for person in people:

    person_folder = os.path.join(DATASET, person)

    embeddings = []

    print(f"\n👤 Processing: {person}")

    for filename in os.listdir(person_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # SFace expects an aligned face.
        # For this first version, use the captured face crop.
        face = cv2.resize(image, (112, 112))

        feature = recognizer.feature(face)

        embeddings.append(feature.flatten())

    if embeddings:

        # Average embedding for this person
        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        # Normalize
        mean_embedding = mean_embedding / (
            np.linalg.norm(mean_embedding) + 1e-8
        )

        database[person] = mean_embedding

        print(
            f"   ✅ {len(embeddings)} images processed"
        )

    else:

        print(
            f"   ❌ No usable images for {person}"
        )


# Save database
with open(OUTPUT, "wb") as file:

    pickle.dump(
        database,
        file
    )

print("\n================================")
print("✅ FACE TRAINING COMPLETED")
print("================================")

for name, embedding in database.items():

    print(
        f"{name}: embedding shape = "
        f"{embedding.shape}"
    )

print(f"\n💾 Saved: {OUTPUT}")