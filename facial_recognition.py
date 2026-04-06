import face_recognition
import cv2
import os
import pickle
import numpy as np

# Configuration
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"
TOLERANCE = 0.6
MODEL = "hog" 
OUTPUT_DIR = "output"
OUTPUT_VIDEO_FILE = os.path.join(OUTPUT_DIR, "facial_recognition_output.avi")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
FACE_CROP_DIR = os.path.join(OUTPUT_DIR, "face_crops")


def normalize_for_face_recognition(image):
    """Return a contiguous uint8 RGB image accepted by face_recognition/dlib."""
    if image is None:
        return None

    # Ensure 8-bit depth first; some cameras return float/16-bit frames.
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 255)
        elif np.issubdtype(image.dtype, np.integer):
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8, copy=False)

    # Convert to RGB with exactly 3 channels.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            return None
    else:
        return None

    return np.ascontiguousarray(image, dtype=np.uint8)

def build_database():
    known_encodings = []
    known_names = []
    
    print(f"Scanning '{KNOWN_FACES_DIR}' for known faces...")
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created directory '{KNOWN_FACES_DIR}'. Please add image files there and run again.")
        return

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            person_name = os.path.splitext(filename)[0].split('_')[0]
            
            print(f"Processing {filename}...")
            cv_image = cv2.imread(filepath)
            
            if cv_image is None:
                print(f"Skipped {filename}: OpenCV could not read the image file.")
                continue

            image = normalize_for_face_recognition(cv_image)
            if image is None:
                print(f"  -> Skipped {filename}: Unsupported image format.")
                continue
            
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 1:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"  -> Added {person_name} to database.")
            elif len(encodings) > 1:
                print(f"  -> Skipped {filename}: Found {len(encodings)} faces.")
            else:
                print(f"  -> Warning: No faces found in {filename}")

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    print(f"Saved {len(known_names)} encodings to {ENCODINGS_FILE}.")

def load_database():
    if not os.path.exists(ENCODINGS_FILE):
        build_database()
    else:
        print(f"Found existing {ENCODINGS_FILE}. Loading...")
        
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.loads(f.read())
    return {"encodings": [], "names": []}

def run_recognition():
    data = load_database()
    if not data["encodings"]:
        print("No known encodings loaded. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(FACE_CROP_DIR, exist_ok=True)

    print("Starting webcam... Press 'q' to quit.")
    video_capture = cv2.VideoCapture(0)

    video_writer = None
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        frame_count += 1
        
        # Scale down for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = normalize_for_face_recognition(small_frame)
        if rgb_small_frame is None:
            print(f"Skipped webcam frame {frame_count}: Unsupported image format.")
            continue

        try:
            face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            print(f"CRASH on webcam frame {frame_count}. Error: {e}")
            break

        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=TOLERANCE)
            name = "Unknown" 

            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data["names"][best_match_index]

            face_names.append(name)

        # Draw boxes
        for index, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names), start=1):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            if name != "Unknown":
                face_crop = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]
                if face_crop.size > 0:
                    crop_path = os.path.join(
                        FACE_CROP_DIR,
                        f"frame_{frame_count:06d}_face_{index}_{name}.jpg",
                    )
                    cv2.imwrite(crop_path, face_crop)

        cv2.imshow('Facial Recognition', frame)

        if video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 20.0, (width, height))

        if video_writer is not None and video_writer.isOpened():
            video_writer.write(frame)

        if frame_count % 30 == 0:
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(snapshot_path, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()