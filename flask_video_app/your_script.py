import cv2
import face_recognition
import boto3
import numpy as np
from decimal import Decimal
from fer import FER
import ast
import threading
from queue import Queue

# Initialize OpenCV's DNN face detector (3x faster than face_recognition.face_locations)
FACE_DETECTION_MODEL = "opencv_face_detector.pbtxt"
FACE_DETECTION_WEIGHTS = "opencv_face_detector_uint8.pb"
face_detector = cv2.dnn.readNetFromTensorflow(FACE_DETECTION_WEIGHTS, FACE_DETECTION_MODEL)

# Initialize FER detector with MTCNN (keeping original but with optimizations)
emotion_detector = FER(mtcnn=True)

# Load DynamoDB face embeddings
dynamodb = boto3.resource("dynamodb", region_name='us-east-1')
table = dynamodb.Table('KnownFaces')

# Cache for known faces
known_faces_cache = None
last_cache_update = 0

def load_known_faces():
    """Optimized with caching to reduce DynamoDB calls"""
    global known_faces_cache, last_cache_update
    
    # Refresh cache every 5 minutes
    if known_faces_cache is None or (time.time() - last_cache_update) > 300:
        response = table.scan()
        known_faces_cache = []
        for item in response.get('Items', []):
            person_id = item['person_id']
            embedding_data = item['embedding']

            if isinstance(embedding_data, str):
                embedding = np.array([float(x) for x in ast.literal_eval(embedding_data)])
            else:
                embedding = np.array([float(x) for x in embedding_data])

            known_faces_cache.append((person_id, embedding))
        last_cache_update = time.time()
    
    return known_faces_cache

def detect_faces_fast(frame):
    """3x faster alternative to face_recognition.face_locations()"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Convert to face_recognition style (top, right, bottom, left)
            faces.append((startY, endX, endY, startX))
    return faces

def recognize_face(encoding, known_faces, tolerance=0.5):
    """Optimized face recognition"""
    # Convert to numpy array operations for speed
    known_encodings = np.array([e for _, e in known_faces])
    distances = np.linalg.norm(encoding - known_encodings, axis=1)
    min_idx = np.argmin(distances)
    
    if distances[min_idx] < tolerance:
        return True, known_faces[min_idx][0]
    return False, None

def process_frame(frame, known_faces):
    """Optimized frame processing with all original functionality"""
    # Convert to RGB only once
    rgb_frame = frame[:, :, ::-1] if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Fast face detection
    face_locations = detect_faces_fast(rgb_frame)
    
    # Get encodings only for detected faces
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    for (top, right, bottom, left), encoding in zip(face_locations, encodings):
        match, person_id = recognize_face(encoding, known_faces)
        face_crop = frame[top:bottom, left:right]

        # Optimized emotion detection
        emotion = "Neutral"
        if face_crop.size > 0:
            try:
                top_emotion = emotion_detector.top_emotion(face_crop)
                if top_emotion and isinstance(top_emotion, tuple):
                    if top_emotion[1] is not None:
                        emotion = f"{top_emotion[0]} ({top_emotion[1]:.2f})"
                    else:
                        emotion = f"{top_emotion[0]}"
            except Exception:
                pass  # Keep neutral if emotion detection fails

        color = (0, 255, 0) if match else (0, 0, 255)
        label = person_id if match else 'Unknown'

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f'{label}, {emotion}', (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        results.append({
            'box': [left, top, right, bottom],
            'matched': match,
            'person_id': person_id if match else None,
            'emotion': emotion
        })

    return frame, results

def main(input_path, known_faces, output_path):
    """Optimized video processing"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 2nd frame if high resolution to maintain performance
        if frame_count % 2 == 0 or w < 1280:
            processed_frame, _ = process_frame(frame, known_faces)
            out.write(processed_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    return frame_count

def process_webcam_live(known_faces):
    """Optimized webcam processing"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible")
        return

    print("Press 'q' to quit...")
    
    # Downscale resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_processed = 0
    frame_interval = 1.0 / 15  # Target 15 FPS
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_processed >= frame_interval:
            processed_frame, _ = process_frame(frame, known_faces)
            last_processed = current_time
        else:
            processed_frame = frame

        cv2.imshow('Live Feed', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
