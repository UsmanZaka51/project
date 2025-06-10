import cv2
import numpy as np
import boto3
from decimal import Decimal
from fer import FER
import ast

# Load YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    model="yunet.onnx",  # Download from OpenCV GitHub if not present
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# Emotion detector
emotion_detector = FER(mtcnn=False)

# DynamoDB setup
dynamodb = boto3.resource("dynamodb", region_name='us-east-1')
table = dynamodb.Table('KnownFaces')

def load_known_faces():
    response = table.scan()
    known_faces = []
    for item in response.get('Items', []):
        person_id = item['person_id']
        embedding_data = item['embedding']
        embedding = np.array([float(x) for x in ast.literal_eval(embedding_data)] if isinstance(embedding_data, str) else embedding_data)
        known_faces.append((person_id, embedding))
    return known_faces

def recognize_face(encoding, known_faces, tolerance=0.5):
    for person_id, known_encoding in known_faces:
        distance = np.linalg.norm(encoding - known_encoding)
        if distance < tolerance:
            return True, person_id
    return False, None

def detect_faces_yunet(frame):
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    return faces if faces is not None else []

def process_frame(frame, known_faces):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces_yunet(frame)

    results = []

    for face in faces:
        x, y, w, h = [int(v) for v in face[:4]]
        top, right, bottom, left = y, x + w, y + h, x
        face_crop = rgb[top:bottom, left:right]

        try:
            encoding = cv2.face_LBPHFaceRecognizer_create().compute(frame[top:bottom, left:right])[0]
        except:
            encoding = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
            if not encoding:
                continue
            encoding = encoding[0]

        match, person_id = recognize_face(encoding, known_faces)
        top_emotion = emotion_detector.top_emotion(face_crop) if face_crop.size > 0 else None
        emotion = f"{top_emotion[0]} ({top_emotion[1]:.2f})" if top_emotion and isinstance(top_emotion, tuple) and top_emotion[1] else "Neutral"

        label = person_id if match else "Unknown"
        color = (0, 255, 0) if match else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}, {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        results.append({
            'box': [left, top, right, bottom],
            'matched': match,
            'person_id': person_id if match else None,
            'emotion': emotion
        })

    return frame, results

def main(input_path, known_faces, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        processed_frame, _ = process_frame(frame, known_faces)
        out.write(processed_frame)

    cap.release()
    out.release()
    return frame_count
