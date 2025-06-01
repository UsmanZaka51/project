import os
import ast
import cv2
import boto3
import numpy as np
import face_recognition
from fer import FER
from flask import Flask, render_template, request, send_file, jsonify

# Optional: Speed optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

app = Flask(__name__)

# ===== Load Known Faces from DynamoDB =====
def load_known_faces_from_dynamodb(table_name="KnownFaces"):
    print("Loading known faces from DynamoDB...")
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    try:
        response = table.scan(ProjectionExpression="person_id,embedding")
        items = response.get('Items', [])

        known_names = []
        known_encodings = []
        for item in items:
            name = item['person_id']
            embedding = np.array(ast.literal_eval(item['embedding']), dtype=np.float32)
            known_names.append(name)
            known_encodings.append(embedding)

        print(f"Loaded {len(known_names)} known faces from DynamoDB.")
        return known_names, known_encodings
    except Exception as e:
        print(f"Failed to load from DynamoDB: {e}")
        return [], []

# Convert float list to decimal list for DynamoDB storage
from decimal import Decimal
def float_list_to_decimal_list(float_list):
    return [Decimal(str(f)) for f in float_list]

# DynamoDB client for adding face
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table("KnownFaces")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video file part", 400
        video_file = request.files['video']
        if video_file.filename == '':
            return "No selected video file", 400

        local_input_path = "/tmp/input_video.mp4"
        local_output_path = "/tmp/output_with_emotions.mp4"
        video_file.save(local_input_path)

        known_names, known_encodings = load_known_faces_from_dynamodb()
        if not known_encodings:
            return "No known faces found in DynamoDB.", 500

        emotion_detector = FER(mtcnn=True)
        cap = cv2.VideoCapture(local_input_path)
        if not cap.isOpened():
            return "Failed to open uploaded video.", 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                    name = known_names[best_match_index]

                # Emotion detection 
                face_crop = frame[top:bottom, left:right]
                top_emotion, score = emotion_detector.top_emotion(face_crop)
                emotion_label = f"{top_emotion} ({score:.2f})" if top_emotion else "Neutral"

                # Draw face box + name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Draw emotion label
                cv2.putText(frame, emotion_label, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        return send_file(local_output_path, as_attachment=True, download_name="output_with_emotions.mp4")

    return render_template('index.html')


@app.route('/admin/add-face', methods=['POST'])
def add_face():
    person_id = request.form['person_id']
    image_file = request.files['face_image']
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return jsonify({'error': 'No face found'}), 400
    embedding = encodings[0].tolist()
    embedding_decimal = float_list_to_decimal_list(embedding)  # convert to Decimal
    table.put_item(Item={'person_id': person_id, 'embedding': embedding_decimal})
    return jsonify({'message': f'Face for {person_id} added successfully'}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
