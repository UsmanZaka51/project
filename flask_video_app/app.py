from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
from decimal import Decimal
import os
import face_recognition
import boto3
import numpy as np
import uuid
import cv2
import time
import threading
from queue import Queue

app = Flask(__name__)

# =============================================
# CONFIGURATION
# =============================================
app.config['INPUT_DIR'] = "input"
app.config['OUTPUT_DIR'] = "output"
app.config['MAX_FRAME_RATE'] = 15
os.makedirs(app.config['INPUT_DIR'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

# =============================================
# YuNet Face Detector Initialization
# =============================================
face_detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# Set backend/target for faster inference
face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# =============================================
# Background Processing Thread
# =============================================
processing_queue = Queue(maxsize=10)
stop_processing = False

def background_processor():
    known_faces = load_known_faces()
    while not stop_processing:
        task = processing_queue.get()
        if task is None:
            break
        frame, callback = task
        processed_frame, results = process_frame_optimized(frame, known_faces)
        callback(processed_frame, results)

processor_thread = threading.Thread(target=background_processor)
processor_thread.daemon = True
processor_thread.start()

# =============================================
# Utility Functions
# =============================================
def detect_faces_yunet(frame):
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    results = face_detector.detect(frame)
    face_locations = []

    if results[1] is not None:
        for det in results[1]:
            x, y, w, h = map(int, det[:4])
            face_locations.append((y, x + w, y + h, x))  # (top, right, bottom, left)
    return face_locations

def load_known_faces():
    table = boto3.resource('dynamodb').Table('knownFaces')
    response = table.scan()
    known_faces = {}
    for item in response.get('Items', []):
        encoding = np.array([float(val) for val in item['embedding']])
        known_faces[item['person_id']] = encoding
    return known_faces

def process_frame_optimized(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = detect_faces_yunet(frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        name = "Unknown"
        if True in matches:
            matched_idx = matches.index(True)
            name = list(known_faces.keys())[matched_idx]
        results.append({"name": name, "box": [int(left), int(top), int(right), int(bottom)]})
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, results

# =============================================
# Flask Routes
# =============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin/add-face', methods=['POST'])
def add_face():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        person_id = request.form.get('person_id', '').strip()
        if not person_id:
            return jsonify({'success': False, 'error': 'person_id is required'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            return jsonify({'success': False, 'error': 'No face found in the image'}), 400

        embedding = [Decimal(str(x)) for x in encodings[0]]
        table = boto3.resource('dynamodb').Table('knownFaces')
        table.put_item(Item={'person_id': person_id, 'embedding': embedding})

        return jsonify({'success': True, 'message': f'Face data added for {person_id}', 'person_id': person_id}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400

    video = request.files['video']
    if not allowed_file(video.filename, {'mp4', 'mov', 'avi'}):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    filename = secure_filename(video.filename)
    input_path = os.path.join(app.config['INPUT_DIR'], filename)
    output_path = os.path.join(app.config['OUTPUT_DIR'], f"processed_{filename}")

    video.save(input_path)
    known_faces = load_known_faces()
    total_frames = main(input_path, known_faces, output_path)

    return jsonify({
        'success': True,
        'output_video': f"/download?path={output_path}",
        'frames_processed': total_frames
    })

@app.route('/process-webcam', methods=['POST'])
def process_webcam():
    try:
        frame_data = request.files['frame'].read()
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        result = {}
        event = threading.Event()

        def callback(processed_frame, results):
            nonlocal result
            _, buffer = cv2.imencode('.jpg', processed_frame)
            result = {
                'frame': buffer.tobytes(),
                'results': results
            }
            event.set()

        processing_queue.put((frame, callback))
        event.wait()

        return jsonify({
            'success': True,
            'results': result['results'],
            'frame': result['frame'].hex(),
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.teardown_appcontext
def shutdown(exception=None):
    global stop_processing
    stop_processing = True
    processing_queue.put(None)
    processor_thread.join()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
