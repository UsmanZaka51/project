
import cv2
import numpy as np
import face_recognition
from fer import FER

def main(input_video_path, output_video_path):
    print(f"Processing video: {input_video_path}")
    emotion_detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            face_crop = frame[top:bottom, left:right]
            top_emotion, score = emotion_detector.top_emotion(face_crop)
            emotion_label = f"{top_emotion} ({score:.2f})" if top_emotion else "Neutral"

            color = (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, emotion_label, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        out.write(frame)
        print(f"Processed frame {frame_count}", end='\r')
        frame_count += 1

    cap.release()
    out.release()
    print(f"\n✅ Video saved to: {output_video_path}")
