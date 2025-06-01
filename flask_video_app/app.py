
from flask import Flask, request, render_template, send_from_directory
import os
from your_script import main

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file part", 400
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400

        input_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
        output_path = os.path.join(PROCESSED_FOLDER, 'output_with_emotions.mp4')
        file.save(input_path)

        # Run processing logic locally
        main(input_path, output_path)

        return render_template('index.html', download_ready=True)

    return render_template('index.html', download_ready=False)

@app.route('/download')
def download_file():
    return send_from_directory(PROCESSED_FOLDER, 'output_with_emotions.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
