import base64

from flask import Flask, render_template, Response, request, send_file
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

from keras.models import model_from_json

from tensorflow.keras.preprocessing.image import img_to_array

import threading
from flask import jsonify
from speech_tester import record_audio, predict_emotion
from text_tester import pred_text, preprocess, get_top


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

@app.route('/')
def index():
    return render_template('index.html')

# Load the pre-trained model and haarcascade classifier
model = model_from_json(open("Video/models/model.json", "r").read())
model.load_weights('Video/models/model.h5')

face_cascade = cv2.CascadeClassifier('Video/models/haarcascade_frontalface_default.xml')

# Global variables for video capture and thread
cap = None
thread = None
stop_thread = False

detected_emotions = {}

# Function to start the video feed
def start_video_feed():
    global cap, stop_thread
    stop_thread = False
    global detected_emotions
    detected_emotions = {}
    cap = cv2.VideoCapture(0)

    while True:
        if stop_thread:
            break

        ret, frame = cap.read()
        frame = cv2.resize(frame,(1080, 720))

        if not ret:
            break

        height, width, channel = frame.shape

        sub_img = frame[0:int(height / 6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        label_color = (10, 10, 200)
        label = "Emotion Detection System"
        label_dimension = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        textX = int((res.shape[1] - label_dimension[0]) / 2)
        textY = int((res.shape[0] + label_dimension[1]) / 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_faces = len(faces)
        if num_faces:
            num_faces = num_faces
        else:
            num_faces = 0

        cv2.putText(frame, f"Faces Detected: {num_faces}", (5, textY + 30), FONT, 0.5, (0, 255, 0), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.array(roi_gray, 'float32')
            predictions = model.predict(roi_gray)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]

            confidence = np.round(np.max(predictions[0]) * 100, 2)

            if emotion_prediction in detected_emotions:
                count, accum_confidence = detected_emotions[emotion_prediction]
                detected_emotions[emotion_prediction] = (count + 1, accum_confidence + confidence)
            else:
                detected_emotions[emotion_prediction] = (1, confidence)

            label_pred = "Sentiment: {}".format(emotion_prediction)
            cv2.putText(frame, label_pred, (int(x), int(y + h + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            label_violation = 'Confidence: {}'.format(str(confidence) + "%")
            violation_text_dimension = cv2.getTextSize(label_violation, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            violation_y_axis = int((y + h + 20) + violation_text_dimension[1])
            cv2.putText(frame, label_violation, (int(x), violation_y_axis + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


# Route to serve the video stream with emotion detection
@app.route('/video')
def video():
    return render_template('video.html')

# Route to start the video feed
@app.route('/start_feed')
def start_feed():
    global thread
    if not thread or not thread.is_alive():
        thread = threading.Thread(target=start_video_feed)
        thread.start()
    return 'Video feed started.'

# Route to stop the video feed
@app.route('/stop_feed')
def stop_feed():
    global stop_thread
    stop_thread = True

    return 'Video feed stopped.'


# Route to get the results
@app.route('/get_results')
def get_results():
    if detected_emotions:
        most_confident_emotion = max(detected_emotions, key=lambda x: detected_emotions[x][1] / detected_emotions[x][0])
        count, accum_confidence = detected_emotions[most_confident_emotion]
        accum_confidence = str(round(accum_confidence/count,2))+"%"

        return jsonify({
            'most_confident_emotion': most_confident_emotion.title(),
            'accumulated_confidence': accum_confidence
        })
    else:
        return jsonify({
            'error': 'Heartless'
        })


# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(start_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/start-recording', methods=['POST'])
def start_recording():
    filename = record_audio()
    return jsonify({'filename': filename})

@app.route('/upload-recording', methods=['POST'])
def upload_recording():
    audio = request.files['audio']
    filename = secure_filename(audio.filename)
    audio.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filename = 'static/audio/' + filename
    return jsonify({'filename': filename})


@app.route('/submit-prediction', methods=['POST'])
def submit_prediction():

    if request.json is not None:
        filename = request.json['filename']
        predicted_emotion = predict_emotion(filename)
        return jsonify({'predicted_emotion': predicted_emotion})
    else:
        return jsonify({'error': 'No JSON data received'})

@app.route('/text')
def text() :
    return render_template('text.html')

@app.route('/predict_text', methods=['POST','GET'])
def predict_text():
    text = request.form.get('text')
    processed_text = preprocess(text)
    top, num_words = get_top(processed_text)
    emotion = pred_text(processed_text)

    return render_template('text-res.html', traits=emotion, num_words=num_words, common_words=top)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
