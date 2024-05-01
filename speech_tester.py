import numpy as np
import pyaudio
import wave
import librosa
import pickle
from keras.models import model_from_json
import time


json_file = open('Audio/models/40/final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('Audio/models/40/final_model.h5')

with open('Audio/meta/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Define constants for audio feature extraction
mfcc_sample_rate = 22050
n_mfcc = 40
axis_mfcc = 1


# Define constants for audio recording
format = pyaudio.paInt16
channels = 1
duration=5
sample_rate=44100
chunk_size=1024
duration=5


# Define function to record audio
def record_audio():
    filename = 'static/audio/recording_{}.wav'.format(time.time())
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)
    frames = []

    # Calculate total number of chunks
    total_chunks = int(sample_rate / chunk_size * duration)

    for i in range(total_chunks):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(audio.get_sample_size(format))
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    return filename

# Function to preprocess audio

def predict_emotion(filename, sample_rate=22050, n_mfcc=40, offset_s=0.5):
    # Load audio file
    y, sr = librosa.load(filename, sr=sample_rate, mono=True, offset=offset_s)

    # Trim initial and final silent portions
    y_trimmed, _ = librosa.effects.trim(y)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)


    # Calculate mean of MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)

    audio_data = np.expand_dims(mfccs_mean, axis=0)

    x_cnn = scaler.transform(audio_data)
    x = np.expand_dims(x_cnn, axis=2)

    # Predict emotion using the model
    predictions = model.predict(x)

    emotions = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprised'}

    predicted_emotion = emotions[np.argmax(predictions)].title()

    return predicted_emotion
