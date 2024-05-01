import numpy as np
import pandas as pd
import pyaudio
import wave
import librosa
from scipy.io.wavfile import write
from keras.models import model_from_json
import pickle


json_file = open('models/40/final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('models/40/final_model.h5')

# Define constants for audio recording
format = pyaudio.paInt16
channels = 1
filename = "test/hello/arsi_angry.wav"
duration=5
sample_rate=44100
chunk_size=1024
duration=5


# Define function to record audio
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)
    print("Recording...")
    frames = []

    # Calculate total number of chunks
    total_chunks = int(sample_rate / chunk_size * duration)
    chunks_per_marker = total_chunks // 10  # Number of chunks for each '#' character

    for i in range(total_chunks):
        data = stream.read(chunk_size)
        frames.append(data)

    print("\nFinished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(audio.get_sample_size(format))
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


# Define constants for audio feature extraction
mfcc_sample_rate = 22050
n_mfcc = 40
axis_mfcc = 1


# Function to preprocess audio

def preprocess_audio(filename, sample_rate=22050, n_mfcc=40, offset_s=0.5):
    # Load audio file
    y, sr = librosa.load(filename, sr=sample_rate, mono=True, offset=offset_s)

    # Trim initial and final silent portions
    y_trimmed, _ = librosa.effects.trim(y)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)


    # Calculate mean of MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)

    return mfccs_mean

#record_audio()


processed_audio = preprocess_audio(filename)
audio_data = np.expand_dims(processed_audio, axis=0)

with open('meta/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

x_cnn = scaler.transform(audio_data)
x = np.expand_dims(x_cnn, axis=2)

# Predict emotion using the model
predictions = model.predict(x)

emotions = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprised'}

predicted_emotion = emotions[np.argmax(predictions)].title()


print("Predicted Emotion:", predicted_emotion)
