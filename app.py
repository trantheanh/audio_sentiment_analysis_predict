import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
import numpy as np
import librosa

import pickle


def extract_feature(file_path):
    feature_all = np.empty((0, 193))
    X, sr = librosa.load(file_path, duration=3)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([feature_all, features])
    return features


def run():
    with open('model/mlp_model_tanh_adadelta.json', 'r') as json_file:
        model_json = json_file.read()
        model: keras.models.Model = model_from_json(model_json)
        model.load_weights("model/mlp_tanh_adadelta_model.h5")

    file_path = "input/test.wav"
    features = extract_feature(file_path)

    y_pred = model.predict(features)
    result = np.argmax(y_pred, axis=1)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    print("EMOTION IS: {}".format(emotions[result[0]]))


if __name__ == "__main__":
    run()



