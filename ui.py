import joblib
from sklearn.preprocessing import StandardScaler
import librosa.display
import librosa
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
app = Flask(__name__, template_folder='template')
classes = ['cello', 'flute', 'oboe', 'sax', 'trumpet', 'viola']
fs = 44100
n_mels = 128       # Number of Mel bands
n_mfcc = 13        # Number of MFCCs
# print("Prediction:  "+classes
#       [int(svclassifier.predict([test_set[140]]))])
# print("Actual:  "+classes[int(test_classes[140])])
scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
# scalerfile = "mfcc_feature_vectors.pl"
# scaler = pickle.load(open(scalerfile, 'rb'))
# test_scaled_set = scaler.transform()


def get_features(y, sr=fs):

    S = librosa.feature.melspectrogram(y, sr=fs, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    feature_vector = np.mean(mfcc, 1)
    # feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
    return feature_vector


feature_vectors = []


def loadmx(f):
    try:
        y, sr = librosa.load(f, sr=fs)
        y /= y.max()  # Normalize
        if len(y) < 2:
            print("Error loading %s" % f)
            # continue
        feat = get_features(y, sr)
        feature_vectors.append(feat)
        print(feat)
        print(np.array([feature_vectors[0]]))
    #     scaler = StandardScaler()
    # # print(feature_vectors)
    #     scaled_feature_vectors = scaler.fit_transform(
    #         np.array([feature_vectors[0]]))
        test_scaled_set = scaler.transform(np.array([feature_vectors[0]]))

        print(test_scaled_set)
        return svclassifier.predict(test_scaled_set)

    # return scaled_feature_vectors

    # feature_vectors.append(feat)
    # sound_paths.append(f)

    except Exception as e:
        print("Error loading %s. Error: %s" % (f, e))
        return 'false'


def load_model():
    global svclassifier
    svclassifier = joblib.load('trainedSVM.joblib')
    feature_vectors = []
    print(classes
          [int(loadmx(
              './audio/london_phill_dataset_multi/trumpet/trumpet_A3_05_pianissimo_normal.mp3'))])

    # print(classes
    #       [int(svclassifier.predict([scaled_feature_vectors[0]]))])
    # svclassifier.predict()


# @app.route('/', methods=['GET', 'POST'])
# @app.route('/')
# def index():
#     return render_template('index.html')


if __name__ == '__main__':
    load_model()
    # app.run(debug=True)
