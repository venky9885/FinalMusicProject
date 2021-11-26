import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
import joblib
from numpy.core.numeric import outer
from sklearn.preprocessing import StandardScaler
import librosa.display
import librosa
import time
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
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
global svclassifier
svclassifier = joblib.load('trainedSVM.joblib')
# app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            filename = secure_filename(file.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))

            # ap.config['UPLOAD_FOLDER']
            # print(file.path)
            pth = os.path.join(
                basedir, app.config['UPLOAD_FOLDER'], filename)
            file.save(pth)
            print(pth)
            output = classes[int(loadmx(pth))]
            print(output)
            return render_template('index.html', output=output)
        except Exception as e:
            print(e)
            # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))

            # ap.config['UPLOAD_FOLDER']
            # print(file.path)
            pth = os.path.join(
                basedir, app.config['UPLOAD_FOLDER'], filename)
            file.save(pth)
            print(pth)
            output = classes[int(loadmx(pth))]
            print(output)
            redirect(url_for('upload_file', name=output))
            time.sleep(4)
            # return render_template('index.html', output=output)
            return redirect(url_for('upload_file', name=output))
    return '''
 
    <!doctype html>
    <title>Upload new File</title>
   
    
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
      
    </form>
    '''

# *******
#  <style>
#     div {
#          background-image: url('https://picsum.photos/200');
#         }
#     </style>


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


# def load_model():
#     global svclassifier
#     svclassifier = joblib.load('trainedSVM.joblib')
#     feature_vectors = []
#     output = classes[int(loadmx(
#         './audio/london_phill_dataset_multi/trumpet/trumpet_A3_05_pianissimo_normal.mp3'))]
#     print(output)
#     return render_template('index.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)


# <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
