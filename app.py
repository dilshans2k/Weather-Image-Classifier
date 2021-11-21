# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:43:20 2021

@author: Dilshan Sandhu
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='weather_mobilenetv2.h5'

# Load your trained model
model = load_model(MODEL_PATH)


class_names = {0:'Weather is Dew',
 1:'Weather is Fog/Smog',
 2:'Weather is Frost',
 3:'Weather is Glaze',
 4:'Weather is Hail',
 5:'Weather is Lightning',
 6:'Weather is Rain',
 7:'Weather is Rainbow',
 8:'Weather is Rime',
 9:'Weather is Sandstorm',
 10:'Weather is Snow'}


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print("above")
    pred_probs = model.predict(x,batch_size=32)
    print("yo",pred_probs)
    pred_class = int(pred_probs.argmax(axis=1))
    print(pred_class)
    pred_label = class_names[pred_class]

    return pred_label


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
