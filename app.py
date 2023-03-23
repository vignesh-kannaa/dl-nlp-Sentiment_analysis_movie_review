
from flask import Flask, jsonify, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text

import numpy as np

app = Flask(__name__)
load_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
model=load_model('model.h5',custom_objects={'KerasLayer':hub.KerasLayer},options=load_option)
#if error exists on loading the model, delete the temp folder and try again

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inputText=request.values.get("input", None)
    if inputText:
        predicted=model.predict([inputText])
        predVal=predicted[0][0]
        # print(predVal)
        result = 'good' if predVal > 0.6 else 'bad' if predVal < 0.4 else 'neutral'
        return result
    else:
        return ''
    
    
# python -m flask run