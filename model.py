
import numpy as np
import pandas as pd
from transformers import pipeline
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model = pipeline('text-classification', model='Ammar-alhaj-ali/arabic-MARBERT-sentiment')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict(text):
    prediction = model(text)
    print(prediction[0]["label"])

    return render_template('index.html', prediction_text='sentiment is {}'.format(prediction[0]["label"]))


if __name__ == "__main__":
    app.run(debug=True)
