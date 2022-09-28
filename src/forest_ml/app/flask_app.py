from crypt import methods
from flask import Flask, request, jsonify
from joblib import load
from forest_ml.data import get_data
from forest_ml.predict import _predict
import numpy as np
import pandas as pd

model_path = 'data/model.joblib'
model = load(model_path)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    forest = request.get_json()
    if 'Id' in forest.keys():
        X = pd.DataFrame(forest).drop('Id')   
    else:   
        X = pd.DataFrame(forest)
    y_pred = _predict(X, model)

    result = {'forest_type' : int(y_pred)}
