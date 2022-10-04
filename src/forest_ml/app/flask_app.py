from flask import Flask, request, jsonify, Response
import click
from joblib import load
import pandas as pd


FOREST_TYPES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]
MODEL_PATH = "data/model.joblib"
model = load(MODEL_PATH)

app = Flask('forest_type')


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    forest = request.get_json()
    assert forest is not None
    if "Id" in forest.keys():
        X = pd.DataFrame(forest).drop("Id")
    else:
        X = pd.DataFrame(forest)
    y_pred = FOREST_TYPES[model.predict(X)[0] - 1]
    prob = model.predict_proba(X).max()
    result = {"forest_type": str(y_pred), "probabilities": float(prob)}
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0", port=9696)
