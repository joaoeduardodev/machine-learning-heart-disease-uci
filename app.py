import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model_heart.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    result = 'Não há presença de doença cardíaca no paciente'

    if(pred[0]):
        result = 'Presença de doença cardíaca no paciente'

    return render_template("index.html", prediction_text=result )


@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    print(pred[0])
    return jsonify(pred[0])