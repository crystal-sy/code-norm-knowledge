# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:41:29 2022

@author: styra
"""

import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from crossdomain import crossdomain
from flask import Flask, jsonify, request

from lstm import code_recommendation, get_model


def read_models(base_path="models/"):
    return set([x.split(".")[0] for x in os.listdir(base_path)])


app = Flask(__name__)

models = {x: get_model(x) for x in read_models()}


def get_args(req):
    if request.method == 'POST':
        args = request.json
    elif request.method == "GET":
        args = request.args
    return args


@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
@crossdomain(origin='*', headers="Content-Type")
def predict():
    args = get_args(request)
    sentence = args.get("keyword", "from ")
    model_name = args.get("model", "char")
    if model_name not in models:
        models[model_name] = get_model(model_name)
    suggestions = code_recommendation(models[model_name], sentence, [0.2, 0.5, 1])
    return jsonify({"data": {"results": [x.strip() for x in suggestions]}})


@app.route("/get_models", methods=["GET", "POST", "OPTIONS"])
@crossdomain(origin='*', headers="Content-Type")
def get_models():
    return jsonify({"data": {"results": list(models)}})


def main(host="127.0.0.1", port=6001):
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
    """
    sentence = 'from keras.layers '
    model_name = 'neural_token'
    if model_name not in models:
        models[model_name] = get_model(model_name)
    suggestions = neural_complete(models[model_name], sentence, [0.2, 0.5, 1])
    print([x.strip() for x in suggestions])
    """