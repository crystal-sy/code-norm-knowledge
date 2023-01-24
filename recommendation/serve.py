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
from flask import Flask, request

from bert_biLSTM_model import ModelConfig, predict, get_answer, readfile


def read_models(base_path="models/"):
    return set([x.split(".")[0] for x in os.listdir(base_path)])


app = Flask(__name__)


def get_args(req):
    if request.method == 'POST':
        args = request.json
    elif request.method == "GET":
        args = request.args
    return args


@app.route("/code_predict", methods=["GET", "POST", "OPTIONS"])
@crossdomain(origin='*', headers="Content-Type")
def code_predict_online():
    args = get_args(request)
    question = args.get("question", "")
    return code_predict(question)


def main(host="127.0.0.1", port=6001):
    app.run(host=host, port=port, debug=True)


def code_predict_offline(question):
    return code_predict(question)
    
def code_predict(question):
    index_to_word = readfile('index_to_word.txt')
    model_config = ModelConfig()
    model_config.output_size = len(index_to_word)
    results = predict(question, model_config)
    print(results)
    answers = get_answer(index_to_word, results)
    print(answers)
    return answers
    

if __name__ == "__main__":
    """
    main()
    """
    test_question = ['import org.springframework.']
    code_predict_offline(test_question)
