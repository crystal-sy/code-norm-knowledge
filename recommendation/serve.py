# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:41:29 2022

@author: styra
"""

import os
import sys
import numpy as np
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from crossdomain import crossdomain
from flask import Flask, request

from bert_biLSTM_model import ModelConfig, predict, readfile


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

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
def code_predict(question):
    index_to_word = readfile('index_to_word.txt')
    model_config = ModelConfig()
    model_config.output_size = len(index_to_word)
    diversities = np.arange(0.05, 1.05, 0.05)
    predictions = [model_predict(question, model_config, index_to_word, diversity=d)
                   for d in diversities]
    # returning the latest sentence, + prediction
    suggestions = [question + x.rstrip("PARAM_END") for x in predictions]
    print(suggestions)
    return suggestions

def model_predict(question, model_config, index_to_word, diversity):
    outputs = ''
    for _ in range(20):
        preds = predict(question, model_config)[0]
        answer_token = sample(preds, diversity)
        new_text_token = str(index_to_word.get(answer_token, ''))
        print('answer_token:', answer_token)
        print('new_text_token:', new_text_token)
        outputs += new_text_token
        question += new_text_token
        if 'PARAM_END' == new_text_token:
            break
    print('Run end on ', diversity)
    return outputs
    

if __name__ == "__main__":
    """
    main()
    """
    test_question = 'import org.springframework.'
    code_predict_offline(test_question)
