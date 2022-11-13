# -*- coding: utf-8 -*-
"""
Created on Sun Sep 4 18:41:29 2022

@author: styra
"""

import just

from encoder_decoder import TextEncoderDecoder, text_tokenize
from model import biLSTMBase

TRAINING_TEST_CASES = ["from tensorflow.keras.layers import"]


def get_data():
    return [values[1] for values in just.multi_read("data/**/*.py")]


def train(ted, model_name):
    lb = biLSTMBase(model_name, ted)
    try:
        lb.train(test_cases=TRAINING_TEST_CASES)
    except KeyboardInterrupt:
        pass
    print("saving")
    lb.save()


def train_token(model_name):
    data = get_data()
    # text tokenize splits source code into python tokens
    ted = TextEncoderDecoder(data, tokenize=text_tokenize, untokenize="".join, padding=" ",
                             min_count=1, maxlen=20)
    train(ted, model_name)


def get_model(model_name):
    return biLSTMBase(model_name)


def code_recommendation(model, text, diversities):
    predictions = [model.predict(text, diversity=d, max_prediction_steps=80,
                                 break_at_token="\n")
                   for d in diversities]
    # returning the latest sentence, + prediction
    suggestions = [text.split("\n")[-1] + x.rstrip("\n") for x in predictions]
    return suggestions


if __name__ == "__main__":
    train_token('neural_token')
