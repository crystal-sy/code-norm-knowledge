# -*- coding: utf-8 -*-
"""
Created on Sun Sep 4 18:41:29 2022

@author: styra
"""

import os
from collections import Counter

import tokenize as tk
from io import BytesIO

data_dir = 'data/'
end_flag = 'PARAM_END'

# 文件加载
def loadfile(dirPath, javaFileList):
    for file in os.listdir(dirPath):
        filePath = dirPath + file
        if os.path.isfile(filePath):
            if file.endswith('.java') :
                javaFileList.append(filePath)
        else:
            loadfile(filePath + '/', javaFileList)
    
    
# 文件读取
def readfile(filePath):
    #文件输入
    content = []
    import_content = []
    with open(filePath, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip()
            # 去除无意义的代码段
            if line.startswith('package ') or line == '\n' or line == '' or line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('*/'):
                continue
            elif line.startswith('import '):
                import_content.append(line + end_flag)
            else:
                content.append(line + end_flag)
        f.close()
    return content, import_content


def writeFile(questions, answers):
    file = data_dir + 'data_java_qa.csv'
    fp = open(file, 'w+', encoding='UTF-8')
    fp.write('Q,A')
    fp.write('\n')
    seqlen = len(questions)
    for i in range(0, seqlen, 1):
        content = questions[i].replace(',', '@COMMA_CHAR_REPLACE') + ',' + answers[i].replace(',', '@COMMA_CHAR_REPLACE')
        fp.write(content)
        fp.write('\n')
    fp.close()


def text_tokenize(txt):
    """ specific tokenizer suitable for extracting 'python tokens' """
    toks = []
    try:
        for x in tk.tokenize(BytesIO(txt.encode('utf-8')).readline):
            toks.append(x)
    except tk.TokenError:
        pass
    tokkies = []
    old = (0, 0)
    for t in toks:
        if not t.string:
            continue
        if t.start[0] == old[0] and t.start[1] > old[1]:
            tokkies.append(" " * (t.start[1] - old[1]))
        tokkies.append(t.string)
        old = t.end
    if txt.endswith(" "):
        tokkies.append(" ")
    toks = [x for x in tokkies if not x.startswith("#")]
    return toks[1:]


class EncoderDecoder():
    def __init__(self, maxlen, min_count, unknown, padding, tokenize, untokenize):
        self.maxlen = maxlen
        self.min_count = min_count
        self.unknown = unknown
        self.padding = padding
        self.tokenize = tokenize
        self.untokenize = untokenize
        self.questions = []
        self.answers = []
        self.ex, self.dx = None, None
        self.ey, self.dy = None, None
        self.build_data()

    def build_data(self):
        raise NotImplementedError

    def encode_x(self, x):
        return self.ex.get(x, 0)

    def encode_y(self, y):
        return self.ey.get(y, 0)

    def decode_x(self, x):
        return self.dx.get(x, self.unknown)

    def decode_y(self, y):
        return self.dy.get(y, self.unknown)

    def build_coders(self, tokens):
        tokens = [item for sublist in tokens for item in sublist]
        word_to_index = {k: v for k, v in Counter(tokens).items() if v >= self.min_count}
        word_to_index = {k: i for i, (k, v) in enumerate(word_to_index.items(), 1)}
        word_to_index[self.unknown] = 0
        index_to_word = {v: k for k, v in word_to_index.items()}
        index_to_word[0] = self.unknown
        return word_to_index, index_to_word

    def build_a_coders(self):
        self.ey, self.dy = self.build_coders([self.answers])
        print("unique answer tokens:", len(self.ey))

class TextEncoderDecoder(EncoderDecoder):
    def __init__(self, texts, tokenize=str.split, untokenize=" ".join,
                 window_step=1, maxlen=20, min_count=1,
                 unknown="UNKNOWN", padding="PADDING"):
        self.texts = texts
        self.window_step = window_step
        c = super(TextEncoderDecoder, self)
        c.__init__(maxlen, min_count, unknown, padding, tokenize, untokenize)

    def build_data(self):
        self.questions = []
        self.answers = []
        for contents in self.texts:
            for content in contents[0]:
                text = self.tokenize(content)
                seqlen = len(text)
                for i in range(4, seqlen, self.window_step):
                    self.questions.append(self.untokenize(text[0 : i]))
                    self.answers.append(text[i])
            
            for content in contents[1]:
                text = self.tokenize(content)
                seqlen = len(text)
                for i in range(2, seqlen, self.window_step):
                    self.questions.append(self.untokenize(text[0 : i]))
                    self.answers.append(text[i])
        self.build_a_coders()
        print("number of QA pairs:", len(self.questions))
        writeFile(self.questions, self.answers)
    
    
if __name__ == "__main__":
    javaFileList = []
    loadfile(data_dir, javaFileList)
    print(javaFileList)
    text = []
    for file in javaFileList:
        contents = []
        content, import_content = readfile(file)
        contents.append(import_content)
        contents.append(content)
        text.append(contents)
    
    TextEncoderDecoder(text, tokenize=text_tokenize, untokenize="".join, padding=" ",
                       min_count=1, maxlen=30)