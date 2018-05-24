#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:51:42 2018

@author: josephhiggins
"""


import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from unicodedata import normalize

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)

def utf8_encode(line):
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    return line

data_text_clean = list(map(lambda x: utf8_encode(x), data['mtgencoded_text']))
data_java_clean = list(map(lambda x: utf8_encode(x), data['java_code']))


def create_tokenizer(lines):
    tokenizer = Tokenizer(
        num_words=None, 
        #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 
        lower=True, 
        #split=' ', 
        char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

text_tokenizer = create_tokenizer(data_text_clean)
text_tokenizer.word_index
java_tokenizer = create_tokenizer(data_java_clean)
java_tokenizer.word_index

#test/train split
split_idx = round(len(data) * 0.80)
text_train = data_text_clean[0:split_idx]
text_test =  data_text_clean[split_idx:len(data)]
java_train = data_java_clean[0:split_idx]
java_test =  data_java_clean[split_idx:len(data)]

def encode_sequences(tokenizer, lines):
    X = tokenizer.texts_to_sequences(lines)
    max_length = max(list(map(lambda x: len(x), trainX)))
    return pad_sequences(X, maxlen=max_length, padding='post')


trainX = encode_sequences(text_tokenizer, text_train)
data_text_clean[0]
data_text_clean[1]
text_train[0]
text_train[1]

text_tokenizer.word_index
for i in range(0,30):
    print(list(text_tokenizer.word_index.keys())[i])




























