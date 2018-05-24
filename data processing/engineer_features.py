#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:01:08 2018

@author: josephhiggins
"""
'''
sys.path.insert(0, '/Users/josephhiggins/Documents/CS 224U/sippycup/')
from parsing import Grammar, Rule
'''


#https://github.com/billzorn/mtgencode
#https://github.com/billzorn/mtgencode#training-a-neural-net
#http://minimaxir.com/2017/04/char-embeddings/, Max Woolf (@minimaxir)
#https://github.com/minimaxir/char-embeddings/blob/master/create_magic_text.py
#https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)


# fit a tokenizer
def create_tokenizer(lines, char_level):
    tokenizer = Tokenizer(
        num_words=None, 
        #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 
        lower=True, 
        #split=' ', 
        char_level=char_level)
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# prepare text tokenizer
text_tokenizer = create_tokenizer(data['mtgencoded_text'], True)
text_vocab_size = len(text_tokenizer.word_index) + 1
text_length = max_length(data['mtgencoded_text'])
print('Text Vocabulary Size: %d' % text_vocab_size)
print('Text Max Length: %d' % (text_length))
 
# prepare java tokenizer
java_tokenizer = create_tokenizer(data['java_code'], True)
java_vocab_size = len(java_tokenizer.word_index) + 1
java_length = max_length(data['java_code'])
print('Java Vocabulary Size: %d' % java_vocab_size)
print('Java Max Length: %d' % (java_length))


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

#test/train split
split_idx = round(len(data) * 0.90)
text_train = data['mtgencoded_text'][0:split_idx]
text_test =  data['mtgencoded_text'][split_idx:len(data)]
java_train = data['java_code'][0:split_idx]
java_test =  data['java_code'][split_idx:len(data)]

# prepare training data
trainX = encode_sequences(text_tokenizer, text_length, text_train)
trainY = encode_sequences(java_tokenizer, java_length, java_train)
trainY = encode_output(trainY, java_vocab_size)
# prepare validation data
testX = encode_sequences(text_tokenizer, text_length, text_test)
testY = encode_sequences(java_tokenizer, java_length, java_test)
testY = encode_output(trainY, java_vocab_size)




























