#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:30:05 2018

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
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from unicodedata import normalize
import tensorflow as tf

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)

###parameters
num_words = None #None is default
num_examples = 1
epoch_num = 1000
batch_size = 1

###test small sample
data = data[0:num_examples]

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(
        num_words=num_words, #None is default
        #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', 
        lower=True, 
        #split=' ', 
        char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

# encode and pad sequences
def encode_sequences(tokenizer, lines):
    tokenized = tokenizer.texts_to_sequences(lines)
    max_length = max(list(map(lambda x: len(x), tokenized)))
    X = pad_sequences(tokenized, maxlen=max_length, padding='post')
    return X, max_length


#clean text into utf-8
data_text_clean = list(map(lambda x: utf8_encode(x), data['mtgencoded_text']))
data_java_clean = list(map(lambda x: utf8_encode(x), data['java_code']))

# prepare tokenizers
text_tokenizer = create_tokenizer(data_text_clean)
java_tokenizer = create_tokenizer(data_java_clean)

if(num_words is None):
    text_vocab_size = len(text_tokenizer.word_index) + 1
    java_vocab_size = len(java_tokenizer.word_index) + 1
else: 
    text_vocab_size = num_words
    java_vocab_size = num_words
    
#encode sequences to tokenizers
X, x_max_len = encode_sequences(text_tokenizer, data_text_clean)
Y, y_max_len = encode_sequences(java_tokenizer, data_java_clean)
Y = np.expand_dims(Y,-1)

#print out stats of cleaned and tokenized inputs
text_length = x_max_len
java_length = y_max_len
print('Text Vocabulary Size: %d' % text_vocab_size)
print('Text Max Length: %d' % (text_length))
print('Java Vocabulary Size: %d' % java_vocab_size)
print('Java Max Length: %d' % (java_length))

#test/validation split
split_idx = round(len(data) * 0.90)
trainX = X[0:split_idx]
validX = X[split_idx:len(data)]
trainY = Y[0:split_idx]
validY = Y[split_idx:len(data)]


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# define model
model = define_model(text_vocab_size, 
                     java_vocab_size, 
                     text_length, 
                     java_length, 
                     256)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)

# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

model.fit(trainX, trainY, 
          #callbacks=[checkpoint],
          epochs=epoch_num,
          batch_size=batch_size, 
          validation_data=(validX, validY), 
          verbose=1)

#Evaluation
#model = load_model('model.h5')
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        print(i)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

'''
evaluate_model(model, java_tokenizer, trainX, list(zip(data_java_clean[0:split_idx],
                                                       data_text_clean[0:split_idx])))


evaluate_model(model, java_tokenizer, validX, list(zip(data_java_clean[split_idx:len(data)],
                                                       data_text_clean[split_idx:len(data)])))

'''



def compare_prediction(index):
    if(index < split_idx):
        ex = trainX[index].reshape((1, trainX[index].shape[0]))
    else:
        v_index = index - split_idx
        ex = validX[v_index].reshape((1, validX[v_index].shape[0]))
    print("INPUT:")
    print(data_text_clean[index])
    print("TARGET:")
    print(data_java_clean[index])
    print("OUTPUT:")
    print(predict_sequence(model, java_tokenizer, ex))

compare_prediction(0)
compare_prediction(1)
compare_prediction(2)
compare_prediction(3)
compare_prediction(4)

#todo: replace card name instances with @ in java
#todo: keep brackets and parens in the java text
'''
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''


