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


def utf8_encode(line):
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    return line

data_text_clean = list(map(lambda x: utf8_encode(x), data['mtgencoded_text']))
data_java_clean = list(map(lambda x: utf8_encode(x), data['java_code']))

# prepare text tokenizer
text_tokenizer = create_tokenizer(data_text_clean, True)
text_vocab_size = len(text_tokenizer.word_index) + 1
text_length = max_length(data['mtgencoded_text'])
print('Text Vocabulary Size: %d' % text_vocab_size)
print('Text Max Length: %d' % (text_length))
#text_tokenizer.word_index 

# prepare java tokenizer
java_tokenizer = create_tokenizer(data_java_clean, True)
java_vocab_size = len(java_tokenizer.word_index) + 1
java_length = max_length(data['java_code'])
print('Java Vocabulary Size: %d' % java_vocab_size)
print('Java Max Length: %d' % (java_length))
#java_tokenizer.word_index

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
split_idx = round(len(data) * 0.80)
text_train = data_text_clean[0:split_idx]
text_test =  data_text_clean[split_idx:len(data)]
java_train = data_java_clean[0:split_idx]
java_test =  data_java_clean[split_idx:len(data)]

# prepare training data
trainX = encode_sequences(text_tokenizer, text_length, text_train)
trainY = encode_sequences(java_tokenizer, java_length, java_train)
trainY = encode_output(trainY, java_vocab_size)
# prepare validation data
testX = encode_sequences(text_tokenizer, text_length, text_test)
testY = encode_sequences(java_tokenizer, java_length, java_test)
testY = encode_output(testY, java_vocab_size)

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

model.compile(optimizer='adam', loss='categorical_crossentropy')
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
          epochs=2,
          batch_size=128, 
          validation_data=(testX, testY), 
          callbacks=[checkpoint], 
          verbose=1)


#Evaluation
model = load_model('model.h5')

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



evaluate_model(model, java_tokenizer, trainX, list(zip(java_train, text_train)))
evaluate_model(model, java_tokenizer, testX, list(zip(java_test, text_test)))















