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
#https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/

#JCH ideas:
##dont predict parens and brackets, since those can be filled in with syntax
##character level tokenization
##shuffle data
##sippycup


import sys, os
sys.path.append('/Users/josephhiggins/Documents/mtg/mtgtext2code/data processing/')

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
from attention_decoder import AttentionDecoder
import pickle

def open_pickle(file_name):
    file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
    return pickle.load( open(file_path+file_name, "rb" ) )

sequences_java_full = open_pickle('sequenced_java.pkl')
sequences_text_full = open_pickle('sequenced_text.pkl')
token_key_java_original = open_pickle('java_token_key.pkl')
token_key_text_original = open_pickle('text_token_key.pkl')

###parameters
num_examples = len(sequences_text_full) #1
num_words_text = 350
num_words_java = 350
epoch_num = 1
batch_size = 256

###take a subset of the data based on num_examples
sequences_text = sequences_text_full[0:num_examples]
sequences_java = sequences_java_full[0:num_examples]

#update tokens based on the data sample we use
##get unique tokens in data subset
def remove_keys_not_used(sequences, token_key):
    flat = [item for sublist in sequences for item in sublist]
    tokens_to_remove = set(token_key.keys()).difference(set(flat))
    for token in tokens_to_remove:
        token_key.pop(token, None)
    return token_key

token_key_text = remove_keys_not_used(sequences_text, token_key_text_original.copy())
token_key_java = remove_keys_not_used(sequences_java, token_key_java_original.copy())

##reindex keys
def re_sequence(token_list, mapping):
    return [mapping[token] for token in token_list]

new_idxs_text = dict((zip(token_key_text.keys(), range(1,len(token_key_text)+1))))
new_idxs_java = dict((zip(token_key_java.keys(), range(1,len(token_key_java)+1))))
sequences_text = list(map(lambda x: re_sequence(x, new_idxs_text), sequences_text))
sequences_java = list(map(lambda x: re_sequence(x, new_idxs_java), sequences_java))

##remake the token keys
token_key_text = dict(zip(range(1,len(token_key_text)+1), token_key_text.values()))
token_key_java = dict(zip(range(1,len(token_key_java)+1), token_key_java.values()))

##remove tokens beyond desired number of words
def remove_n_lowest_freq_keys(token_key, num_words):
    tokens_to_remove = list(range(num_words+1,len(token_key)+1))
    for token in tokens_to_remove:
        token_key.pop(token, None)
    return token_key

token_key_text = remove_n_lowest_freq_keys(token_key_text.copy(), num_words_text)
token_key_java = remove_n_lowest_freq_keys(token_key_java.copy(), num_words_java)

def UNKify_sequence(token_list, num_words):
    return [num_words+1 if token > num_words else token for token in token_list]

sequences_text = list(map(lambda x: UNKify_sequence(x, num_words_text), sequences_text))
sequences_java = list(map(lambda x: UNKify_sequence(x, num_words_java), sequences_java))

##add the UNK key
token_key_java.update({num_words_text+1:'<UNK>'})
token_key_text.update({num_words_java+1:'<UNK>'})
##add the empty padding key 
token_key_java.update({0:''})
token_key_text.update({0:''})

#get vocab size
vocab_size_java = len(token_key_java)
vocab_size_text = len(token_key_text)
    
# pad sequences
def pad_sequences_wrapper(sequences):
    max_length = max(list(map(lambda x: len(x), sequences)))
    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    return X, max_length

#encode sequences to tokenizers
X, x_max_len = pad_sequences_wrapper(sequences_text)
Y, y_max_len = pad_sequences_wrapper(sequences_java)
Y = np.expand_dims(Y,-1)

#print out stats of cleaned and tokenized inputs
max_length_text = x_max_len
max_length_java = y_max_len
print('Text Vocabulary Size: %d' % vocab_size_text)
print('Text Max Length (token number): %d' % (max_length_text))
print('Java Vocabulary Size: %d' % vocab_size_java )
print('Java Max Length (token number): %d' % (max_length_java))

#test/validation split
split_idx = round(num_examples * 0.90)
trainX = X[0:split_idx]
validX = X[split_idx:num_examples]
trainY = Y[0:split_idx]
validY = Y[split_idx:num_examples]


'''
# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
'''
'''
# define the encoder-decoder with attention model
def attention_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(AttentionDecoder(150, n_features))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model
'''
n_features = vocab_size_text
n_timesteps_in = 5
n_timesteps_out = 2
n_repeats = 10

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(AttentionDecoder(n_units, n_features))
    return model

# define model
model = define_model(vocab_size_text, 
                     vocab_size_java, 
                     max_length_text, 
                     max_length_java, 
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
          callbacks=[checkpoint],
          epochs=epoch_num,
          batch_size=batch_size, 
          validation_data=(validX, validY), 
          verbose=1)


#Evaluation
#model = load_model('model.h5')
def unsequence(sequence):
    if sum(sequence) == 0:
        return ''
    token_list = [token_key_java[token] for token in sequence]    
    return ' '.join(token_list)

def predict_sequence(model, text_sequence):
    seq = text_sequence.reshape((1, text_sequence.shape[0]))
    prediction = model.predict(seq, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    return unsequence(integers)

def compare_single_prediction(index):
    if(index < split_idx):
        ex = trainX[index]
    else:
        v_index = index - split_idx
        ex = validX[v_index]
    print("INPUT:")
    print(' '.join(re_sequence(sequences_text_full[index], token_key_text_original)))
    print("TARGET:")
    #print(' '.join(re_sequence(sequences_java_full[index], token_key_java_original)))
    print(' '.join(re_sequence(sequences_java[index], token_key_java)))
    print("OUTPUT:")
    print(predict_sequence(model, ex))


def evaluate_model(model, target_java, text_sequences):
    actual, predicted = list(), list()
    for i, source in enumerate(text_sequences):
        # translate encoded source text
        translation = predict_sequence(model, source).split()
        print(i)
        target = [' '.join(re_sequence(target_java[i], token_key_java)).split()]
        if i < 10:
            print('target=[%s],\n predicted=[%s]' % (target, translation))
        actual.append(target)
        predicted.append(translation)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


compare_single_prediction(0)

evaluate_model(model, sequences_java[0:split_idx], trainX)
evaluate_model(model, sequences_java[split_idx:num_examples], validX)



#evaluate_model(model, sequences_java[split_idx:split_idx+1], validX[0:1])

'''
file_path = '/Users/josephhiggins/Documents/mtg/mtgtext2code/models/'
file_name = 'hi_model'
model.save(file_path+file_name+'.hdf5')
loaded_model=load_model(file_path+file_name+'.hdf5')

#model = loaded_model
'''


