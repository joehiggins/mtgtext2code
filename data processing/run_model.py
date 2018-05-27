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
import pickle

def open_pickle(file_name):
    file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
    return pickle.load( open(file_path+file_name, "rb" ) )

sequences_java_full = open_pickle('sequenced_java.pkl')
sequences_text_full = open_pickle('sequenced_text.pkl')
token_key_java_original = open_pickle('java_token_key.pkl')
token_key_text_original = open_pickle('text_token_key.pkl')

###parameters
num_examples = 2
epoch_num = 1000
batch_size = 1

###test small sample
sequences_text = sequences_text_full[0:num_examples]
sequences_java = sequences_java_full[0:num_examples]

#update tokens based on the data sample we use
##add spaces

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

##add the empty padding key 
token_key_java.update({0:''})
token_key_text.update({0:''})
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
          #callbacks=[checkpoint],
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
        ex = trainX[index]
    else:
        v_index = index - split_idx
        ex = validX[v_index]
    print("INPUT:")
    print(' '.join(re_sequence(sequences_text_full[index], token_key_text_original)))
    print("TARGET:")
    print(' '.join(re_sequence(sequences_java_full[index], token_key_java_original)))
    print("OUTPUT:")
    print(predict_sequence(model, ex))

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


