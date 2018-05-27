#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:53:11 2018

@author: josephhiggins
"""

'''
for i in range(0,len(text_encoded)):
    if text_encoded[i].find('EEEE') > -1:
        print(str(i) + ": " + text_encoded[i])
'''

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
from unicodedata import normalize
from collections import Counter
import pickle

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)

text_encoded = data['mtgencoded_text']
card_names = data['card_name_self']

def utf8_encode(line):
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    return line

text_encoded = list(map(lambda x: utf8_encode(x), text_encoded))
card_names = list(map(lambda x: utf8_encode(x), card_names))

#replace names with symbol
##which symbol safe to use?
def has_symbol(line, symbol):
    return line.find(symbol)

symb = '$'
test = list(map(lambda x: has_symbol(x,symb), data['mtgencoded_text']))
if not [x for x in test if x != -1]:
    print(symb + ' is safe to use')
    
##replacement
text_encoded = list(map(lambda x: x[0].replace(x[1], symb),
                        zip(text_encoded, card_names)))

#tokenize
no_space_mwes = [
    ('|','0'),('|','1'),('|','2'),('|','3'),('|','4'),
    ('|','5'),('|','6'),('|','7'),('|','8'),('|','9'),
]

space_mwes = [
    ('when','@','enters','the','battlefield'),
]

seqs_to_insert_spaces_for = [
   '|0','|1','|2','|3','|4','|5','|6','|7','|8','|9','|',
   '/','.','?','\\',
   'XX','CC','EE',
   'WW','UW','BW','RW','GW',
   'WU','UU','BU','RU','GU',
   'WB','UB','BB','RB','GB',
   'WR','UR','BR','RR','GR',
   'WG','UG','BG','RG','GG',
   'WP','UP','BP','RP','GP',  
]

#create tokenizers
no_space_mwe_tokenizer = MWETokenizer(separator='')
space_mwe_tokenizer = MWETokenizer(separator=' ')

for token in no_space_mwes:
    no_space_mwe_tokenizer.add_mwe(token)
for token in space_mwes:
    space_mwe_tokenizer.add_mwe(token)
    
#tokenize sentences
def tokenize(representation):
    line = representation
    for seq in seqs_to_insert_spaces_for:
        if seq in line:
            line = (' '+seq+' ').join(line.split(seq))
    word_level = word_tokenize(line)
    no_space_level = no_space_mwe_tokenizer.tokenize(word_level)
    space_level = space_mwe_tokenizer.tokenize(no_space_level)
    return space_level

test = text_encoded[5]
tokenize(test)

encoded_text_tokenized = list(map(lambda x: tokenize(x), text_encoded))
tokenized_flat = [item for sublist in encoded_text_tokenized for item in sublist]
tokenized_counter = Counter(tokenized_flat)
tokens = list(list(zip(*tokenized_counter.most_common()))[0])
counts = list(list(zip(*tokenized_counter.most_common()))[1])

word_index = dict(zip(tokens, range(0,len(tokens))))

#JCH todo: add this ability to substitude in <UNK> tokens for rare words

#encode to number
#order from largest to smallest, so bigger tokens get applied first
def sequence_tokens(token_list):
    return [word_index[token] for token in token_list]

sequenced = list(map(lambda x: sequence_tokens(x), encoded_text_tokenized))

output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'sequenced_text.pkl'
with open(output_file_path + file_name, 'wb') as fp:
    pickle.dump(sequenced, fp)


