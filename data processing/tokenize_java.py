#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:41:20 2018

@author: josephhiggins
"""
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
from unicodedata import normalize
from collections import Counter
import pickle


file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)
'''
idx = data[data['card_name'] == 'sphinxofthechimes'].index[0]
print(data.loc[idx]['text'])
print(data.loc[idx]['java_code'])
print(java_encoded[idx])

for i in range(0,len(stripped_java)):
    if stripped_java[i].find('discardtwononlandcardswiththesamenamecost') > -1:
        print(i)
        
for i in range(0,len(auto_gend_tokens)):
    if re.match("^.$", auto_gend_tokens[i]):
        print(str(i) + ': ' + auto_gend_tokens[i])
        
'''
java_encoded = data['java_code']
card_names = data['card_name']

#encode to utf-8
def utf8_encode(line):
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    return line

java_encoded = list(map(lambda x: utf8_encode(x), java_encoded))

#replace names with symbol
##which symbol safe to use?
def has_symbol(line, symbol):
    return line.find(symbol)

symb = '$'
test = list(map(lambda x: has_symbol(x,symb), data['java_code']))
if not [x for x in test if x != -1]:
    print(symb + ' is safe to use')

##replacement
java_encoded = list(map(lambda x: x[0].replace(x[1], symb),
                        zip(java_encoded,card_names)))

#JCH TODO: perhaps add carets ^^^ for power and toughness indicators?

#tokenize
no_space_mwes = [
    ('{','w','}'),('{','u','}'),('{','b','}'),('{','r','}'),('{','g','}'),
    ('{','c','}'),('{','x','}'),
    ('{','1','}'),('{','2','}'),('{','3','}'),('{','4','}'),('{','5','}'),
    ('{','6','}'),('{','7','}'),('{','8','}'),('{','9','}'),('{','0','}'),
    ('+','1'),('+','2'),('+','3'),('+','4'),('+','5'),('+','6'),('+','x'),
    ('-','1'),('-','2'),('-','3'),('-','4'),('-','5'),('-','6'),('-','x'),
    ('<','i','>'),('<','/','i','>'),('\'','.','\''),('\'','?','\''),
    ('(',')'),('[',']'),('{','}'),
    ('&','&'),('|','|'),('=','='),('!','='),('+','+'),('+','='),
    ('-','-'),('-','='),
    (')',';'),
    ('$','effect'),
    ('$','(','this',')'),
    ('super','(','card',')'),
    ('copy','(',')'),
    #('this.addability','(','vigilanceability.getinstance','(',')',')',';'),
    #('this.addability','(','flyingability.getinstance','(',')',')',';'),
]

space_mwes = [
    ('public','class','$','extends'),
    ('public','$', 'copy()'),
    ('return','new','$(this)'),
]

#create tokenizers
no_space_mwe_tokenizer = MWETokenizer(separator='')
space_mwe_tokenizer = MWETokenizer(separator=' ')

for token in no_space_mwes:
    no_space_mwe_tokenizer.add_mwe(token)
for token in space_mwes:
    space_mwe_tokenizer.add_mwe(token)

#tokenize sentences
def tokenize_java(java_representation):
    line = ' . '.join(java_representation.split('.'))
    line = ' / '.join(line.split('/'))
    line = ' ++ '.join(line.split('++'))
    word_level = word_tokenize(line)
    no_space_level = no_space_mwe_tokenizer.tokenize(word_level)
    space_level = space_mwe_tokenizer.tokenize(no_space_level)
    return space_level

encoded_java_tokenized = list(map(lambda x: tokenize_java(x), java_encoded))
tokenized_flat = [item for sublist in encoded_java_tokenized for item in sublist]
tokenized_counter = Counter(tokenized_flat)
tokens = list(list(zip(*tokenized_counter.most_common()))[0])
counts = list(list(zip(*tokenized_counter.most_common()))[1])

word_index = dict(zip(tokens, range(1,len(tokens)+1)))

#JCH todo: add this ability to substitude in <UNK> tokens

#encode to number
#order from largest to smallest, so bigger tokens get applied first
def sequence_java_tokens(token_list):
    return [word_index[token] for token in token_list]

sequenced = list(map(lambda x: sequence_java_tokens(x), encoded_java_tokenized))
    
output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'sequenced_java.pkl'
with open(output_file_path + file_name, 'wb') as fp:
    pickle.dump(sequenced, fp)

token_key = dict(zip( range(1,len(tokens)+1),tokens))
file_name = 'java_token_key.pkl'
with open(output_file_path + file_name, 'wb') as fp:
    pickle.dump(token_key, fp)

