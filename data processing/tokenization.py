#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:41:20 2018

@author: josephhiggins
"""
from keras.preprocessing.text import Tokenizer

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)
'''
idx = data[data['card_name'] == 'sphinxofthechimes'].index[0]
print(data.loc[idx]['java_code'])
print(java_encoded[idx])
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

#extract tokens
#define up front what i think are good tokens if i say so myself
#order matters, start with super sets before subsets
pre_defined_tokens = [
    '{w}','{u}','{b}','{r}','{g}','{c}','{x}',
    '{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}','{9}','{0}',
    ';\n','{\n','}\n','\n\n',
    '<i>','</i>','$effect','\'.\'',
    '()','[]',
    '&&','||','!=','==','++','+=','--','-=',
    '{','}','(',')','\n','$',';',',','?','"','.','=','+','~'
]

#remove what i think are good tokens in a copy of text to extract the rest of 
#the tokens automatically
def remove_pd_tokens(line):
    for token in pre_defined_tokens:
        line = line.replace(token,' ')
    line = ' '.join(line.split())
    return line

stripped_java = list(map(lambda x: remove_pd_tokens(x), java_encoded))


for i in range(0,len(stripped_java)):
    if stripped_java[i].find('discardtwononlandcardswiththesamenamecost') > -1:
        print(i)

print(data.loc[8086]['java_code'])
print(java_encoded[8086])

#
stripped_tokenizer = Tokenizer(num_words=None,
                               lower=True,
                               split=' ',
                               char_level=False)

stripped_tokenizer.fit_on_texts(stripped_java)
stripped_tokenizer.word_index.keys()


output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'input_data.pkl'
text_code.to_pickle(output_file_path + file_name)