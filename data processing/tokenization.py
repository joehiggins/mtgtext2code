#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:41:20 2018

@author: josephhiggins
"""

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)
'''
data[data['card_name'] == 'fireservant'].index[0]
print(data.loc[988]['java_code'])
print(java_encoded[988])
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
    '{','}','(',')','\n','$',';',',','?','"',".","=","+"
]

#remove what i think are good tokens in a copy of text to extract the rest of 
#the tokens automatically
def remove_pd_tokens(line):
    for token in pre_defined_tokens:
        line = line.replace(token,' ')
    line = ' '.join(line.split())
    return line

stripped_java = list(map(lambda x:, remove_pd_tokens)


#
tokenizer = Tokenizer(num_words=None
        lower=True, 
        #split=' ', 
        char_level=False)
    tokenizer.fit_on_texts(lines)


output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'input_data.pkl'
text_code.to_pickle(output_file_path + file_name)