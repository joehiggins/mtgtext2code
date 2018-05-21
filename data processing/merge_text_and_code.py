#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:31:49 2018

@author: josephhiggins
"""

import pandas as pd

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_card_text.pkl'
text = pd.read_pickle(file_path + file_name)
file_name = 'card_name_to_java_code.pkl'
code = pd.read_pickle(file_path + file_name)

text_code = pd.DataFrame()
text_code = pd.merge(text, code, on='card_name', how='outer')
text_code = text_code.rename(columns={
        'card_name': 'card_name',
        'java_code_x': 'card_text', 
        'java_code_y': 'java_code'})


print("text_code: "+str(len(text_code))+"\n"+\
     "text: "+str(len(text))+"\n"+\
     "code: "+str(len(code)))

list(text_code[text_code['card_name'] == 'grizzlybears']['java_code'])