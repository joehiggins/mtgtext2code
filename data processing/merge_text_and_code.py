#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:31:49 2018

@author: josephhiggins
"""

import pandas as pd
import numpy as np
import math

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_card_text.pkl'
text = pd.read_pickle(file_path + file_name)
file_name = 'card_name_to_encoded_text.pkl'
encoded_text = pd.read_pickle(file_path + file_name)
file_name = 'card_name_to_java_code.pkl'
code = pd.read_pickle(file_path + file_name)

text_code = pd.DataFrame()
text_code = pd.merge(text, code, on='card_name', how='inner')
text_code = pd.merge(text_code, encoded_text, on='card_name', how='inner')

#remove unstable (joke set)
def check_list_members(x, member):
    try:
        if(math.isnan(x)):
            return False
    except:
        return member in x 

UST_filter = list(map(lambda x: not check_list_members(x, 'UST'), text_code['printings']))
UNH_filter = list(map(lambda x: not check_list_members(x, 'UNH'), text_code['printings']))
UGL_filter = list(map(lambda x: not check_list_members(x, 'UGL'), text_code['printings']))
total_filter = [ust and unh and ugl for ust, unh, ugl in zip(UST_filter,UNH_filter,UGL_filter)]
text_code = text_code[total_filter]

#remove very long java examples(~11% of all examples at 2000)
java_lengths = list(map(lambda x: len(x), text_code['java_code']))
short_indexes = [True if x < 2000 else False for x in java_lengths] 
text_code = text_code[short_indexes]
##sum(short_indexes)/len(short_indexes)

output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
text_code.to_pickle(output_file_path + file_name)

#list(text_code[text_code['card_name'] == 'grizzlybears']['java_code'])
#list(text_code[text_code['card_name'] == 'adorablekitten']['printings'])