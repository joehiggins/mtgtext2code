#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:41:38 2018

@author: josephhiggins
"""

import pandas as pd
import re

file_path = '/Users/josephhiggins/Documents/mtg/mtgencode/data/'
file_name = 'output.txt'

def find_card_name(encoded_card):
    start_idx = encoded_card.find('|1', 0)+2
    str_len = encoded_card[start_idx:].find('|',0)
    name = encoded_card[start_idx:start_idx+str_len]
    clean = re.sub('[~,\- \']', '', name)
    clean = clean.lower()
    return clean

data = open(file_path + file_name, 'r')
card_list = data.read().split('\n\n')
card_names = list(map(lambda x: find_card_name(x), card_list))

output = pd.DataFrame({
    'card_name': card_names,
    'mtgencoded_text': card_list,
})
    
output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_encoded_text.pkl'
output.to_pickle(output_file_path + file_name)
