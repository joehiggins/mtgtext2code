#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:05:02 2018

@author: josephhiggins
"""

import pandas as pd
import json
import os
import re

file_path = '/Users/josephhiggins/Documents/mtg/mtgjson/'
file_name = 'AllCards-x.json'


with open(file_path + file_name) as data_file:
    json_data = json.load(data_file)

def get_card_text(card_name):
    if('text' in json_data[card_name]):
        return json_data[card_name]['text'].lower()
    else:
        return '<NONE>'
    
def clean_name(card_name):
    clean = re.sub('[,\- \']', '', card_name)
    clean = clean.lower()
    return clean

card_names = list(json_data.keys())
card_texts = list(map(lambda x: get_card_text(x), card_names))
card_names_clean = list(map(lambda x: clean_name(x), card_names))

output = pd.DataFrame({
    'card_name': card_names_clean,
    'java_code': card_texts,
})

    
output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_card_text.pkl'
output.to_pickle(output_file_path + file_name)