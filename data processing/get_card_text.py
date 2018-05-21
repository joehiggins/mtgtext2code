#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:05:02 2018

@author: josephhiggins
"""

import json
import os

file_path = '/Users/josephhiggins/Documents/mtg/mtgjson/'
file_name = 'AllCards-x.json'


with open(file_path + file_name) as data_file:
    json_data = json.load(data_file)

def get_card_text(card_name):
    if('text' in json_data[card_name]):
        return json_data[card_name]['text']
    else:
        return '<NONE>'

card_names = list(json_data.keys())
card_texts = list(map(lambda x: get_card_text(x), list(json_data.keys())))


card_texts[card_names.index('Wojek Siren')]