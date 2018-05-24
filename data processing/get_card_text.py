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

def get_field(name, field):
    return json_data[name].get(field) if json_data[name].get(field) else '<NA>'
    
def clean_lower_no_char(card_name):
    clean = re.sub('[,\- \']', '', card_name)
    clean = clean.lower()
    return clean

def clean_lower(card_name):
    clean = card_name.lower()
    return clean


card_names = list(json_data.keys())
card_names_lower_no_char = list(map(lambda x: clean_lower_no_char(x), card_names))
card_names_lower = list(map(lambda x: clean_lower(x), card_names))

manaCost =  list(map(lambda x: get_field(x, 'manaCost'), card_names))
colors =    list(map(lambda x: get_field(x, 'colors'), card_names))
layout =    list(map(lambda x: get_field(x, 'layout'), card_names))
supertype = list(map(lambda x: get_field(x, 'type'), card_names))
types =     list(map(lambda x: get_field(x, 'types'), card_names))
power =     list(map(lambda x: get_field(x, 'power'), card_names))
toughness = list(map(lambda x: get_field(x, 'toughness'), card_names))
printings = list(map(lambda x: get_field(x, 'printings'), card_names))
text =      list(map(lambda x: get_field(x, 'text'), card_names))


output = pd.DataFrame({
    'card_name': card_names_lower_no_char,
    'card_name_self': card_names_lower,
    'manaCost': manaCost,
    'layout': layout,
    'colors': colors,
    'type': supertype,
    'types': types,
    'power': power,
    'toughness': toughness,
    'printings': printings,
    'text': text,
})
    
output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_card_text.pkl'
output.to_pickle(output_file_path + file_name)


#text[card_names.index('Wojek Siren')]
#text[card_names.index('Grizzly Bears')]
#json_data['Wojek Siren']
#json_data['Grizzly Bears']