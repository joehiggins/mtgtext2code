#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:05:50 2018

@author: josephhiggins
"""

import pandas as pd
import json
import os
import re

i_file_root = '/Users/josephhiggins/Documents/mtg/mage/Mage.Sets/src/mage/cards/'

#utility functions
def is_empty(java_line):
    if(java_line == ''):
        return True
    else:
        return False
    
def is_comment(java_line):
    if(re.match(r"^[*]", java_line) or
       re.match(r"^[/]+[*]", java_line) or
       re.match(r"^[ ]+[*]", java_line) or
       re.match(r"^[*]+[/]", java_line) or
       re.match(r"^\s*[/]+[/]", java_line)
     ):    
        return True
    else:
        return False 
    
def is_java_util(java_line):
    if(re.match("^package", java_line) or
       re.match("^import", java_line) or
       re.match("^.*Override*$", java_line)
     ):    
        return True
    else:
        return False 

def clean_name(card_name):
    clean = re.sub('[,\- \']', '', card_name)
    clean = clean.lower()
    return clean

card_names_clean = []
targets = []
#traverse the entire directory
for root, dirs, files in os.walk(i_file_root):
    for file in files:
        full_file_path = root + os.sep + file

        if(full_file_path[-5:] != '.java'):
            continue
        
        card_name_clean = clean_name(file[:-5])
        
        with open(full_file_path, 'r') as myfile:   
            data=myfile.read().split('\n')

        data_f = list(filter(lambda line: not is_empty(line), data))
        data_f = list(filter(lambda line: not is_comment(line), data_f))
        data_f = list(filter(lambda line: not is_java_util(line), data_f))
        
        target = ("\n").join(data_f)
        target = re.sub(' +',' ', target)
        target = target.lower()

        card_names_clean.append(card_name_clean)
        targets.append(target)

output = pd.DataFrame({
    'card_name': card_names_clean,
    'java_code': targets,
})

    
output_file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'card_name_to_java_code.pkl'
output.to_pickle(output_file_path + file_name)