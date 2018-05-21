#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:05:50 2018

@author: josephhiggins
"""

import json
import os
import re

i_file_root = '/Users/josephhiggins/Documents/mtg/mage/Mage.Sets/src/mage/cards/a/'
i_file_path = i_file_root

counter = 0

#utility functions
def is_empty(java_line):
    if(java_line == ''):
        return True
    else:
        return False
    
    
def is_comment(java_line):
    if(re.match(r"^[/]+[*]", java_line) or
       re.match(r"^[ ]+[*]", java_line) or
       re.match(r"^[*]+[/]", java_line) or
       re.match(r"^[/]+[/]", java_line)
     ):    
        return True
    else:
        return False 
    
def is_lib_import(java_line):
    if(re.match("^package", java_line) or
       re.match("^import", java_line)
     ):    
        return True
    else:
        return False 




#traverse the entire directory
for root, dirs, files in os.walk(i_file_root):
    for file in files:
        full_file_path = root + os.sep + file

        if(full_file_path[-5:] != '.java'):
            continue
        
        card_name = file[:-5]
        
        with open(full_file_path, 'r') as myfile:   
            data=myfile.read().split('\n')

        data_f = list(filter(lambda line: not is_empty(line), data))
        data_f = list(filter(lambda line: not is_comment(line), data_f))
        data_f = list(filter(lambda line: not is_lib_import(line), data_f))

'''
re.match(r"^[/]+[*]", "fas*hi")
re.match(r"^[/]+[/]", "//hi")
re.match(r"^[ ]+[*]", " */ hi")
'''

data_f

