#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:05:50 2018

@author: josephhiggins
"""

import json
import os

i_file_root = '/Users/josephhiggins/Documents/mtg/mage/Mage.Sets/src/mage/cards/'
i_file_path = i_file_root

counter = 0

#traverse the entire directory
for root, dirs, files in os.walk(i_file_root):
    for file in files:
        full_file_path = root + os.sep + file

        if(full_file_path[-5:] != '.java'):
            continue
        
        card_name = file[:-5]
        
        with open(full_file_path, 'r') as myfile:
            #data=myfile.read().replace('\n', '')
            data=myfile.read()

        
        
data    
