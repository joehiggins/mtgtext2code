#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:01:08 2018

@author: josephhiggins
"""


#https://github.com/billzorn/mtgencode
#https://github.com/billzorn/mtgencode#training-a-neural-net
#http://minimaxir.com/2017/04/char-embeddings/, Max Woolf (@minimaxir)
#https://github.com/minimaxir/char-embeddings/blob/master/create_magic_text.py


import pandas as pd
import sys
'''
sys.path.insert(0, '/Users/josephhiggins/Documents/CS 224U/sippycup/')
from parsing import Grammar, Rule
'''

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
data = pd.read_pickle(file_path + file_name)

#Prepare output
feature_data = pd.DataFrame()

#Create type indicator variables