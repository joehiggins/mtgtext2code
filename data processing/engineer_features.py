#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:01:08 2018

@author: josephhiggins
"""

import pandas as pd

file_path = '/Users/josephhiggins/Documents/mtg/mungeddata/'
file_name = 'merged_text_and_code.pkl'
text = pd.read_pickle(file_path + file_name)