# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 03:40:23 2019

@author: rezas
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))

with open('movie_lines.txt', 'r') as text_file:
    doc = text_file.readline()
    for sent in doc:
        tokens= sent_tokenize(sent)
        print tokens
#    token = [w for w in tokens if not w in stopset]
#    print token