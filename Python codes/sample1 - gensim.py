# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:32:11 2018

@author: rezas
"""

from gensim.summarization import keywords

doc1 = 'hello, my name is reza'
doc2 = 'Reza is a fine man'

print(keywords(doc1))