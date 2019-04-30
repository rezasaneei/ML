# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 02:56:05 2019

@author: rezas
"""

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk import CFG

docs = []

with open('movie_lines.txt') as f:
    docs = f.read().splitlines()

#rules = nltk.data.load('grammars/large_grammars/atis.cfg', 'text')
#grammar = nltk.parse_cfg(rules)
#parser =  nltk.parse.ChartParser(grammar)
                                         
stop_words = set(stopwords.words('english'))
grammar = "NP: {<DT>?<JJ>*<NN>}"
parsed_grammar = nltk.parse_cfg(grammar)
parser = nltk.ChartParser(parsed_grammar)
clean_sent = ''

for doc in docs:
    sents = sent_tokenize(doc)
    for tree in parser.parse(str(sents)):
        tree.draw()
    
    
    
#    for sent in sents:
#        word = word_tokenize(sent)
#        for i in word:
#            if i not in stop_words:
#                print i
#                clean_sent = clean_sent + i + ' '  
#        print clean_sent
##        print type(clean_sent)
#        print list(clean_sent)
#        for tree in parser.parse(list(clean_sent)):
#            tree.draw()
