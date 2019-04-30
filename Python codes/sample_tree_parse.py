# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 03:21:32 2019

@author: rezas
"""

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


groucho_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")

docs = []

with open('movie_lines.txt') as f:
    docs = f.read().splitlines()
    for doc in docs:
        sent = sent_tokenize(doc)
        parser = nltk.ChartParser(groucho_grammar)
#        for tree in parser.parse(sent):
#            tree.draw()