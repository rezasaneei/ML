# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:07:50 2019

@author: rezas
"""
import numpy as np
from scipy import spatial
import pandas as pd
import csv
import gensim

docs = []

with open('movie_lines.txt') as f:
    docs = f.read().splitlines()

len_docs = len(docs)

print(docs[0])
     
#print 'number of documents' , len(docs), '\n'

# empty list to be the movie corpus
my_corpus = []  

# open a plot_summaries file containing movie summaries, from http://www.cs.cmu.edu/~ark/personas/
# and put movie summaries as a list of strings to be used as our corpus
#with open('plot_summaries.txt') as f:
#    rd = csv.reader(f, delimiter='\t', quotechar='"')
#    for row in rd:
#        my_corpus.append(row[1])
        
with open('plot_summaries.txt', 'r') as myfile:
    my_corpus = myfile.read()

        
model = gensim.models.Word2Vec(my_corpus, min_count=1)
model.train(my_corpus, total_examples=len(my_corpus), epochs=5)

words = list(model.wv.vocab)
print(words)

s1_afv = avg_feature_vector(docs[0], model=model, num_features=300, index2word_set=index2word_set)
s2_afv = avg_feature_vector(docs[1], model=model, num_features=300, index2word_set=index2word_set)
sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
print(sim)


index2word_set = set(model.wv.index2word)