# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:33:31 2019

@author: rezas
"""

# imports needed and logging
import gzip
import gensim 
import logging
 
#logging.basicConfig(format=’%(asctime)s %(levelname)s %(message)s’, level=logging.INFO)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='myapp.log', filemode='w')

logging.info("reading file {0}...this may take a while".format('reviews_data.txt.gz'))

with gzip.open ('reviews_data.txt.gz', 'rb') as f:
        for i,line in enumerate (f):
            for i, line in enumerate(f):
                if (i % 10000 == 0):
                    logging.info("read {0} reviews".format(i))
                    # do some pre-processing and return list of words for each review
                    # text
                documents = gensim.utils.simple_preprocess(line)
                

# build vocabulary and train model
model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10)

model.train(documents, total_examples=len(documents), epochs=10)

w1 = "dirty"
model.wv.most_similar(positive = w1)
                