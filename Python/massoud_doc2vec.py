# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:23:04 2019

@author: rezas
"""

from scipy import spatial
import csv
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot

docs = []
with open('movie_lines.txt') as f:
    docs = f.read().splitlines()

len_docs = len(docs)

my_corpus = []        
    
with open('plot_summaries.txt') as f:
    rd = csv.reader(f, delimiter='\t', quotechar='"')
    for row in rd:
        my_corpus.append(row[1])
        
documents = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(my_corpus):
    words = text.lower().split()
    tags = [i]
    documents.append(analyzedDocument(words, tags))
    
#print documents[:10]

#model = Doc2Vec(documents, dm = 0, alpha = 0.025, min_alpha = 0.001, min_count = 1, vector_size = 20, workers=7, epochs = 100)

#model.save('my_model.doc2vec')

model = Doc2Vec.load('my_model.doc2vec')  # you can continue training with the loaded model!

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


doc_vector = []
for d in docs:   
    doc_vector.append(model.infer_vector(d))

print 'Euclidean distance of each document text to others'
for i in range(len_docs):
    print 'document',i+1,'distance to others:'
    for j in range(len_docs):
        print spatial.distance.euclidean(np.array(doc_vector[i]),np.array(doc_vector[j]))
    print '\n'
    
print 'Cosine similarity of each document text to others'
for i in range(len_docs):
    print 'document',i+1,'similarity to others:'
    for j in range(len_docs):
        print (1 - spatial.distance.cosine(np.array(doc_vector[i]), np.array(doc_vector[j])))*100,'%'
    print '\n'
   
X = doc_vector
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
for i, doc in enumerate(doc_vector):
	pyplot.annotate(i+1, xy=(result[i, 0], result[i, 1]))
pyplot.show()
