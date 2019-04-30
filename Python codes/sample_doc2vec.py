# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:14:24 2019

@author: rezas
"""

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
from matplotlib import pyplot

print common_texts
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=7)

from gensim.test.utils import get_tmpfile

fname = get_tmpfile("my_doc2vec_model")

model.save(fname)

model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

vector = model.infer_vector(["system", "response"])

print vector

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
