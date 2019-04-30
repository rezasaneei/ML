from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import csv
import numpy as np

docs = []

with open('movie_lines.txt') as f:
    docs = f.read().splitlines()

len_docs = len(docs)

     
#print 'number of documents' , len(docs), '\n'

# empty list to be the movie corpus
my_corpus = []  

# open a plot_summaries file containing movie summaries, from http://www.cs.cmu.edu/~ark/personas/
# and put movie summaries as a list of strings to be used as our corpus
with open('plot_summaries.txt', encoding='utf8') as f:
    rd = csv.reader(f, delimiter='\t', quotechar='"')
    for row in rd:
        my_corpus.append(row[1])
        
#print top 10 movie corpus text
#print(my_corpus[:10])

# initialize the vectorizer
vect = TfidfVectorizer(stop_words = stopwords.words('english'))

# vectorize the corpus
corpus_dtm = vect.fit(docs)

# transfor the given documents of texts according to corpus
docs_dtm = vect.transform(docs)

# print features (extracted words)
#print(vect.get_feature_names())

# make a pandas dataframe for better visualization
pd_docs = pd.DataFrame(docs_dtm.toarray(),columns = vect.get_feature_names())
#print(type(docs_dtm))

   
for i in range(len_docs):
    print ('document',i+1,'top 5 keywords :')
    print (pd_docs.loc[i].sort_values(ascending = False)[:5])
    print ('\n')

print ('Euclidean distance of each document text from others')
for i in range(len_docs):
    print ('document',i+1,'distance to others:')
    for d in docs_dtm:
        print (euclidean_distances(docs_dtm[i], d))
    print ('\n')
