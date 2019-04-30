# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:23:07 2019

@author: rezas
"""

import nltk
import re
import pprint
from nltk import Tree

my_corpus = ""

with open('movie_lines.txt', 'r') as myfile:
    my_corpus = myfile.read().replace('\n', '')


grammar = """
    NP:    {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
           {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS>+}
           
    """

NPChunker = nltk.RegexpParser(grammar)

def prepare_text(input):
     tokenized_sentence = nltk.sent_tokenize(input)  # Tokenize the text into sentences.
     tokenized_words = [nltk.word_tokenize(sentence) for sentence in tokenized_sentence]  # Tokenize words in sentences.
     tagged_words = [nltk.pos_tag(word) for word in tokenized_words]  # Tag words for POS in each sentence.
     word_tree = [NPChunker.parse(word) for word in tagged_words]  # Identify NP chunks
     return word_tree  # Return the tagged & chunked sentences.

sentences = prepare_text(my_corpus)
for tree in sentences:
    tree.draw()

def return_a_list_of_NPs(sentences):
    nps = []  # an empty list in which to NPs will be stored.
    for sent in sentences:
        tree = NPChunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                t = subtree
                t = ' '.join(word for word, tag in t.leaves())
                nps.append(t)
    return nps

print(return_a_list_of_NPs(sentences))




