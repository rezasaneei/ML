from sklearn.feature_extraction.text import CountVectorizer

Sentences = ['We are using the Bag of Word model', 'Bag of Word model is used for extracting the features.']

vectorizer_count = CountVectorizer()

features_text = vectorizer_count.fit_transform(Sentences).todense()

print(vectorizer_count.vocabulary_)
