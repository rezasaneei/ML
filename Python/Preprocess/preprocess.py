
# coding: utf-8

# # Summarizing Text with Amazon Reviews

# The objective of this project is to build a model that can create relevant summaries for reviews written about fine foods sold on Amazon. This dataset contains above 500,000 reviews, and is hosted on [Kaggle.](https://www.kaggle.com/snap/amazon-fine-food-reviews)
# 
# To build our model we will use a two-layered bidirectional RNN with LSTMs on the input data and two layers, each with an LSTM using bahdanau attention on the target data. [Jaemin Cho's tutorial](https://github.com/j-min/tf_tutorial_plus/tree/master/RNN_seq2seq/contrib_seq2seq) for seq2seq was really helpful to get the code in working order because this is my first project with TensorFlow 1.1; some of the functions are very different from 1.0. The architecture for this model is similar to Xin Pan's and Peter Liu's, here's their [GitHub page.](https://github.com/tensorflow/models/tree/master/textsum)
# 
# The sections of this project are:
# - Inspecting the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Making Our Own Summaries

# In[49]:

import pandas as pd
import numpy as np
#import tensorflow as tf
import re
from nltk.corpus import stopwords
import pickle
#import time
#from tensorflow.python.layers.core import Dense
#from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
#print('TensorFlow Version: {}'.format(tf.__version__))


# ## Insepcting the Data

# In[2]:

reviews = pd.read_csv("Reviews.csv")


# In[3]:

reviews.shape


# In[4]:

reviews.head()


# In[5]:

# Check for any nulls values
reviews.isnull().sum()


# In[6]:

# Remove null values and unneeded features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)


# In[7]:

reviews.head()


# In[8]:

# Inspecting some of the reviews
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


# ## Preparing the Data

# In[9]:

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# In[203]:

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'<br >', ' ', text) 
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
       

    text = re.sub(r'\'', ' ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# We will remove the stopwords from the texts because they do not provide much use for training our model. However, we will keep them for our summaries so that they sound more like natural phrases. 

# In[204]:

# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")


# In[206]:

# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(5):
    print("Clean Review #",i+1)
    print(clean_summaries[i])
    print(clean_texts[i])
    print()


# In[207]:

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# In[208]:

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)
            
print("Size of Vocabulary:", len(word_counts))


# In[209]:

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
with open('numberbatch-en-17.02.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))


# In[210]:

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# I use a threshold of 20, so that words not in CN can be added to our word_embedding_matrix, but they need to be common enough in the reviews so that the model can understand their meaning.

# In[211]:

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

#save int to vocab dictionary in a file

a = {'hello': 'world'}

with open('intToVocabDict', 'wb') as handle:
    pickle.dump(int_to_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


# In[212]:

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


# In[213]:

def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# In[214]:

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


# In[215]:

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


# In[216]:

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())


# In[217]:

# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))


# In[218]:

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))


# In[219]:

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# In[220]:

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
# Original data
sorted_original_summaries=[]
sorted_original_texts = []

max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length): 
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])
            
            sorted_original_summaries.append(reviews.Summary[count])
            sorted_original_texts.append(reviews.Text[count])
        
# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


# ## Building the Model

# In[221]:




# In[222]:




# In[223]:




# In[228]:

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[229]:

def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# In[230]:

# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75


# In[231]:

# ## Training the Model

# Since I am training this model on my MacBook Pro, it would take me days if I used the whole dataset. For this reason, I am only going to use a subset of the data, so that I can train it over night. Normally I use [FloydHub's](https://www.floydhub.com/) services for my GPU needs, but it would take quite a bit of time to upload the dataset and ConceptNet Numberbatch, so I'm not going to bother with that for this project.
# 
# I chose not use use the start of the subset because I didn't want to make it too easy for my model. The texts that I am using are closer to the median lengths; I thought this would be more fair.

# In[234]:

# Subset the data for training
start = 1
end = start + 500000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
# Original data 
sorted_original_summaries_short = sorted_original_summaries[start:end]
sorted_original_texts_short = sorted_original_texts[start:end]

print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))

print("Length of the summaries list:",len(sorted_summaries))
print("Legth of the shorted summaries:",len(sorted_summaries_short))

with open('Summaries-file','wb') as fb:
    pickle.dump(sorted_summaries_short,fb)
    
with open('Texts-file','wb') as fb:
    pickle.dump(sorted_texts_short,fb)   
    
    
#Creat file of original data lists
with open('Original-Summaries-file','wb') as fb:
    pickle.dump(sorted_original_summaries_short,fb)
    
with open('Original-Texts-file','wb') as fb:
    pickle.dump(sorted_original_texts_short,fb) 


# In[158]:

# Train the Model
#learning_rate_decay = 0.95
#min_learning_rate = 0.0005
#display_step = 20 # Check training loss after every 20 batches
#stop_early = 0 
#stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
#per_epoch = 3 # Make 3 update checks per epoch
#update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1
#
#update_loss = 0 
#batch_loss = 0
#summary_update_loss = [] # Record the update losses for saving improvements in the model
#
#checkpoint = "best_model.ckpt" 

# ## Making Our Own Summaries

# To see the quality of the summaries that this model can generate, you can either create your own review, or use a review from the dataset. You can set the length of the summary to a fixed value, or use a random value like I have here.

# In[114]:

def text_to_seq(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# In[167]:

# Create your own review or use one from the dataset
#input_sentence = "I have never eaten an apple before, but this red one was nice. \
                  #I think that I will try a green apple next time."
#text = text_to_seq(input_sentence)
random = np.random.randint(0,len(clean_texts))
input_sentence = clean_texts[random]
text = text_to_seq(clean_texts[random])

checkpoint = "./best_model.ckpt"



# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"] 

print('Original Text:', input_sentence)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')



# Examples of reviews and summaries:
# - Review(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
# - Summary(1): great coffee
# 
# 
# - Review(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!
# - Summary(2): omg gross gross
# 
# 
# - Review(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
# - Summary(3): love it

# ## Summary

# I hope that you found this project to be rather interesting and informative. One of my main recommendations for working with this dataset and model is either use a GPU, a subset of the dataset, or plenty of time to train your model. As you might be able to expect, the model will not be able to make good predictions just by seeing many reviews, it needs so see the reviews many times to be able to understand the relationship between words and between descriptions & summaries. 
# 
# In short, I'm pleased with how well this model performs. After creating numerous reviews and checking those from the dataset, I can happily say that most of the generated summaries are appropriate, some of them are great, and some of them make mistakes. I'll try to improve this model and if it gets better, I'll update my GitHub.
# 
# Thanks for reading!

# In[ ]:



