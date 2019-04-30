import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


custom_sent_tokenizer = PunktSentenceTokenizer()

with open('movie_lines.txt','r') as f:
    docs = f.read().splitlines()
    sample_text = str(docs)
    
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                                  }<VB.?|IN|DT>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse (tagged)
            chunked.draw()

            
    except Exception as e:
        print(str(e))


process_content()
