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
            chunkGram1 = "NP:{<DT>?<JJ>*<NN>}"
            chunkGram2 = r"""Chunk: {<.*>+}
                                  }<VB.?|IN|DT>+{"""
            chunkGram3 = """
                NP: {<JJ>*<NN>+}
                {<JJ>*<NN><CC>*<NN>+}
                """
            chunkGram4 = """
                NP:    {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
                       {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                       {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                       {<JJ>*<NN|NNS|NNP|NNPS>+}
                       """
            chunkParser = nltk.RegexpParser(chunkGram4)
            chunked = chunkParser.parse (tagged)
            chunked.draw()

            
    except Exception as e:
        print(str(e))


process_content()
