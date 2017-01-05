import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def get_cleaned_tokens(text):
    """ Returns a list of tokens from text, cleaned 
        and with stopwords removed. """
    words = nltk.word_tokenize(text.lower())
    stops = set(stopwords.words("english"))
    tokens = [w.encode('ascii',errors='ignore').decode()
              for w in words if w[0].isalpha() and w not in stops]
    return tokens


def frequency_from_postings(postings):
    c = Counter()
    for text in postings.values():
        for token in set(get_cleaned_tokens(text)):
            c[token] += 1
    return c


def vectorize_postings(postings):
    texts = postings.values()
    return vectorize_texts(texts)

def vectorize_texts(texts,min_df=0.1):
    V = CountVectorizer(min_df=min_df, analyzer=get_cleaned_tokens)
    X = V.fit_transform(texts)
    return V,X

