import numpy as np
import pandas as pd
import nltk
import re
import os,sys
import codecs
from sklearn import feature_extraction
import mpld3
import glob
import random
import spacy
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
encodingTot = sys.stdout.encoding or 'utf-8'
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import *
from sklearn.externals import joblib

from decimal import Decimal
from sklearn.cluster import MeanShift, estimate_bandwidth

stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


all_files_txt = glob.glob(r"*.txt")
tweet_df = pd.concat((pd.read_csv(f,names=['id','time','tweet'], delimiter='|',encoding = "ISO-8859-1").assign(news_agency=os.path.basename(f).replace('.txt','')) for f in all_files_txt), ignore_index=True)
tweet_df.drop('id',axis=1,inplace=True)
tweets = [re.sub(r"http\S+", "", tweet) for tweet in tweet_df['tweet'].tolist()]

#tweets = tweet_df['tweet'].tolist()
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in tweets:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0.05,stop_words = 'english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(tweets ) #fit the vectorizer to synopses
#####################################
dist = cosine_distances(tfidf_matrix)
dis2 = euclidean_distances(tfidf_matrix)
#####################################
bandwidth = estimate_bandwidth(dis2, quantile=0.5, n_samples=200)
print(bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1)
ms = ms.fit(dis2)
clusters = ms.labels_.tolist()
skplt.metrics.plot_silhouette(dis2, clusters)
plt.show()
plt.savefig('cosine_avergae.png', dpi=200) #save figure

joblib.dump(ms,"ms_eud.pkl")

cleaned_tweets = [re.sub(r"http\S+", "", tweet) for tweet in tweet_df['tweet'].tolist()]
cluster_tweet = pd.DataFrame({"Tweet": cleaned_tweets, "Cluster_Num": clusters})

for word in ['rt','video','new','d','workload','say','RT','good','u','http','cnn','cnnhealth','want','amp','drsanjaygupta','know','way','year','official','cdcgov','minute','help']:
    stop_words.add(word)
    w = WordNetLemmatizer().lemmatize(word)
    stop_words.add(w)
for cluster in cluster_tweet['Cluster_Num'].unique():
    print (cluster)
    df = cluster_tweet[cluster_tweet.Cluster_Num == cluster]
    new = df['Tweet'].str.split('http://',n = 1,expand = True)
    df['Tweet']= [re.sub(r'\b\w{1,3}\b', '', tweet ) for tweet in new[0]]
    tokens = word_tokenize(str(df['Tweet'].tolist()))
    
    # Convert the tokens into lowercase: lower_tokens
    lower_tokens = [t.lower() for t in tokens]
    
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stop_words]
    # Lemmatize all tokens into a new list: lemmatized
    lemmatized = [WordNetLemmatizer().lemmatize(t) for t in no_stops]

    # Create the bag-of-words: bow
    bow = Counter(lemmatized)
   # bow_filter = [w for w,y in bow if len(w) >3]

    # Print the 30 most common tokens
    for x in bow.most_common(8):
        print (x[0])
    print("############################################")

