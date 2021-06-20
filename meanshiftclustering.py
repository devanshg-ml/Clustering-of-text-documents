from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from math import*
from decimal import Decimal
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import re
import scikitplot as skplt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from autocorrect import spell
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
import os
import glob
stemmer = SnowballStemmer("english")
stop_words = nltk.corpus.stopwords.words('english')

def get_data(fname):
    content = []
    with open(fname, 'r') as f:
        for line in f:
            tweet = line.split("|")[2]
            content.append(tweet)
    return content
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

def extract_words_stem(sentence):
    sentence = re.sub(r"http\S+", "", sentence)
    words = re.sub("[^A-Za-z]", " ",  sentence.lower())



    tokenized_words = word_tokenize(words)
    stop_words = stopwords.words('english')
    stop_words_extra =['cnn', 'com', 'http', 'rt','stefaniei','video',
                            'new','d','workload','say', 'good','health',
                            'u','cnnhealth','want', 'www', 'html',
                            'upwav', 'twimg', 'amp', 'index', 'cnnallergi',
                            'cnnliving','drashmancnn','https']
    stop_words.extend(stop_words_extra)
    for word in list(tokenized_words):
        if word in stop_words:
            if word in tokenized_words:
                tokenized_words.remove(word)
        if len(word) < 3:
            if word in tokenized_words:
                tokenized_words.remove(word)
    lemmatizer = WordNetLemmatizer()
    for i in range(len(tokenized_words)):
        tokenized_words[i] = lemmatizer.lemmatize(tokenized_words[i])
    tokenized_words[i] = lemmatizer.lemmatize(spell(tokenized_words[i]))
    return tokenized_words


def generate_vocab(content):
    vocab = []
    for sentence in content:
        w = extract_words(sentence)
        vocab.extend(w)
    vocab = sorted(list(vocab))
    return vocab


def tfidf_vectorizer_(content):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',tokenizer=extract_words_stem,
                                            max_df=0.95, max_features=200000,
                                            min_df=0.05, use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)
    terms = tfidf_vectorizer.get_feature_names()
    return terms, tfidf_matrix, tfidf_vectorizer

def print_cohesion_measures(dmatrix, labels, metric):
    print("Silhouette Score: %0.5f" % silhouette_score(X=dmatrix,
                labels=labels, sample_size=1000, metric=metric))



def main():
    all_files_txt = glob.glob('cnnhealth.txt')
    tweet_df = pd.concat((pd.read_csv(f,names=['id','time','tweet'], delimiter='|',encoding = "ISO-8859-1").assign(news_agency=os.path.basename(f).replace('.txt','')) for f in all_files_txt), ignore_index=True)
    tweet_df.drop('id',axis=1,inplace=True)
    tweets = [re.sub(r"http\S+", "", tweet) for tweet in tweet_df['tweet'].tolist()]
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
    fname = 'cnnhealth.txt'
    #data = get_data(fname)
    print('Vectorising data, will take a while')
    #terms, tfidf_matrix, tfidf_vectorizer = tfidf_vectorizer_(data)
    dist = cosine_distances(tfidf_matrix)
    dis2 = euclidean_distances(tfidf_matrix)
    #features =tfidf_matrix.todense()
    # write features matrix to file?
    # if Yes, then make features_matrix equal to True
    # else, features_matrix equal to False
    features_matrix = False
    if features_matrix is True:
        np.savetxt('features_matrix.txt',features)

    vocab_dict = tfidf_vectorizer.vocabulary_
    #d_euclidean_matrix = euclidean_distances(features)
    #d_cosine_matrix = cosine_distances(features)

    # Write distance matrices to file?
    # if Yes then make distance_matrices == True
    # Else distance_matrices = False
    distance_matrices = False
    if distance_matrices is True:
        np.savetxt('distance_euclidean_matrix.txt',d_euclidean_matrix)
        np.savetxt('distance_cosine_matrix.txt',d_cosine_matrix)


    print('fitting MeanShift cosine distance matrix')
    #bandwidth = estimate_bandwidth(d_cosine_matrix, quantile=0.2, n_samples=200)
    bandwidth = estimate_bandwidth(dis2, quantile=0.5, n_samples=200)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=4)
    ms = ms.fit(dis2)
    joblib.dump(ms,"ms_eud.pkl")
    clusters = ms.labels_.tolist()
    skplt.metrics.plot_silhouette(dis2, clusters)
    plt.show()
    plt.savefig('euclidean_average.png', dpi=200) #save figure
    cleaned_tweets = [re.sub(r"http\S+", "", tweet) for tweet in tweet_df['tweet'].tolist()]
    cluster_tweet = pd.DataFrame({"Tweet": cleaned_tweets, "Cluster_Num": clusters})

    #stop_words = stopwords.words('english')
    for word in ['rt','video','new','d','workload','say','RT','good','u','http','cnn','cnnhealth','want','amp','drsanjaygupta','know','way','year','official','cdcgov','minute','help']:
        stop_words.append(word)
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
    for x in bow.most_common(8):
        print (x[0])
    print("############################################")


    bandwidth = estimate_bandwidth(dis, quantile=0.5, n_samples=200)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=4)
    ms = ms.fit(d_euclidean_matrix)
    joblib.dump(ms,  'ms_cosine.pkl')
    #ms = joblib.load('ms_cosine.pkl')
    clusters = ms.labels_.tolist()
    skplt.metrics.plot_silhouette(dis, clusters)
    plt.show()
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
    for x in bow.most_common(8):
        print (x[0])
    print("############################################")
    """
    tweets = {'tweets': data, 'cluster':clusters}
    frame = pd.DataFrame(tweets, index=[clusters], columns=['tweets', 'cluster'])
    order_centroids = ms.cluster_centers_.argsort()[:, ::-1]
    def centeroid(i):
        m = 0
        print('cluster {}'.format(i))
        for ind in order_centroids[i,:]:
            if m > 7:
                print()
                return
            if ind in list(vocab_dict.values()):
                print(list(vocab_dict.keys())[list(vocab_dict.values()).index(ind)])
                m = m + 1

    print('Printing centroids for Cosine distances')
    for i in range(ms.cluster_centers_.shape[0]):
        centeroid(i)
    print('Printing Cohesian Measure: Silhouette score')
    print_cohesion_measures(d_cosine_matrix, clusters, "cosine")
    skplt.metrics.plot_silhouette(X=d_euclidean_matrix, cluster_labels=ms.labels_,
                                    title='Silhouette Analysis', metric='euclidean')
    plt.show()


    print()
    print('fitting MeanShift Euclidean distance matrix')

    bandwidth = estimate_bandwidth(d_euclidean_matrix, quantile=0.2, n_samples=200)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=1)
    ms.fit(d_euclidean_matrix)
    joblib.dump(ms,  'ms_cluster_euclidean.pkl')
    ms = joblib.load('ms_cluster_cosine.pkl')
    clusters = ms.labels_.tolist()
    tweets = {'tweets': data, 'cluster':clusters}
    frame = pd.DataFrame(tweets, index=[clusters], columns=['tweets', 'cluster'])
    order_centroids = ms.cluster_centers_.argsort()[:, ::-1]
    print('Printing centroids for Euclidean distances')
    for i in range(ms.cluster_centers_.shape[0]):
        centeroid(i)
    print('Printing Euclidean Measure: Silhouette score')
    print_cohesion_measures(d_cosine_matrix, clusters, "euclidean")
    #skplt.metrics.plot_silhouette(X=d_cosine_matrix, cluster_labels=ms.labels_,
                                    title='Silhouette Analysis', metric='cosine')
    plt.show()
    """
if __name__ == "__main__":
    main()
