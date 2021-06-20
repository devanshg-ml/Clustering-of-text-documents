from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from autocorrect import spell
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib


def get_data(fname):
    content = []
    with open(fname, 'r') as f:
        for line in f:
            tweet = line.split("|")[2]
            content.append(tweet)
    return content

def tokenize_(content):
    tw = []
    for i in content:
        clean = extract_words(i)
        tw.extend(clean)
    return tw


def extract_words_and_stem(sentence):
    words = re.sub("[^A-Za-z]", " ",  sentence.lower())
    tokenized_words = word_tokenize(words)
    stop_words =['cnn', 'com', 'http', 'cnn']
    for word in list(tokenized_words):
        if word in stopwords.words('english'):
            tokenized_words.remove(word)
        if word in stop_words:
            tokenized_words.remove(word)
    stemmer = SnowballStemmer("english")
    for i in range(len(tokenized_words)):
        tokenized_words[i] = stemmer.stem(tokenized_words[i])
    tokenized_words[i] = stemmer.stem(spell(tokenized_words[i]))
    return tokenized_words

def extract_words(sentence):
    words = re.sub("[^A-Za-z]", " ",  sentence.lower())
    tokenized_words = word_tokenize(words)
    stop_words =['cnn', 'com', 'http', 'cnn']
    for word in list(tokenized_words):
        if word in stopwords.words('english'):
            tokenized_words.remove(word)
        if word in stop_words:
            tokenized_words.remove(word)

def generate_vocab(content):
    vocab = []
    for sentence in content:
        w = extract_words(sentence)
        vocab.extend(w)
    vocab = sorted(list(vocab))
    return vocab

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

def main():
    fname = 'cnnhealth.txt'
    content = get_data(fname)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True,tokenizer=extract_words, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)
    terms = tfidf_vectorizer.get_feature_names()
    print(tfidf_matrix.shape)
    dist = 1 - cosine_similarity(tfidf_matrix)
    print(dist)

    print("fitting euclidean dbscan")
    dbscan = DBSCAN(min_samples=5, metric='euclidean',
                    algorithm='auto', leaf_size=30,
                    n_jobs=4)
    dbscan.fit(tfidf_matrix)

    print("fitting minkowski dbscan")
    dbscan1 = DBSCAN(min_samples=5, metric='manhattan',
                    algorithm='auto', leaf_size=30,
                    n_jobs=4)
    dbscan1.fit(tfidf_matrix)

    clusters = dbscan.labels_.tolist()
    clusters1 = dbscan1.labels_.tolist()
    labels = dbscan.labels_
    labels1 = dbscan1.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters_1 = len(set(labels1)) - (1 if -1 in labels1 else 0)

    joblib.dump(dbscan,  'doc_cluster.pkl')
    joblib.dump(dbscan1,  'doc_cluster1.pkl')
    #plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
    #         'Expectation-maximization')

if __name__ == "__main__":
    main()
