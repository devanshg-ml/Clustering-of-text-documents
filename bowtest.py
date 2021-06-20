import numpy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from autocorrect import spell
nltk.download('stopwords')
nltk.download('punkt')
numpy.set_printoptions(threshold=numpy.inf)

def generate_vocab(content):
    vocab = []
    for sentence in content:
        w = extract_words(sentence)
        vocab.extend(w)
    vocab = sorted(list(vocab))
    return vocab

def extract_words(sentence):
    words = re.sub("[^A-Za-z]", " ",  sentence.lower())
    tokenized_words = word_tokenize(words)
    for word in tokenized_words:
        if word in stopwords.words('english'):
            tokenized_words.remove(word)
    stemmer = PorterStemmer()
    for i in range(len(tokenized_words)):
        tokenized_words[i] = stemmer.stem(tokenized_words[i])
    tokenized_words[i] = stemmer.stem(spell(tokenized_words[i]))
    return tokenized_words


def generate_bow(content):
    data = []
    vocab = generate_vocab(content)
    f = open('bowmatrix.txt', 'w')
    for sentence in content:
        words = extract_words(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                if word==w:
                    bag_vector[i] += 1
        dobj = {"sentence": sentence,
                "array": numpy.array(bag_vector)}
        data.append(dobj.copy())
        print("{}\n{}\n".format(sentence,numpy.array(bag_vector)[1:10]))
        f.write("{}\n{}\n".format(sentence,numpy.array(bag_vector)))
    f.close()
    return data


def get_data(fname):
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    return content

def main():
    fname = 'cnnhealth.txt'
    content = get_data(fname)
    print(content[1])
    print(len(content))
    data =generate_bow(content)
    print(len(data))

if __name__ == "__main__":
    main()
