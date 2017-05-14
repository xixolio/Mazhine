#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import csv
import re
import sys
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
reload(sys)
sys.setdefaultencoding('utf-8')

MAX_NB_WORDS = 200000

def loadGloveModel():
    print("Loading Glove Model")
    f = open("/user/i/iaraya/glove.42B.300d.txt",'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

if __name__ == "__main__":   

    
    texts_1 = [] 
    texts_2 = []
    labels = []

    with codecs.open('/user/i/iaraya/files/Mazhine/quora_competition/train.csv', encoding='utf-8') as f:
	    reader = csv.reader(f, delimiter=',')
	    header = next(reader)
	    i=0
	    for values in reader:
	        i+=1
	        texts_1.append(text_to_wordlist(values[3]))
	        texts_2.append(text_to_wordlist(values[4]))
	        labels.append(int(values[5]))
    print('Found %s texts in train.csv' % len(texts_1))

    sequences_1 = []
    sequences_2 = []

    for i in range(len(texts_1)):
	    sequences_1.append(text_to_word_sequence(texts_1[i]))
	    sequences_2.append(text_to_word_sequence(texts_2[i]))	    
    embedded_words = loadGloveModel()            
    e_sentences_1 =[]
    e_sentences_2 =[]

    for i in range(len(texts_1)):
        ns1 = len(sequences_1)
        ns2 = len(sequences_2)
        s1 = np.array((ns1,300))
        s2 = np.array((ns2,300))
        print(sequences_1[i][j])
        print(s1.shape) 
        for j in ns1:
            s1[j,:]=np.asarray(embedded_words[sequences_1[i][j]])
        for i in ns2:
            s2[j,:]=np.asarray(embedded_words[sequences_2[i][j]])
        e_sentences_1.append(s1)
        e_sentences_2.append(s2)
        

    
     

