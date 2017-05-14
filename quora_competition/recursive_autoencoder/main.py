#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:00:26 2017

@author: ignacio
"""
import numpy as np

def create_pairs(words,word_size,max_pairs):
    n = len(words)
    n_words = np.zeros((n,1))
    n_pairs = np.zeros((n,1))
    pairs = np.zeros((max_pairs,2*word_size))
    cp = 0
    for i in range(n):
        n_words[i,0] = words[i].shape[0]
        for j in range(int(n_words[i,0]-1)):
            pairs[cp+j,0:word_size] = words[i][j,:]
            pairs[cp+j,word_size:] = words[i][j+1,:]
        cp += int(n_words[i,0]-1)

    pairs = pairs[0:cp,:]
    n_pairs = n_words - 1
    return pairs,n_words,n_pairs

if __name__ == "__main__":   
    a = np.array([[1,2],[3,4],[5,6]]).reshape(-1,2)
    b = np.array([[5,6],[7,8],[9,10]]).reshape(-1,2)
    words = []
    words.append(a)
    words.append(b)
    
    word_size = words[0].shape[1]
    pairs,n_words,n_pairs=create_pairs(words,word_size,4)
    temp_pairs =pairs
    temp_words = words
    temp_n_words = n_words

    
    encoder,decoder,autoencoder = Autoencoder(4,[2])

    for i in range(10):
        autoencoder.fit(pairs,pairs,epochs=100)
        temp_pairs,temp_words,temp_n_words = word_pairs(encoder,decoder,
                                                        temp_pairs,temp_words,
                                                        temp_n_words,n_pairs,word_size)
    
    temp_pairs =pairs
    temp_words = words
    temp_n_words = n_words
    for i in range(10):
        temp_pairs,temp_words,temp_n_words = word_pairs(encoder,decoder,
                                                        temp_pairs,temp_words,
                                                        temp_n_words,n_pairs,word_size) 
    print('succes')

