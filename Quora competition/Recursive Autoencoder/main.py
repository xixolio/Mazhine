#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:00:26 2017

@author: ignacio
"""
import numpy as np

def create_pairs(words,word_size,max_words):
    n = len(words)
    n_words = np.zeros((n,1))
    pairs = np.zeros((n*max_words,2*word_size))
    cp = 0
    for i in range(n):
        n_words[i] = words[i].shape[0]
        for j in range(n_words[i]-1):
            pairs[cp+j,0:word_size] = words[i][j,:]
            pairs[cp+j,word_size:] = words[i][j+1,:]
        cp += n_words[i,0]-1

    pairs = pairs[0:cp,:]
    return pairs,n_words

def main(words):
    pairs,n_words=create_pairs(words,1,4)
    next_pairs,next_words,new_n_words = word_pairs(model,model2,pairs,words,n_words,1)
    return next_pairs,next_words,new_n_words
