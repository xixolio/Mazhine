#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:22:04 2016

@author: ignacio
"""
from __future__ import absolute_import
from keras import backend as K

def custom_loss(y_true,y_pred):
#    return K.mean(K.square(y_pred-y_true)) + K.mean(K.square((y_pred[0:4]-y_pred[5:]) - (y_true[0:4]-y_pred[5:])))
    return K.mean(K.square(y_pred[4:8]-y_true[4:8])) + K.mean(K.square(y_pred[0:4]-y_pred[4:8] - (y_true[0:4]-y_pred[4:8])))+K.mean(K.square(y_pred[4:8]-y_pred[8:] - (y_true[4:8]-y_pred[8:])))