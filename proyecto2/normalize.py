#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:14:08 2016

@author: ignacio
"""

def normalize(data):
    max_values = data.max(0)
    min_values = data.min(0)   
    norm_data =  (data-min_values)/(max_values-min_values)
    return norm_data,max_values,min_values