#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:20:15 2016

@author: ignacio
"""

def partition_data(data,output,first_data,last_data,lag):
    return data[first_data:last_data-lag-1,:],data[last_data-lag,:],output[first_data:last_data-lag-1,:],output[last_data-lag,:]