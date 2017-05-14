#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:54:51 2017

@author: ignacio
"""
def algo():
    a = {}
    a['a']=0
    b = ['b','a']
    for i in range(2):
        try:
            for j in range(2):
                print a[b[j]]
        except KeyError:
            print 'gg'
            continue

       