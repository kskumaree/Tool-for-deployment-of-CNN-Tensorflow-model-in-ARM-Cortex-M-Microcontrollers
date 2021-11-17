# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:55:43 2021

@author: saile
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import asarray
import scipy.io
from sklearn.metrics import mean_squared_error
import math
def fully_connected(bias,weight,x):
    l1=weight.shape
    l2=x.shape
    result=np.zeros([l1[0],l2[1]])
    for i in range(len(weight)):
    # iterating by column by B
        for j in range(len(x[0])):
            # iterating by rows of B
            for k in range(len(x)):
                result[i][j] += weight[i][k] * x[k][j]
    return result+bias
def relu(x):
    r=x.shape 
    x1=np.zeros(r)
    x2=np.maximum(x,x1)
    return x2
def max_pooling(conv_out_relu,f_pool,S_pool):
    [n_H_prev, n_W_prev, n_C_prev] = conv_out_relu.shape
    f1 = f_pool[0]
    f2 = f_pool[1]
    stride = S_pool[0]
    n_H = math.floor(1 + (n_H_prev - f1) / stride)
    n_W = math.floor(1 + (n_W_prev - f2) / stride)
    n_C = n_C_prev
    max_pool=np.zeros([n_C,n_H,n_W])
    for h in range(0,n_H):                     
        vert_start = (h-1)*stride+1
        vert_end = vert_start+f1-1
        for w in range(0,n_W):                
            horiz_start = (w-1)*stride+1
            horiz_end = horiz_start+f2-1
            for c in range(0,n_C):
                a_prev_slice = conv_out_relu[vert_start:vert_end+1,horiz_start:horiz_end+1,c]
                max_pool[c,h, w]= np.max(a_prev_slice)
    return max_pool
def zero_pad(x,pad):
    p=x.shape
    if len(p)==3:
        l=np.pad(x,((pad,pad),(pad,pad),(0,0)),mode='constant')
    elif len(p)==2:
        l=np.pad(x,[pad,pad],mode='constant')
    return l
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    b = np.squeeze(b)
    Z = Z + b
    return Z
def conv2d(A_prev,W,b,pad,S):
    pad=pad[0]
    stride=S[0]
    [n_H_prev, n_W_prev, n_C_prev] = A_prev.shape
    [f, f, n_C_prev, n_C] = W.shape
    n_H = math.floor(((n_H_prev+(2*pad)-f)/stride)+1)
    n_W = math.floor(((n_W_prev+(2*pad)-f)/stride)+1)
    Z = np.zeros((n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev,pad)
    for h in range(n_H):           
        vert_start = h*stride;
        vert_end = vert_start+f
        for w in range(n_W):
            horiz_start = w*stride
            horiz_end = horiz_start+f
            for c in range(n_C):
                a_slice_prev = A_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                weights = W[:,:,:,c]
                biases = b[:,:,c]
                Z[h, w, c] = conv_single_step(a_slice_prev,weights,biases)
    return Z
r_out=scipy.io.loadmat('relu_out.mat')
r2=r_out['conv3_out_relu_int8']
f=np.array([1,5])
s=np.array([1,1])
mp1=max_pooling(r2,f,s)
p_out=scipy.io.loadmat('pool_out.mat')
p1=p_out['max_pool3_int8']
c_out=scipy.io.loadmat('conv_out.mat')
c2=c_out['Z']
c_in=scipy.io.loadmat('conv_in.mat')
c1=c_in['A_prev']
b_out=scipy.io.loadmat('bias.mat')
b1=b_out['b']
w_out=scipy.io.loadmat('weight.mat')
w1=w_out['W']
pad=np.array([1,1])
stride=np.array([1,1])
c3=conv2d(c1,w1,b1,pad,stride)