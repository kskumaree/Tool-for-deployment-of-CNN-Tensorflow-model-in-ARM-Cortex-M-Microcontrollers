# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:55:43 2021

@author: saile
"""

import tensorflow as tf
import numpy as np
import math
from PIL import Image
from numpy import asarray
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
    [n_C_prev,n_H_prev, n_W_prev] = conv_out_relu.shape
    f1 = f_pool[0]
    f2 = f_pool[1]
    stride = S_pool[0]
    n_H = math.floor(1 + (n_H_prev - f1) / stride)
    n_W = math.floor(1 + (n_W_prev - f2) / stride)
    n_C = n_C_prev
    max_pool=np.zeros([n_C,n_H,n_W])
    for h in range(0,n_H):                     
        vert_start = (h)*stride
        vert_end = vert_start+f1
        for w in range(0,n_W):                
            horiz_start = (w)*stride
            horiz_end = horiz_start+f2
            for c in range(0,n_C):
                a_prev_slice = conv_out_relu[c,vert_start:vert_end,horiz_start:horiz_end]
                max_pool[c,h, w]= np.max(a_prev_slice)
    return max_pool
def zero_pad(x,pad):
    p=x.shape
    if len(p)==3:
        l=np.pad(x,((0,0),(pad,pad),(pad,pad)),mode='constant')
    elif len(p)==2:
        l=np.pad(x,[pad,pad],mode='constant')
    elif len(p)==4:
        l=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    return l
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    b = np.squeeze(b)
    Z = Z + b
    return Z
def conv2d(A_prev,W,b,pad,S):
    stride=S
    [n_C_prev, n_H_prev, n_W_prev] = A_prev.shape
    [n_C,n_C_prev,f, f] = W.shape
    n_H = math.floor(((n_H_prev+(2*pad)-f)/stride)+1)
    n_W = math.floor(((n_W_prev+(2*pad)-f)/stride)+1)
    Z = np.zeros((n_C,n_H, n_W))
    A_prev_pad = zero_pad(A_prev,pad)
    for h in range(n_H):           
        vert_start = h*stride;
        vert_end = vert_start+f
        for w in range(n_W):
            horiz_start = w*stride
            horiz_end = horiz_start+f
            for c in range(n_C):
                a_slice_prev = A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end]
                weights = W[c,:,:,:]
                biases = b[:,:,c]
                Z[c,h, w] = conv_single_step(a_slice_prev,weights,biases)
    return Z
def flatten(inp):
    [n_c,n_h,n_w]=inp.shape
    Z=[]
    for w in range(n_w):
        for h in range(n_h):
            for c in range(n_c):
                Z.append(inp[c,h,w])
    Z=np.array(Z)
    Z=Z.reshape(len(Z),1)
    return Z

model=tf.keras.models.load_model('0-9-A-Z_selva.h5')    
w=model.layers[0].get_weights()
a=w[0]
b=w[1]
aa = np.transpose(a, (3, 2, 0, 1))
img = Image.open('102.png')
p = asarray(img)
[x,y]=p.shape
p=p.reshape(1,x,y)
inp=p/255
pad=1
stride=1
b=b.reshape(1,1,len(b))
conv1_out=conv2d(inp,aa,b,pad,stride)
conv1_out_relu=relu(conv1_out)
max_pool1=max_pooling(conv1_out_relu,(2,2),(2,2))
w=model.layers[2].get_weights()
a=w[0]
b=w[1]
b=b.reshape(1,1,len(b))
aa = np.transpose(a, (3, 2, 0, 1))
conv2_out=conv2d(max_pool1,aa,b,pad,stride)
conv2_out_relu=relu(conv2_out)
max_pool2=max_pooling(conv2_out_relu,(2,2),(2,2))
X=flatten(max_pool2)
w=model.layers[6].get_weights()
a=w[0]
b=w[1]
b=b.reshape(len(b),1)
a=np.transpose(a)
pred=fully_connected(b,a,X)
 
