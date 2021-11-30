# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:17:44 2021

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
                result[i] += weight[i][k] * x[k][j]
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
        for h in range(n_c):
            for c in range(n_h):
                Z.append(inp[w][c][h])
    Z=np.array(Z)
    Z=Z.reshape(len(Z),1)
    return Z

model=tf.keras.models.load_model('0-9-A-Z_selva.h5')
classes=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','H','K','N','R','U','Y','E','P','S','X','F','L','T','Z','G','O','I','J','M','Q','V','W']
imagelist=['102.png','146.png','149.png','150.png','157.png'] 
inp=[]
val=0
for idx in range(len(model.layers)):
    val+=1
maxval=np.zeros((val,len(imagelist)))
maxvalout=np.zeros((val,len(imagelist)))
outmax=np.zeros(len(imagelist))
itr=-1
for img1 in imagelist:
    itr+=1
    img = Image.open(img1)
    p = asarray(img)
    [x,y]=p.shape
    p=p.reshape(1,x,y)
    inp=p/255
    for idx in range(len(model.layers)):
        a=(model.get_layer(index=idx).name)
        if a.find('conv2d')!=-1:
            maxval[idx][itr]=np.max(np.abs(inp))
            w=model.layers[idx].get_weights()
            a=w[0]
            b=w[1]
            b=b.reshape(1,1,len(b))
            stride=model.get_layer(index = idx).get_config()['strides']
            pad=model.get_layer(index=idx).get_config()['padding']
            fsize=model.get_layer(index=idx).get_config()['kernel_size']
            if(pad=='valid'):
                pval=0
            elif pad=='same':
                c=inp.shape
                if len(c)==2:
                    pval=math.floor(((c[0]-1)*stride[0]+fsize[0]-c[0])/2)
                elif len(c)==3:
                    pval=math.floor(((c[1]-1)*stride[0]+fsize[0]-c[1])/2)
            aa = np.transpose(a, (3, 2, 0, 1))
            inp1=conv2d(inp,aa,b,pval,stride[0])
            inp=relu(inp1)
            maxvalout[idx][itr]=np.max(np.abs(inp))
        elif a.find('max_pooling2d')!=-1:
            stride=model.get_layer(index=idx).get_config()['strides']
            pool=model.get_layer(index=idx).get_config()['pool_size']
            pad=model.get_layer(index=idx).get_config()['padding']
            if pad=='same':
                c=inp.shape
                if len(c)==2:
                    pval=math.floor(((c[0]-1)*stride[0]+fsize[0]-c[0])/2)
                    inp=zero_pad(inp,pval)
                elif len(c)==3:
                    pval=math.floor(((c[1]-1)*stride[0]+fsize[0]-c[1])/2)
                    inp=zero_pad(inp,pval)
            inp=max_pooling(inp,pool,stride)
        elif a.find('dense')!=-1:
            w=model.layers[idx].get_weights()
            a=w[0]
            b=w[1]
            b=b.reshape(len(b),1)
            a=np.transpose(a)
            maxval[idx][itr]=np.max(np.abs(inp))
            inp=fully_connected(b,a,inp)
            maxvalout[idx][itr]=np.max(np.abs(inp))
        elif a.find('flatten')!=-1:
            inp = np.transpose(inp, (1,0,2))
            inp=flatten(inp)
    m = np.max(inp);
    e = np.exp(inp-m)
    dist = e /np.sum(e)
    score=np.max(dist)
    i = np.where(dist == dist.max())
    digit=classes[i[0][0]]
    print(digit)
cl=0
fc=0
for idx in range(len(model.layers)):
    a=(model.get_layer(index=idx).name)
    if a.find('conv2d')!=-1:
        w=model.layers[idx].get_weights()
        a=w[0]
        b=w[1]
        b=b.reshape(len(b),1)
        w = np.transpose(a, (3, 2, 0, 1))
        sa=np.max(maxval[idx])/255
        c=w.shape
        Wq=np.zeros(c)
        bq=np.zeros((c[0],1))
        sw=np.zeros((c[0],1))
        sb=np.zeros((c[0],1))
        for i in range(c[0]):
            sw[i]=np.max(np.abs(w[i,:,:,:]))/127
            sb[i]=sa*sw[i]
            Wq[i,:,:,:]=np.round(w[i,:,:,:]/sw[i])
            bq[i]=np.round(b[i]/sb[i])        
    elif a.find('dense')!=-1:
            w=model.layers[idx].get_weights()
            a=w[0]
            b=w[1]
            bf=b.reshape(len(b),1)
            wf=np.transpose(a)
            saf=np.max(maxval[idx])/255
            swf=np.max(np.abs(wf))/127
            sbf=swf*saf
            bfq=np.round(bf/sbf)
            wfq=np.round(wf/swf)
