# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:22:46 2021
 
@author: sailesh
"""

import numpy as np
import math
flag=True
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
def nextpow2(n):
    count = 0
    if (n and not(n & (n - 1))):
        return n  
    while( n != 0):
        n >>= 1
        count += 1   
    return 1 << count
def RELU(f,layer,p):
    arm_relu="arm_relu_q7"
    b2="buffer2"
    b3="buffer3"
    conv="CONV"
    n=str(layer)
    out="OUT"
    und="_"
    c=","
    x="x"
    y="y"
    o="("
    star="*"
    close=")"
    ch="CH"
    d="DIM"
    ip="IP"
    if p=='conv':
        if flag==True:
            f.write(arm_relu+o+b3+c)
        elif flag==False:
            f.write(arm_relu+o+b2+c)
        f.write(conv+n+und+out+und+x+star+conv+n+und+out+und+y+star)
        f.write(conv+n+und+out+und+ch)
    else:
        if flag==True:
            f.write(arm_relu+o+b3)
        else:
            f.write(arm_relu+o+b2)
        f.write(c+ip+n+und+out+und+d)
    f.write(close+";\n")
def CONVOLUTION(f,layer,itr):
    arm_convolution="arm_convolve_HWC_q7_basic_nonsquare"
    data="data"
    weight="W"
    und="_"
    c=","
    o="("
    close=")"
    x="x"
    y="y"
    IN="IN"
    out="OUT"
    ch="CH"
    stride="STRIDE"
    lshift="LSHIFT"
    rshift="RSHIFT"
    null="NULL"
    pad="PAD"
    k="KER"
    b2="buffer2"
    b3="buffer3"
    Bias="BIAS"
    conv="CONV"
    b="b"
    misc="(q15_t*)col_buffer1"
    n=str(layer)
    global flag
    if itr==1:
        f.write(arm_convolution+o+data+c)
    elif flag==True:
        f.write(arm_convolution+o+b3+c)
    elif flag==False:
        f.write(arm_convolution+o+b2+c)
    f.write(conv+n+und+IN+und+x+c+conv+n+und+IN+und+y+c+conv+n+und+IN+und+ch+c)
    f.write(weight+und+n+c+conv+n+und+out+und+ch+c+conv+n+und+k+und+x+c+conv+n+und+k+und+y+c)
    f.write(conv+n+und+pad+und+x+c+conv+n+und+pad+und+y+c+conv+n+und+stride+und+x+c)
    f.write(conv+n+und+stride+und+y+c+b+und+n+c)
    f.write(conv+n+und+Bias+und+lshift+c+conv+n+und+out+und+rshift+c)
    if flag==True:
        f.write(b2+c)
    elif flag==False:
        f.write(b3+c)
    flag=not(flag)
    f.write(conv+n+und+out+und+x+c+conv+n+und+out+und+y+c+misc+c+null)
    f.write(close+";\n")
def MAX_POOL(f,layer):
    arm_max_pool="arm_max_pool_s8_opt"
    und="_"
    c=","
    o="("
    close=")"
    stride="STRIDE"
    m="MAX"
    k="KERNEL"
    x="x"
    y="y"
    IN="IN"
    out="OUT"
    null="NULL"
    b2="buffer2"
    b3="buffer3"
    pad="PAD"
    act="ACT"
    m1="min"
    m2="max"
    ch="CHANNEL"
    n=str(layer)
    global flag
    f.write(arm_max_pool+o+m+n+und+IN+und+y+c+m+n+und+IN+und+x+c)
    f.write(m+n+und+out+und+y+c+m+n+und+out+und+x+c+m+n+und+stride+und+y+c)
    f.write(m+n+und+stride+und+x+c+m+n+und+k+und+y+c+m+n+und+k+und+x+c)
    f.write(m+n+und+pad+und+y+c+m+n+und+pad+und+x+c+m+n+und+act+und+m1+c)
    f.write(m+n+und+act+und+m2+c+m+n+und+ch+und+"in"+c)
    if flag==True:
        f.write(b3+c+null+c+b2)
    elif flag==False:
        f.write(b2+c+null+c+b3)
    flag=not(flag)
    f.write(close+";\n")
def FULLY_CONNECTED(f,layer,ii,val):
    arm_full="arm_fully_connected_q7"
    weight="wf"
    IN="IN"
    ip="IP"
    out="OUT"
    d="DIM"
    b="bf"
    und="_"
    c=","
    o="("
    close=")"
    Bias="BIAS"
    lshift="LSHIFT"
    rshift="RSHIFT"
    buff="buffer1"
    b2="buffer2"
    b3="buffer3"
    n=str(layer)
    global flag
    if ii==1:
        f.write(arm_full+o+"data"+c+weight+und+n+c)
    elif flag==True:
        f.write(arm_full+o+b3+c+weight+und+n+c)
    else:
        f.write(arm_full+o+b2+c+weight+und+n+c)
    f.write(ip+n+und+IN+und+d+c+ip+n+und+out+und+d+c+ip+n+und+Bias+und+lshift+c)
    f.write(ip+n+und+out+und+rshift+c+b+und+n+c)
    #global val
    if ii==val:
        f.write("FC_OUT,"+buff)
    elif flag==True:
        f.write(b2+c+buff)
    else:
        f.write(b3+c+buff)
    flag=not(flag)
    f.write(close+";\n")
def SOFTMAX(f,layer):
    f.write('arm_softmax_q7(FC_OUT,'+'IP'+str(layer)+'_OUT_DIM,SOFT_OUT);\n')
