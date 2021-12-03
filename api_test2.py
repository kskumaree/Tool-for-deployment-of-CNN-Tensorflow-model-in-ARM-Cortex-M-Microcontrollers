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
def nextpow2(n):
    count = 0
    if (n and not(n & (n - 1))):
        return n  
    while( n != 0):
        n >>= 1
        count += 1   
    return 1 << count

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
mp=0
for idx in range(len(model.layers)):
    a=(model.get_layer(index=idx).name)
    if a.find('conv2d')!=-1:
        w=model.layers[idx].get_weights()
        a=w[0]
        b=w[1]
        b=b.reshape(len(b),1)
        w = np.transpose(a, (3, 2, 0, 1))
        sa=nextpow2(math.ceil(np.max(maxval[idx])))/128
        c=w.shape
        p=np.max(np.abs(w))
        if p<1:
            p=1
        else:
            p=nextpow2(math.ceil(p))
        sw=p/128
        sb=sa*sw
        Wq=np.round(w/sw)
        #bq=np.round(b/sb)
        bq=np.round(b/sw)
        left_shift=str(int(math.log(1/sa,2)))
        sa1=nextpow2(math.ceil(np.max(maxvalout[idx])))/128
        scale=sb/sa1   
        right_shift=str(int(math.log(1/scale,2)))
        layer=model.layers[idx]
        in_size=layer.input_shape
        out_size=layer.output_shape
        ksize=model.get_layer(index=idx).get_config()['kernel_size']
        k_x=str(ksize[0])
        k_y=str(ksize[1])
        x_size=in_size[1]
        y_size=in_size[2]
        ch_size=in_size[3]
        out_x=str(out_size[1])
        out_y=str(out_size[2])
        out_ch=str(out_size[3])
        padsize=model.get_layer(index=idx).get_config()['padding']
        Stride=model.get_layer(index = idx).get_config()['strides']
        if padsize=='same':
            pval=int(math.floor(((x_size-1)*Stride[0]+ksize[0]-x_size)/2))
        else:
            pval=0
        pval=str(pval)
        stride_x=str(Stride[0])
        cl=cl+1
        W="W"
        und="_"
        c=","
        o="{"
        close="}"
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
        Bias="BIAS"
        conv="CONV"
        hdef="#define"
        nxt="\n"
        sp="\t"
        con="const"
        dtype="q7_t"
        B="b"
        n=str(cl)
        fname='conv'+n+'.h'
        f=open(fname,'w')
        f.write('#include  '+ "<arm_const_structs.h>"+'\n')
        f.write('#include  '+ "<arm_nnfunctions.h>"+'\n')
        f.write(hdef+" "+conv+n+und+IN+und+y+sp+str(y_size)+nxt)
        f.write(hdef+" "+conv+n+und+IN+und+x+sp+str(x_size)+nxt)
        f.write(hdef+" "+conv+n+und+IN+und+ch+sp+str(ch_size)+nxt)
        f.write(hdef+" "+conv+n+und+k+und+x+sp+k_x+nxt)
        f.write(hdef+" "+conv+n+und+k+und+y+sp+k_y+nxt)
        f.write(hdef+" "+conv+n+und+pad+und+y+sp+pval+nxt)
        f.write(hdef+" "+conv+n+und+pad+und+x+sp+pval+nxt)
        f.write(hdef+" "+conv+n+und+stride+und+y+sp+stride_x+nxt)
        f.write(hdef+" "+conv+n+und+stride+und+x+sp+stride_x+nxt)
        f.write(hdef+" "+conv+n+und+Bias+und+lshift+sp+left_shift+nxt)
        f.write(hdef+" "+conv+n+und+out+und+rshift+sp+right_shift+nxt)
        f.write(hdef+" "+conv+n+und+out+und+y+sp+out_y+nxt)
        f.write(hdef+" "+conv+n+und+out+und+x+sp+out_x+nxt)
        f.write(hdef+" "+conv+n+und+out+und+ch+sp+out_ch+nxt)
        f.write(con+" "+dtype+" "+B+und+n+"["+conv+n+und+out+und+ch+"]"+"="+o)
        for ii in range(len(bq)):
            if(ii==0):
                f.write(str(int(bq[ii][0])))
            else:
                f.write(","+str(int(bq[ii][0])))
        f.write(close+";"+nxt)
        f.write(con+" "+dtype+" "+W+und+n+"["+conv+n+und+IN+und+ch+"*"+conv+n+und+k+und+y+"*"+conv+n+und+k+und+x+"*"+conv+n+und+out+und+ch+"]"+"="+o+nxt)
        p=Wq.shape
        itr=0
        flag=0
        for ii in range(p[0]):
            for jj in range(p[1]):
                for kk in range(p[2]):
                    for ll in range(p[3]):
                        itr+=1
                        if(flag==0):
                            f.write(str(int(Wq[ii,jj,kk,ll])))
                            flag=1
                        else:
                            f.write(","+str(int(Wq[ii,jj,kk,ll])))
                        if (ii==p[0]-1 and jj==p[1]-1 and kk==p[2]-1 and ll==p[3]-1):
                            continue
                        elif itr==100:
                            f.write(c+nxt)
                            flag=0
                            itr=0
        f.write(close+";"+nxt)
        f.close()
    elif a.find('dense')!=-1:
        w=model.layers[idx].get_weights()
        a=w[0]
        b=w[1]
        bf=b.reshape(len(b),1)
        wf=np.transpose(a)
        saf=nextpow2(math.ceil(np.max(maxval[idx])))/128
        p=np.max(np.abs(wf))
        if p<1:
            p=1
        else:
            p=nextpow2(math.ceil(p))
        swf=p/128
        sbf=swf*saf
        #bfq=np.round(bf/sbf)
        bfq=np.round(bf/swf)
        lshiftf=str(int(math.log(1/saf,2)))
        wfq=np.round(wf/swf)
        saf1=nextpow2(math.ceil(np.max(maxvalout[idx])))/128
        scalef=sbf/saf1
        rshiftf=str(int(math.log(1/scalef,2)))
        fc+=1
        n=str(fc)
        layer=model.layers[idx]
        ishape=(layer.input_shape)
        oshape=(layer.output_shape)
        ishape=str(ishape[1])
        oshape=str(oshape[1])
        weight="wf"
        IN="IN"
        ip="IP"
        out="OUT"
        d="DIM"
        B="bf"
        und="_"
        c=","
        o="{"
        close="}"
        Bias="BIAS"
        lshift="LSHIFT"
        rshift="RSHIFT"
        hdef="#define"
        fc="FC"
        con="const"
        dtype="q7_t"
        fname='FC'+n+'.h'
        sp='\t'
        f=open(fname,'w')
        f.write('#include  '+ "<arm_const_structs.h>"+'\n')
        f.write('#include  '+ "<arm_nnfunctions.h>"+'\n')
        f.write(hdef+sp+ip+n+und+IN+und+d+sp+ishape+nxt)
        f.write(hdef+sp+ip+n+und+out+und+d+sp+oshape+nxt)
        f.write(hdef+sp+ip+n+und+Bias+und+lshift+sp+lshiftf+nxt)
        f.write(hdef+sp+ip+n+und+out+und+rshift+sp+rshiftf+nxt)
        f.write(con+' '+dtype+' '+weight+und+n+'['+ip+n+und+out+und+d+'*'+ip+n+und+IN+und+d+']'+'='+o+nxt)
        pp=wfq.shape
        flag=0
        itr=0
        for ii in range(pp[0]):
            for jj in range(pp[1]):
                itr+=1
                if flag==0:
                    f.write(str(int(wfq[ii][jj])))
                    flag=1
                else:
                    f.write(c+str(int(wfq[ii][jj])))
                if (ii==pp[0]-1 and jj==pp[1]-1):
                    continue
                elif (itr==100):
                    flag=0
                    itr=0
                    f.write(c+nxt)
        f.write(close+';'+nxt)
        f.write(con+' '+dtype+' '+B+und+n+'['+ip+n+und+out+und+d+']'+'='+o+nxt)
        for ii in range(len(bfq)):
            if ii==0:
                f.write(str(int(bfq[ii][0])))
            else:
                f.write(c+str(int(bfq[ii][0])))
        f.write(close+';'+nxt)
        f.close()
    elif a.find('max_pooling2d')!=-1:
        mp+=1
        hdef="#define"
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
        Pad="PAD"
        act="ACT"
        m1="min"
        m2="max"
        ch="CHANNEL"
        und="_"
        sp='\t'
        n=str(mp)
        Stride=model.get_layer(index=idx).get_config()['strides']
        pad=model.get_layer(index=idx).get_config()['padding']
        layer=model.layers[idx]
        in_size=layer.input_shape
        out_size=layer.output_shape
        ksize=model.get_layer(index=idx).get_config()['pool_size']
        k_x=str(ksize[0])
        k_y=str(ksize[1])
        x_size=in_size[1]
        y_size=in_size[2]
        ch_size=in_size[3]
        out_x=str(out_size[1])
        out_y=str(out_size[2])
        out_ch=str(out_size[3])
        stride_x=str(Stride[0])
        if pad=='same':
            pval=int(math.floor(((x_size-1)*Stride[0]+ksize[0]-x_size)/2))
        else:
            pval=0
        pval=str(pval)
        fname='max_pooling'+n+'.h'
        f=open(fname,'w')
        f.write('#include  '+ "<arm_const_structs.h>"+'\n')
        f.write('#include  '+ "<arm_nnfunctions.h>"+'\n')
        f.write(hdef+sp+m+n+und+IN+und+x+sp+str(x_size)+nxt)
        f.write(hdef+sp+m+n+und+IN+und+y+sp+str(y_size)+nxt)
        f.write(hdef+sp+m+n+und+ch+und+'in'+sp+str(ch_size)+nxt)
        f.write(hdef+sp+m+n+und+out+und+x+sp+out_x+nxt)
        f.write(hdef+sp+m+n+und+out+und+y+sp+out_y+nxt)
        f.write(hdef+sp+m+n+und+k+und+x+sp+k_x+nxt)
        f.write(hdef+sp+m+n+und+k+und+y+sp+k_y+nxt)
        f.write(hdef+sp+m+n+und+stride+und+x+sp+stride_x+nxt)
        f.write(hdef+sp+m+n+und+stride+und+y+sp+stride_x+nxt)
        f.write(hdef+sp+m+n+und+Pad+und+x+sp+pval+nxt)
        f.write(hdef+sp+m+n+und+Pad+und+y+sp+pval+nxt)
        f.write(hdef+sp+m+n+und+act+und+m1+sp+str(0)+nxt)
        f.write(hdef+sp+m+n+und+act+und+m2+sp+str(127)+nxt)
        f.close()
