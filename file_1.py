# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:38:08 2021

@author: saile
"""
import numpy as np
import onnx
import tensorflow as tf
from onnx import numpy_helper
import tf2onnx
def relu(f,layer):
    arm_relu="arm_relu_q7"
    b_in="buffer2"
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
    f.write(arm_relu+o+b_in+c+conv+n+und+out+und+x+star+conv+n+und+out+und+y+star)
    f.write(conv+n+und+out+und+ch)
    f.write(close+";\n")
def convolution(f,layer):
    arm_convolution="arm_convolve_HWC_q7_fast_nonsquare"
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
    if layer==1:
        f.write(arm_convolution+o+data+c)
    else:
        f.write(arm_convolution+o+b3+c)
    f.write(conv+n+und+IN+und+x+c+conv+n+und+IN+und+y+c+conv+n+und+IN+und+ch+c)
    f.write(weight+und+n+c+conv+n+und+out+und+ch+c+conv+n+und+k+und+x+c+conv+n+und+k+und+y+c)
    f.write(conv+n+und+pad+und+x+c+conv+n+und+pad+und+y+c+conv+n+und+stride+und+x+c)
    f.write(conv+n+und+stride+und+y+c+b+und+n+c)
    f.write(conv+n+und+Bias+und+lshift+c+conv+n+und+out+und+rshift+c+b2+c)
    f.write(conv+n+und+out+und+x+c+conv+n+und+out+und+y+c+misc+c+null)
    f.write(close+";\n")
def max_pool(f,layer):
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
    f.write(arm_max_pool+o+m+n+und+IN+und+y+c+m+n+und+IN+und+x+c)
    f.write(m+n+und+out+und+y+c+m+n+und+out+und+x+c+m+n+und+stride+und+y+c)
    f.write(m+n+und+stride+und+x+c+m+n+und+k+und+y+c+m+n+und+k+und+x+c)
    f.write(m+n+und+pad+und+y+c+m+n+und+pad+und+x+c+m+n+und+act+und+m1+c)
    f.write(m+n+und+act+und+m2+c+m+n+und+ch+und+"in"+c+b2+c+null+c+b3)
    f.write(close+";\n")
def fully_connected(f,layer):
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
    fc="FC"
    b_in="buffer3"
    buff="buffer1"
    n=str(layer)
    f.write(arm_full+o+b_in+c+weight+und+n+c)
    f.write(ip+n+und+IN+und+d+c+ip+n+und+out+und+d+c+ip+n+und+Bias+und+lshift+c)
    f.write(ip+n+und+out+und+rshift+c+b+und+n+c+fc+n+und+out+c+buff)
    f.write(close+";\n")
    
f=open('inference.c','w')
model=tf.keras.models.load_model('0-9-A-Z_selva.h5')
conv=0
fc=0
mp=0
for idx in range(len(model.layers)):
    a=(model.get_layer(index=idx).name)
    if a.find('conv2d')!=-1:
        conv+=1
        f.write('#include '+'<conv'+str(conv)+'.h>'+'\n')
    elif a.find('max_pooling2d')!=-1:
        mp=mp+1
        f.write('#include '+'<max_pooling'+str(mp)+'.h>'+'\n')
    elif a.find('dense')!=-1:
        fc=fc+1
        f.write('#include '+'<FC'+str(fc)+'.h>'+'\n')
#s.append("#include<inference.h>\n")
func="void inference_find(void)\n{\n"
f.write(func)
#load tensorflow model
conv=0
fc=0
mp=0
for idx in range(len(model.layers)):
    a=(model.get_layer(index=idx).name)
    if a.find('conv2d')!=-1:
        print('convolution')
        conv=conv+1
        convolution(f,conv)
        relu(f,conv)
    elif a.find('max_pooling2d')!=-1:
        print('max_pooling')
        mp=mp+1
        max_pool(f,mp)
    elif a.find('dense')!=-1:
        print('fully_connected')
        fc=fc+1
        fully_connected(f,fc)
    else:
        print(a)
f.write("\n}\n")
f.close()
