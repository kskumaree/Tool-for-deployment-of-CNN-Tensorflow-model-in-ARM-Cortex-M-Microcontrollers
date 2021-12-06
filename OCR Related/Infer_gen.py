# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:38:08 2021

@author: saile
"""
import tensorflow as tf
flag=True
val=0
def relu(f,layer,p):
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
def convolution(f,layer,itr):
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
def fully_connected(f,layer,ii):
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
    
f=open('inference.c','w')
model=tf.keras.models.load_model('0-9-A-Z_selva.h5')
#model=tf.keras.models.load_model('sine.hdf5')
conv=0
fc=0
mp=0
f.write("#include <arm_const_structs.h>\n")
f.write("#include <arm_nnfunctions.h\n")
for idx1 in range(len(model.layers)):
    a=(model.get_layer(index=idx1).name)
    if a.find('conv2d')!=-1:
        conv+=1
        val+=1
        f.write('#include '+'<conv'+str(conv)+'.h>'+'\n')
    elif a.find('max_pooling2d')!=-1:
        mp=mp+1
        val+=1
        f.write('#include '+'<max_pooling'+str(mp)+'.h>'+'\n')
    elif a.find('dense')!=-1:
        fc=fc+1
        val+=1
        f.write('#include '+'<FC'+str(fc)+'.h>'+'\n')
func="void inference_find(void)\n{\n"
f.write(func)
#load tensorflow model
conv=0
fc=0
mp=0
f.write("//data variable is input 1D array, dtype=q7_t\n")
f.write("//buffer2,buffer3 and col_buffer1 are 1D array, dtype=q7_t\n")
f.write("//FC_OUT, q7_t, 1D array, is output of final fully connected layer and size should be defined accordingly\n")
f.write("//buffer2 and 3 should be maximum possible size of the available input and output layers\n")
f.write("//col_buffer1 should be atleast the size of max(2*ch_im_in*dim_kernel*dim_kernel)\n")
f.write("//buffer1 size should be atleast the max input length of fully connected layer, dtype=q15_t, 1D array\n")
itr=0
for idx in range(len(model.layers)):
    a=(model.get_layer(index=idx).name)
    if a.find('conv2d')!=-1:
        print('convolution')
        conv=conv+1
        itr+=1
        convolution(f,conv,itr)
        o=(model.get_layer(index=idx).get_config()['activation'])
        if(o=='relu'):
            relu(f,conv,'conv')
    elif a.find('max_pooling2d')!=-1:
        print('max_pooling')
        mp=mp+1
        itr+=1
        max_pool(f,mp)
    elif a.find('dense')!=-1:
        print('fully_connected')
        fc=fc+1
        itr+=1
        fully_connected(f,fc,itr)
        o=(model.get_layer(index=idx).get_config()['activation'])
        if(o=='relu'):
            relu(f,fc,'fc')
f.write("\n}\n")
f.write("int main(void)\n{\n\tinference_find();\n\twhile(1)\n\t{\n\t}\n}")
f.close()