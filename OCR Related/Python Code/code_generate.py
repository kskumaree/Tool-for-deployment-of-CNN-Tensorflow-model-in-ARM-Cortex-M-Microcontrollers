# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:17:44 2021

@author: sailesh
"""
#this code will generate the header and .c files
from all_apis import * #to import functions needed, not to be removed
import tensorflow as tf
import numpy as np
import math
from PIL import Image
from numpy import asarray

'''
The areas to be modified are model name, datalist variable and preprocessing data.
For Neural networks with convolution, maxpooling and fully connected layers with
ReLu as activation, the code will be generated correctly.
'''
#specify the tensorflow model here
model_name='0-9-A-Z_selva.h5'
model=tf.keras.models.load_model(model_name)
'''
Give the name of data file as shown below, atleast 5-10 sample data from different
class should yield good result. Give more data if the code generated isn't giving
desired results
'''
datalist=['102.png','146.png','149.png','150.png','157.png'] 
inp=[]
val=0
for idx in range(len(model.layers)):
    val+=1
maxval=np.zeros((val,len(datalist)))
maxvalout=np.zeros((val,len(datalist)))
outmax=np.zeros(len(datalist))
itr=-1
for data in datalist:
    '''
    The below code is for reading the data and preprocessing it accordingly before
    giving it to model. The code is given for reading and preprocessing a 
    grayscale image.
    For example, if the application is speech recognition, then modify the below 
    part to read a speech file and taking spectogram or any other technique.
    Finally, load the input into the variable inp with dimensions as
    [channels, height, width] format and as a numpy array
    '''
    itr+=1 #don't modify this variable
    
    #perform modifications according to the dataset from here onwards
    #read the image, if any other form of data is to be used, modify accordingly
    img = Image.open(data)
    p = asarray(img)
    #the code is for reading a grayscale image, if RGB or other colourspace
    #is used, modify the reshape function accordingly
    [x,y]=p.shape
    p=p.reshape(1,x,y)
    #make sure that the final preprocessed data is in inp variable with dimensions
    #as [channel,height,width]
    inp=p/255
    
    
    
    
    #Don't modify anything from this part onwards....
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
            inp=conv2d(inp,aa,b,pval,stride[0])
            o=(model.get_layer(index=idx).get_config()['activation'])
            if(o=='relu'):
                inp=relu(inp)
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
cl=0
FC=0
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
        if cl==1:
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
        else:
            for ii in range(p[0]):
                for jj in range(p[2]):
                    for kk in range(p[3]):
                        for ll in range(p[1]):
                            itr+=1
                            if(flag==0):
                                f.write(str(int(Wq[ii,ll,jj,kk])))
                                flag=1
                            else:
                                f.write(","+str(int(Wq[ii,ll,jj,kk])))
                            if (ii==p[0]-1 and jj==p[2]-1 and kk==p[3]-1 and ll==p[1]-1):
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
        FC+=1
        n=str(FC)
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
        nxt='\n'
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


f=open('inference.c','w')
model=tf.keras.models.load_model(model_name)
val=0
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
f.write("//Copy paste the contents in the main.c file\n")
func="void inference_find(void)\n{\n"
f.write(func)
#load tensorflow model
conv=0
fc=0
mp=0
f.write("//data variable is input 1D array, dtype=q7_t\n")
f.write("//buffer2,buffer3 and col_buffer1 are 1D array, dtype=q7_t\n")
f.write("//FC_OUT and SOFT_OUT, q7_t, 1D array, is output of final fully connected layer and size should be defined accordingly\n")
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
        CONVOLUTION(f,conv,itr)
        o=(model.get_layer(index=idx).get_config()['activation'])
        if(o=='relu'):
            RELU(f,conv,'conv')
    elif a.find('max_pooling2d')!=-1:
        print('max_pooling')
        mp=mp+1
        itr+=1
        MAX_POOL(f,mp)
    elif a.find('dense')!=-1:
        print('fully_connected')
        fc=fc+1
        itr+=1
        FULLY_CONNECTED(f,fc,itr,val)
        o=(model.get_layer(index=idx).get_config()['activation'])
        print(o)
        if(o=='relu'):
            RELU(f,fc,'fc')
        elif o=='softmax':
            SOFTMAX(f,fc)
f.write("\n}\n")
f.write("int main(void)\n{\n\tinference_find();\n\twhile(1)\n\t{\n\t}\n}")
f.close()