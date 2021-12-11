# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:08:39 2021

@author: saile
"""

import tensorflow as tf
from PIL import Image
import numpy as np
from numpy import asarray
model=tf.keras.models.load_model('0-9-A-Z_selva.h5')    
img = Image.open('102.png')
p = asarray(img)
[x,y]=p.shape
p=p.reshape(x,y)
inp=p/255
aa=np.round(inp*127)
f=open('test.c','w')
f.write("//Copy paste the value from here to main file as input for OCR model\n")
f.write("q7_t data[]={\n")
for ii in range(x):
    for jj in range(y):
        if ii+jj==0:
            f.write(str(int(aa[ii][jj])))
        else:
            f.write(','+str(int(aa[ii][jj])))
f.write("\n};")
f.close()