# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:32:53 2021

@author: Allan
"""

import numpy as np
#import onnx
import tensorflow as tf
#from onnx import numpy_helper
#import tf2onnx

#load tensorflow model
model=tf.keras.models.load_model('0-9-A-Z_selva.h5')
#model1=tf.keras.models.load_model('0-9-A-Z_quant.TFLITE')
#convert to onnx model
#spec=(tf.TensorSpec((None,20,20,1),tf.float32,name='input'),)
#output_path=model.name+'.onnx'
#onnx_model = tf2onnx.convert.from_keras(model,input_signature=spec,opset=13,output_path=output_path)
#output_names = [n.name for n in onnx_model.graph.output]
 
    
   
    
w=model.layers[0].get_weights()
a=w[0]
b=w[1]
aa = np.transpose(a, (3, 2, 0, 1))
print(aa[0][0])
print(aa[1][0])
'''
#load onnx model
model_o=onnx.load('sequential.onnx')

INTIALIZERS=model_o.graph.initializer

#get every layer weights and bias
Weight=[]
for initializer in INTIALIZERS:
    W= numpy_helper.to_array(initializer)
    Weight.append(W)
'''