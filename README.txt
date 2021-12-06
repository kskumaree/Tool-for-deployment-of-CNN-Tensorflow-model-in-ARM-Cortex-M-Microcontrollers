# MicroQuantization-Tool
Automated Inference Code Generation for ARM Microcontrollers
1. The "OCR Related" folder has the files needed for deploying the model in microcontroller. Both .h5 and.tflite model is provided. 
Use the OCR_Test python file to generate the files for weights, biases and other parameters for all layers. Next run the Infer_gen to generate the .c file containing inference
function for prediction This will generate a inference.c file, use it wherever necessary . A main.c file has been added for reference. Use input_generate_OCR to
generate a file containing the variable data needed as input in inference.c 
