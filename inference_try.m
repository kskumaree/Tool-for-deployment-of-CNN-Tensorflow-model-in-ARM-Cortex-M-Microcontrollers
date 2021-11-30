clc;
close all;
clear all;
%CL1
sa0=0.003921568859368563;
sw1=[0.0017426731,0.0015977592,0.0015251781,0.0016185867,0.0016740632,0.0016916838,0.0015270688,0.0015731148,0.0014213565,0.0015661800,0.0017329623,0.0015208790,0.0015016894,0.0016833844,0.0013887297,0.0017495891,0.0013711541,0.0017182189,0.0015628298,0.0016898111,0.0011963974,0.0014681807,0.0016952109,0.0016127513,0.0016469752,0.0014165718,0.0017453862,0.0016016284,0.0015462812,0.0014921135,0.0015413135,0.0016377481];
sb1=sw1.*sa0;
%CL2
sa1= 0.004344021435827017;
sw2=[0.0013413618,0.0013520624,0.0014790706,0.0013841089,0.0013851273,0.0014564196,0.0014002883,0.0012513357,0.0013370903,0.0012955811,0.0012568060,0.0014668640,0.0012727248,0.0012385187,0.0012952576,0.0013502504,0.0012132599,0.0015596404,0.0013251913,0.0013092585,0.0015084486,0.0013389776,0.0016267981,0.0014239589,0.0012652392,0.0014389638,0.0011626292,0.0014880087,0.0013608973,0.0012395494,0.0013110966,0.0013931195,0.0013697280,0.0014148237,0.0014269729,0.0013266760,0.0012327449,0.0012139351,0.0015460884,0.0013417637,0.0013802294,0.0014213255,0.0012610285,0.0013050266,0.0015125464,0.0012627936,0.0014482051,0.0014785447,0.0013420437,0.0014645664,0.0012542326,0.0012927066,0.0013275787,0.0012959859,0.0012568854,0.0013344585,0.0013331615,0.0015960069,0.0013499231,0.0012803966,0.0013268593,0.0013781232,0.0012753461,0.0012161036];
sb2=sw2.*sa1;
%FC1
sa2=0.017036087810993195;
swf1=0.0015732995234429836;
sbf1=0.000026802868887898512;

net = importKerasNetwork('0-9-A-Z_selva.h5');
%analyzeNetwork(net)
%Input Layer Details
n_L=net.Layers(1,1).InputSize;
%Convolution Layer1 Details
n_C1=net.Layers(2,1).NumChannels;
f1=net.Layers(2,1).FilterSize;
S1=net.Layers(2,1).Stride;
padmode1=net.Layers(2,1).PaddingMode;
pad1=net.Layers(2,1).PaddingSize;
W1=net.Layers(2,1).Weights;
for i=1:32  
    Wq1(:,:,:,i)=round(W1(:,:,:,i).*128);  
end
b1=net.Layers(2,1).Bias;
for i=1:32
    bq1(:,:,i)=round(b1(i)*(128*256));
end
  
%File Writting
fileID1 = fopen('W1_float.txt','w');
for i=1:size(W1,4)
     fprintf(fileID1,'\n');
    weight=W1(:,:,:,i);
    for j=1:size(W1,3)
        
    weight1=weight(:,:,j);
    for k=1:size(W1,2)
        for l=1:size(W1,1)
          fprintf(fileID1,'%f,',single(weight1(k,l)));  
        end
         fprintf(fileID1,'\n');
    end
        end
   
    
end
fclose(fileID1);

fileID1 = fopen('bias_float.txt','w');
for i=1:size(W1,4)
    fprintf(fileID1,'%f,',single(b1(:,:,i)));
end
fclose(fileID1);
%%File Quantized
fileID1 = fopen('W1_int8.txt','w');
for i=1:size(Wq1,4)
     fprintf(fileID1,'\n');
    weight=Wq1(:,:,:,i);
    for j=1:size(Wq1,3)
        
    weight1=weight(:,:,j);
    for k=1:size(Wq1,2)
        for l=1:size(Wq1,1)
          fprintf(fileID1,'%d,',int8(weight1(k,l)));  
        end
         fprintf(fileID1,'\n');
    end
        end
   
    
end
fclose(fileID1);

fileID1 = fopen('bias1_int32.txt','w');
for i=1:size(Wq1,4)
    fprintf(fileID1,'%d,',int32(bq1(:,:,i)));
end
fclose(fileID1);


%Convolution Layer2 Details
n_C2=net.Layers(5,1).NumChannels;
f2=net.Layers(5,1).FilterSize;
S2=net.Layers(5,1).Stride;
padmode2=net.Layers(5,1).PaddingMode;
pad2=net.Layers(5,1).PaddingSize;
W2=net.Layers(5,1).Weights;
b2=net.Layers(5,1).Bias;
for i=1:64
       Wq2(:,:,:,i)=round(W2(:,:,:,i).*128);
    end


for i=1:64
    bq2(:,:,i)=round(b2(i).*(64*128));
end
%File Writting
fileID1 = fopen('W2_float.txt','w');
for i=1:size(W2,4)
     fprintf(fileID1,'\n');
    weight2=W2(:,:,:,i);
    for j=1:size(W2,3)
        
    weight12=weight2(:,:,j);
    for k=1:size(W2,2)
        for l=1:size(W2,1)
          fprintf(fileID1,'%f,',single(weight12(k,l)));  
        end
         fprintf(fileID1,'\n');
    end
        end
   
    
end
fclose(fileID1);

fileID1 = fopen('b2_float.txt','w');
for i=1:size(W2,4)
    fprintf(fileID1,'%f,',single(b2(:,:,i)));
end
fclose(fileID1);
%%File Quantized
fileID1 = fopen('W2_int8.txt','w');
for i=1:size(Wq2,4)
     fprintf(fileID1,'\n');
    weight=Wq2(:,:,:,i);
    for j=1:size(Wq2,3)
        
    weight1=weight(:,:,j);
    for k=1:size(Wq2,2)
        for l=1:size(Wq2,1)
          fprintf(fileID1,'%d,',int8(weight1(k,l)));  
        end
         fprintf(fileID1,'\n');
    end
        end
   
    
end
fclose(fileID1);

fileID1 = fopen('bias2_int32.txt','w');
for i=1:size(Wq2,4)
    fprintf(fileID1,'%d,',int32(bq2(:,:,i)));
end
fclose(fileID1);

  
%Extract Theta1 from the net
wf1=net.Layers(10,1).Weights;
bf1=net.Layers(10,1).Bias;
wfq1=round(wf1.*128);
bfq1=round(bf1.*(16*128));
fileID1 = fopen('wf1_int8.txt','w');
for i=1:size(wfq1,1)
    for j=1:size(wfq1,2)
fprintf(fileID1,'%d,',int8(wfq1(i,j)));
    end
    fprintf(fileID1,'\n');
end
fclose(fileID1);

fileID1 = fopen('bf1_int32.txt','w');
for i=1:size(bfq1,1)
    fprintf(fileID1,'%d,',int32(bfq1(i)));
end
fclose(fileID1);
%%inference
a=imread('102.png');
XTest1_q=single(a);
a=double(a);
XTest1=a/255;
%File Handling
fileID1 = fopen('XTest_int8.txt','w');
for i=1:size(XTest1,1)
    for j=1:size(XTest1,2)
fprintf(fileID1,'%d,',uint8(XTest1_q(i,j)));
    end
    fprintf(fileID1,'\n');
end
fclose(fileID1);
imshow(a);

%analyzeNetwork(net1)
classes=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','H','K','N','R','U','Y','E','P','S','X','F','L','T','Z','G','O','I','J','M','Q','V','W'];

%%CL1
padd='both';
conv1_out=conv_infer(XTest1,W1,b1,pad1,S1,padd);
conv1_out_q=conv_infer(XTest1_q,Wq1,bq1,pad1,S1,padd);
%Relu
conv1_out_relu=max(conv1_out,0);
conv1_out_relu_q=max(conv1_out_q,0);
%MaxPooling
padd='both';
f_pool1=net.Layers(4,1).PoolSize;
S_pool1=net.Layers(4,1).Stride;
Mpad1=net.Layers(4,1).PaddingSize;
conv1_out_relu=zero_pad(conv1_out_relu,Mpad1(2),padd);
conv1_out_relu_q=zero_pad(conv1_out_relu_q,Mpad1(2),padd);
max_pool1=pool_infer(conv1_out_relu,f_pool1,S_pool1);
max_pool1_q=pool_infer(conv1_out_relu_q,f_pool1,S_pool1);
%Requantization
for i=1:32
    scale1(i)=sb1(i)/sa1;
    max_pool1_qs(:,:,i)=round(max_pool1_q(:,:,i).*(64/(128*256)));
     max_pool1_qstest(:,:,i)=max_pool1_q(:,:,i)./(128*256);
   
   
end
fileID1 = fopen('scale1.txt','w');
for i=1:32
    fprintf(fileID1,'%f,',single(scale1(i)));
end
fclose(fileID1);



%%CL2
padd='both';
conv2_out=conv_infer(max_pool1,W2,b2,pad2,S2,padd);
conv2_out_q=conv_infer(max_pool1_qs,Wq2,bq2,pad2,S2,padd);
%Relu
conv2_out_relu=max(conv2_out,0);
conv2_out_relu_q=max(conv2_out_q,0);
f_pool2=net.Layers(7,1).PoolSize;
S_pool2=net.Layers(7,1).Stride;
Mpad2=net.Layers(7,1).PaddingSize;
conv2_out_relu=zero_pad(conv2_out_relu,Mpad2(2),padd);
conv2_out_relu_q=zero_pad(conv2_out_relu_q,Mpad2(2),padd);
max_pool2=pool_infer(conv2_out_relu,f_pool2,S_pool2);
max_pool2_q=pool_infer(conv2_out_relu_q,f_pool2,S_pool2);
%Requantization
for i=1:64
     scale2(i)=sb2(i)/sa2;
    max_pool2_qs(:,:,i)=round(max_pool2_q(:,:,i).*(16/(64*128)));
   max_pool2_qstest(:,:,i)=max_pool2_q(:,:,i)./(64*128);
   
end
fileID1 = fopen('scale2.txt','w');
for i=1:64
    fprintf(fileID1,'%f,',single(scale2(i)));
end
fclose(fileID1);

%%Flatten
max_poolT=permute(max_pool2,[3 2 1]);
max_poolT_q=permute(max_pool2_qs,[3 2 1]);
X=max_poolT(:);
X_q=max_poolT_q(:);
%%FC1
[pred g1]= digit_FC(bf1,wf1,X);
[pred_q g1_q]= digit_FC(bfq1,wfq1,X_q);
pred_qs=pred_q/(16*128);
m = max(pred);
m_q = max(pred_qs);
e = exp(pred-m);
e_q = exp(pred_qs-m_q);
dist = e /sum(e);
dist_q = e_q /sum(e_q);
[score idx]=max(dist);
[score_q idx_q]=max(dist_q);
digit=classes(idx)
digit_q=classes(idx_q)
