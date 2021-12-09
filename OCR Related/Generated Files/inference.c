#include <arm_const_structs.h>
#include <arm_nnfunctions.h
#include <conv1.h>
#include <max_pooling1.h>
#include <conv2.h>
#include <max_pooling2.h>
#include <FC1.h>
//Copy paste the contents in the main.c file
void inference_find(void)
{
//data variable is input 1D array, dtype=q7_t
//buffer2,buffer3 and col_buffer1 are 1D array, dtype=q7_t
//FC_OUT and SOFT_OUT, q7_t, 1D array, is output of final fully connected layer and size should be defined accordingly
//buffer2 and 3 should be maximum possible size of the available input and output layers
//col_buffer1 should be atleast the size of max(2*ch_im_in*dim_kernel*dim_kernel)
//buffer1 size should be atleast the max input length of fully connected layer, dtype=q15_t, 1D array
/* the input value should be in data variable. To obtain this, multiply
the floating point input by sa0 variable, which is stored in weight files
already and round off the value and save it in data variable*/
arm_convolve_HWC_q7_basic_nonsquare(data,CONV1_IN_x,CONV1_IN_y,CONV1_IN_CH,W_1,CONV1_OUT_CH,CONV1_KER_x,CONV1_KER_y,CONV1_PAD_x,CONV1_PAD_y,CONV1_STRIDE_x,CONV1_STRIDE_y,b_1,CONV1_BIAS_LSHIFT,CONV1_OUT_RSHIFT,buffer2,CONV1_OUT_x,CONV1_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(buffer2,CONV1_OUT_x*CONV1_OUT_y*CONV1_OUT_CH);
arm_max_pool_s8_opt(MAX1_IN_y,MAX1_IN_x,MAX1_OUT_y,MAX1_OUT_x,MAX1_STRIDE_y,MAX1_STRIDE_x,MAX1_KERNEL_y,MAX1_KERNEL_x,MAX1_PAD_y,MAX1_PAD_x,MAX1_ACT_min,MAX1_ACT_max,MAX1_CHANNEL_in,buffer2,NULL,buffer3);
arm_convolve_HWC_q7_basic_nonsquare(buffer3,CONV2_IN_x,CONV2_IN_y,CONV2_IN_CH,W_2,CONV2_OUT_CH,CONV2_KER_x,CONV2_KER_y,CONV2_PAD_x,CONV2_PAD_y,CONV2_STRIDE_x,CONV2_STRIDE_y,b_2,CONV2_BIAS_LSHIFT,CONV2_OUT_RSHIFT,buffer2,CONV2_OUT_x,CONV2_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(buffer2,CONV2_OUT_x*CONV2_OUT_y*CONV2_OUT_CH);
arm_max_pool_s8_opt(MAX2_IN_y,MAX2_IN_x,MAX2_OUT_y,MAX2_OUT_x,MAX2_STRIDE_y,MAX2_STRIDE_x,MAX2_KERNEL_y,MAX2_KERNEL_x,MAX2_PAD_y,MAX2_PAD_x,MAX2_ACT_min,MAX2_ACT_max,MAX2_CHANNEL_in,buffer2,NULL,buffer3);
arm_fully_connected_q7(buffer3,wf_1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bf_1,FC_OUT,buffer1);
arm_softmax_q7(FC_OUT,IP1_OUT_DIM,SOFT_OUT);

}
int main(void)
{
	inference_find();
	while(1)
	{
	}
}