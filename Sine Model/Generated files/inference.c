#include <arm_const_structs.h>
#include <arm_nnfunctions.h
#include <FC1.h>
#include <FC2.h>
#include <FC3.h>
//Copy paste the contents in the main.c file
void inference_find(void)
{
//data variable is input 1D array, dtype=q7_t
//buffer2,buffer3 and col_buffer1 are 1D array, dtype=q7_t
//FC_OUT and SOFT_OUT, q7_t, 1D array, is output of final fully connected layer and size should be defined accordingly
//buffer2 and 3 should be maximum possible size of the available input and output layers
//col_buffer1 should be atleast the size of max(2*ch_im_in*dim_kernel*dim_kernel)
//buffer1 size should be atleast the max input length of fully connected layer, dtype=q15_t, 1D array
arm_fully_connected_q7(data,wf_1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bf_1,buffer2,buffer1);
arm_relu_q7(buffer2,IP1_OUT_DIM);
arm_fully_connected_q7(buffer2,wf_2,IP2_IN_DIM,IP2_OUT_DIM,IP2_BIAS_LSHIFT,IP2_OUT_RSHIFT,bf_2,buffer3,buffer1);
arm_relu_q7(buffer3,IP2_OUT_DIM);
arm_fully_connected_q7(buffer3,wf_3,IP3_IN_DIM,IP3_OUT_DIM,IP3_BIAS_LSHIFT,IP3_OUT_RSHIFT,bf_3,FC_OUT,buffer1);

}
int main(void)
{
	inference_find();
	while(1)
	{
	}
}