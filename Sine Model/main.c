#include "math.h"
#include "arm_const_structs.h"
#include "arm_nnfunctions.h"
#include <FC1.h>
#include <FC2.h>
#include <FC3.h>
q7_t data[1];
q7_t buffer2[200],buffer3[300],FC_OUT[IP3_OUT_DIM];
q15_t buffer1[100];
void inference_find(void)
{
	data[0]=round(6.28*sa0);
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
