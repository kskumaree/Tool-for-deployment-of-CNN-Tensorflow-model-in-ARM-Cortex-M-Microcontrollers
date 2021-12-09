#include  <arm_const_structs.h>
#include  <arm_nnfunctions.h>
#define	IP3_IN_DIM	16
#define	IP3_OUT_DIM	1
#define	IP3_BIAS_LSHIFT	5
#define	IP3_OUT_RSHIFT	6
const q7_t wf_3[IP3_OUT_DIM*IP3_IN_DIM]={
4,60,16,-9,-9,16,71,-8,9,-43,-20,-18,1,-14,-8,0};
const q7_t bf_3[IP3_OUT_DIM]={
-9};
