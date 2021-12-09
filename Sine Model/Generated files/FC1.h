#include  <arm_const_structs.h>
#include  <arm_nnfunctions.h>
float sa0=15.875;//multiply input by sa0 and give as input to inference model
#define	IP1_IN_DIM	1
#define	IP1_OUT_DIM	16
#define	IP1_BIAS_LSHIFT	4
#define	IP1_OUT_RSHIFT	6
const q7_t wf_1[IP1_OUT_DIM*IP1_IN_DIM]={
-54,31,23,-3,-26,37,-49,58,0,0,3,-65,-63,35,-33,-26};
const q7_t bf_1[IP1_OUT_DIM]={
0,-72,-31,11,0,0,0,-29,0,0,112,0,0,0,0,0};
