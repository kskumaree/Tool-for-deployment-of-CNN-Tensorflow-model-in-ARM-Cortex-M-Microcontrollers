#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "arm_const_structs.h"
#include "arm_nnfunctions.h"


#define CONV1_IN_y 32
#define CONV1_IN_x 39
#define CONV1_IN_CH 1
#define CONV1_OUT_CH 12																		
#define CONV1_KER_y 3
#define CONV1_KER_x 3
#define CONV1_PAD_y 1
#define CONV1_PAD_x 1																		
#define CONV1_STRIDE_y 1
#define CONV1_STRIDE_x 1
#define CONV1_BIAS_LSHIFT 4
#define CONV1_OUT_RSHIFT 8
#define CONV1_OUT_y 32
#define CONV1_OUT_x 39

#define MAX1_IN_y 32
#define MAX1_IN_x 39
#define MAX1_OUT_y 16
#define MAX1_OUT_x 20
#define MAX1_STRIDE_y 2
#define MAX1_STRIDE_x 2
#define MAX1_KERNEL_y 3
#define MAX1_KERNEL_x 3
#define MAX1_PAD_y 0
#define MAX1_PAD_x 1
#define MAX1_ACT_min 0
#define MAX1_ACT_max 127
#define MAX1_CHANNEL_in 12


const q7_t b_1[CONV1_OUT_CH]={14,-7,6,6,1,-1,4,16,9,3,9,14};

const q7_t W_1[CONV1_IN_CH*CONV1_KER_y*CONV1_KER_x*CONV1_OUT_CH]={

-13,-8,-34,
10,-7,-7,
27,26,16,

18,-11,-16,
-7,9,-20,
-12,-6,-6,

-3,44,44,
-1,-18,-16,
12,-32,-19,

23,23,-17,
15,-5,-24,
-9,27,-20,

-3,-17,-4,
-14,9,31,
-18,19,54,

11,-8,7,
35,-22,-22,
14,25,-28,

-6,6,-5,
-29,-18,-7,
21,-28,13,

37,37,31,
23,1,35,
39,46,24,

11,-17,4,
-12,4,8,
-12,-9,-25,

-16,-29,-5,
-12,19,28,
-6,-5,36,

15,24,51,
53,69,49,
-8,46,48,

-25,18,-11,
-2,15,-23,
-10,-4,21





};






