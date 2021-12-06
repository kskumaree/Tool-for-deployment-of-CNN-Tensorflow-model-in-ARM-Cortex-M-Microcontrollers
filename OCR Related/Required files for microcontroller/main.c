/* USER CODE BEGIN Header */
#include "main.h"
#include "arm_const_structs.h"
#include "arm_nnfunctions.h"
#include <conv1.h>
#include <max_pooling1.h>
#include <conv2.h>
#include <max_pooling2.h>
#include <FC1.h>
void SystemClock_Config(void);

q7_t buffer2[16000];
q7_t buffer3[16000];
q15_t buffer1[IP1_IN_DIM];
q7_t col_buffer1[2000];
q7_t FC_OUT[IP1_OUT_DIM];
//Buffers
q7_t col_buffer1[2000];
char predict;
float s,ex[36],max,ex1[36];
int loc;
q7_t data[]={
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,94,118,118,118,118,118,116,114,106,91,55,7,0,0,0,0,0,0,0,15,102,127,127,127,127,127,127,127,127,127,127,111,41,0,0,0,0,0,0,15,102,127,127,116,99,99,99,101,112,125,127,127,124,22,0,0,0,0,0,15,102,127,127,79,1,0,0,1,8,75,127,127,127,89,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,11,127,127,127,112,4,0,0,0,0,15,102,127,127,79,1,0,0,0,0,5,126,127,127,112,3,0,0,0,0,15,102,127,127,79,1,0,0,0,1,62,127,127,127,90,0,0,0,0,0,15,102,127,127,102,61,60,62,68,93,123,127,127,121,28,0,0,0,0,0,15,102,127,127,127,127,127,127,127,127,127,127,118,44,0,0,0,0,0,0,15,102,127,127,127,127,127,127,123,114,97,59,17,1,0,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,102,127,127,79,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,33,41,41,25,0,0,0,0,0,0,0,0,0,0,0,0
};

void inference_find(void)
{
	arm_convolve_HWC_q7_basic_nonsquare(data,CONV1_IN_x,CONV1_IN_y,CONV1_IN_CH,W_1,CONV1_OUT_CH,CONV1_KER_x,CONV1_KER_y,CONV1_PAD_x,CONV1_PAD_y,
			CONV1_STRIDE_x,CONV1_STRIDE_y,b_1,CONV1_BIAS_LSHIFT,CONV1_OUT_RSHIFT,buffer2,CONV1_OUT_x,CONV1_OUT_y,(q15_t*)col_buffer1,NULL);
	arm_relu_q7(buffer2,CONV1_OUT_x*CONV1_OUT_y*CONV1_OUT_CH);
	arm_max_pool_s8_opt(MAX1_IN_y,MAX1_IN_x,MAX1_OUT_y,MAX1_OUT_x,MAX1_STRIDE_y,MAX1_STRIDE_x,MAX1_KERNEL_y,MAX1_KERNEL_x,MAX1_PAD_y,MAX1_PAD_x,
			MAX1_ACT_min,MAX1_ACT_max,MAX1_CHANNEL_in,buffer2,NULL,buffer3);
	arm_convolve_HWC_q7_basic_nonsquare(buffer3,CONV2_IN_x,CONV2_IN_y,CONV2_IN_CH,W_2,CONV2_OUT_CH,CONV2_KER_x,CONV2_KER_y,CONV2_PAD_x,CONV2_PAD_y,
			CONV2_STRIDE_x,CONV2_STRIDE_y,b_2,CONV2_BIAS_LSHIFT,CONV2_OUT_RSHIFT,buffer2,CONV2_OUT_x,CONV2_OUT_y,(q15_t*)col_buffer1,NULL);
	arm_relu_q7(buffer2,CONV2_OUT_x*CONV2_OUT_y*CONV2_OUT_CH);
	arm_max_pool_s8_opt(MAX2_IN_y,MAX2_IN_x,MAX2_OUT_y,MAX2_OUT_x,MAX2_STRIDE_y,MAX2_STRIDE_x,MAX2_KERNEL_y,MAX2_KERNEL_x,MAX2_PAD_y,MAX2_PAD_x,
			MAX2_ACT_min,MAX2_ACT_max,MAX2_CHANNEL_in,buffer2,NULL,buffer3);
	arm_fully_connected_q7(buffer3,wf_1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bf_1,FC_OUT,buffer1);
char class[]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','H','K','N','R','U','Y','E','P','S','X','F','L','T','Z','G','O','I','J','M','Q','V','W'};
int max_fout=FC_OUT[0];
loc=0;
	for(int i=0; i<36;i++){
		if(FC_OUT[i]>max_fout)
		{
			max_fout=FC_OUT[i];
			loc=i;
		}
	}
predict=class[loc];
s=0;
	for(int i=0;i<36;i++)
	{
		ex[i]=exp(FC_OUT[i]-max_fout);
	}
	for(int i=0;i<36;i++)
	{
		s=s+ex[i];
	}
	for(int i=0;i<36;i++)
	{
		ex1[i]=ex[i]/s;
	}
	max=ex1[0];
	loc=0;
	for(int i=0; i<36;i++){
		if(ex1[i]>max)
		{
			max=ex1[i];
			loc=i;
		}
	}
}

int main(void)
{

  HAL_Init();
  SystemClock_Config();
  inference_find();
  while (1)
  {
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

