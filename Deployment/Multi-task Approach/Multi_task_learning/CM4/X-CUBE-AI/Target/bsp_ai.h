/**
  ******************************************************************************
  * @file    bsp_ai.h
  * @author  X-CUBE-AI C code generator
  * @brief   AI BSP dependencies
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef BSP_H
#define BSP_H
#ifdef __cplusplus
 extern "C" {
#endif
#include "main.h"
#include "stm32wlxx.h"
#include "app_x-cube-ai.h"
#include "constants_ai.h"
#define UartHandle huart2
#define MX_UARTx_Init MX_USART2_UART_Init
void MX_USART2_UART_Init(void);
#ifdef __cplusplus
}
#endif

#endif /* BSP_H */
