/**
 ******************************************************************************
 * @file    port.h
 * @author  MCD/AIS Team
 * @brief   STM32 Helper functions for STM32 AI test application
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

/*
 * Simple header file with helper macro/function to adapt the ai test application.
 *  - before to include this file, NO_STM32_BSP_AI can be defined to "emulate" the
 *    HAL functions or low-level IO functions. Only ARM tool-chain related macro
 *    will be defined.
 *
 * History:
 *  - v1.0: initial version
 */

#ifndef __AI_STM32_ADAPTOR_H__
#define __AI_STM32_ADAPTOR_H__

#include "cmsis_compiler.h"


#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * MACRO - ARM tool chain definition
 * -----------------------------------------------------------------------------
 */

#undef _IS_AC5_COMPILER
#undef _IS_AC6_COMPILER
#undef _IS_GCC_COMPILER
#undef _IS_IAR_COMPILER

/* ARM Compiler 5 tool-chain */
#if defined ( __CC_ARM )
// #if ((__ARMCC_VERSION >= 5000000) && (__ARMCC_VERSION < 6000000))
#define _IS_AC5_COMPILER    1

/* ARM Compiler 6 tool-chain */
#elif defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
#define _IS_AC6_COMPILER    1

/* GCC tool-chain */
#elif defined ( __GNUC__ )
#define _IS_GCC_COMPILER    1

/* IAR tool-chain */
#elif defined ( __ICCARM__ )
#define _IS_IAR_COMPILER    1

#else
#error ARM MCU tool chain is not detected/supported
#endif

#undef _IS_ACx_COMPILER
#if defined(_IS_AC5_COMPILER) && _IS_AC5_COMPILER   \
    || defined(_IS_AC6_COMPILER) && _IS_AC6_COMPILER
#define _IS_ACx_COMPILER    1
#endif


/* -----------------------------------------------------------------------------
 * MACRO - MEM align definition
 * -----------------------------------------------------------------------------
 */

#define _CONCAT_ARG(a, b)     a ## b
#define _CONCAT(a, b)         _CONCAT_ARG(a, b)

#if defined(_IS_IAR_COMPILER) && _IS_IAR_COMPILER
  #define MEM_ALIGNED(x)         _CONCAT(MEM_ALIGNED_,x)
  #define MEM_ALIGNED_1          _Pragma("data_alignment = 1")
  #define MEM_ALIGNED_2          _Pragma("data_alignment = 2")
  #define MEM_ALIGNED_4          _Pragma("data_alignment = 4")
  #define MEM_ALIGNED_8          _Pragma("data_alignment = 8")
  #define MEM_ALIGNED_16         _Pragma("data_alignment = 16")
  #define MEM_ALIGNED_32         _Pragma("data_alignment = 32")
#elif defined(_IS_ACx_COMPILER) && _IS_ACx_COMPILER
  #define MEM_ALIGNED(x)         __attribute__((aligned (x)))
#elif defined(_IS_GCC_COMPILER) && _IS_GCC_COMPILER
  #define MEM_ALIGNED(x)         __attribute__((aligned(x)))
#else
  #define MEM_ALIGNED(x)
#endif


/* -----------------------------------------------------------------------------
 * Low-level IO functions
 * -----------------------------------------------------------------------------
 */

#if !defined(NO_STM32_BSP_AI)

#if (defined(CORSTONE_300) || defined(ARMCM55)) && !defined(STM32N6)

#include <bsp_ai.h>  /* generated STM32 platform file to import the HAL and the UART definition */
#include "uart.h"

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

#define port_hal_get_hal_version()        0
#define port_hal_get_dev_id()             0x155
#define port_hal_get_rev_id()             0
#define port_hal_rcc_get_hclk_freq()      25000000
#define port_hal_rcc_get_sys_clock_freq() 25000000

#define port_hal_delay(delay_)

#define HAS_NO_RCC_IP                     1

#define HAS_DEDICATED_PERF_COUNTER        1

#define PERF_COUNTER_INIT() \
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;\
	ARM_PMU_Enable(); \
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk); \
  LC_PRINT(" Timestamp    : based on ARM PMU\r\n")

#define PERF_COUNTER_GET() \
  ARM_PMU_Get_CCNTR()


#define HAS_DEDICATED_PRINT_PORT          1

__STATIC_INLINE void port_io_dedicated_putc(unsigned char ch)
{
  uart1_putc(ch);
}

__STATIC_INLINE bool port_io_dedicated_write(uint8_t *buff, int count)
{
  for (int i=0; i<count; i++)
    uart1_putc(*buff++);

  return true;
}

__STATIC_INLINE bool port_io_get(uint8_t *c, uint32_t timeout)
{
  *c = uart_getc();
	return 1;
}

__STATIC_INLINE bool port_io_write(uint8_t *buff, int count)
{
  for (int i=0; i<count; i++)
    uart_putc(*buff++);

  return true;
}

__STATIC_INLINE bool port_io_read(uint8_t *buff, int count)
{
  for (int i=0; i<count; i++)
    *buff++ = uart_getc();

  return true;
}

#elif defined(STM32N657xx) || defined(STM32N6)

#include "stm32n6xx_hal.h"
#include "stm32n6xx_stice_uart.h"
   
#define port_hal_get_hal_version()        HAL_GetHalVersion()
#define port_hal_get_dev_id()             0x486
#define port_hal_get_rev_id()             0
#define port_hal_rcc_get_hclk_freq()      HAL_RCC_GetHCLKFreq()
#define port_hal_rcc_get_sys_clock_freq() HAL_RCC_GetSysClockFreq()

#define port_hal_delay(delay_)            HAL_Delay(delay_)


__STATIC_INLINE bool port_io_get(uint8_t *c, uint32_t timeout)
{
  return (uart_read(c, 1) == 1);
}

__STATIC_INLINE bool port_io_write(uint8_t *buff, int count)
{
  int res = uart_write((char *)buff, count);

  return (res == count);
}

__STATIC_INLINE bool port_io_read(uint8_t *buff, int count)
{
  int res = uart_read(buff, count);

  return (res == count);
}

#else

#include <bsp_ai.h>  /* generated STM32 platform file to import the HAL and the UART definition */

#define port_hal_get_hal_version()        HAL_GetHalVersion()      
#define port_hal_get_dev_id()             HAL_GetDEVID() 
#define port_hal_get_rev_id()             HAL_GetREVID() 

#if defined(STM32MP1)
#define port_hal_rcc_get_hclk_freq()              HAL_RCC_GetSystemCoreClockFreq()
#else
#define port_hal_rcc_get_hclk_freq()              HAL_RCC_GetHCLKFreq()
#endif
#define port_hal_rcc_get_sys_clock_freq()         HAL_RCC_GetSysClockFreq()
#define port_hal_rcc_get_system_core_clock_freq() HAL_RCC_GetSystemCoreClockFreq()

#define port_hal_delay(delay_)                    HAL_Delay(delay_)

extern UART_HandleTypeDef UartHandle;

__STATIC_INLINE bool port_io_get(uint8_t *c, uint32_t timeout)
{
  HAL_StatusTypeDef status;

  if (!c)
    return false;

  status = HAL_UART_Receive(&UartHandle, (uint8_t *)c, 1, timeout);

  if (status == HAL_TIMEOUT)
    return false;

  return (status == HAL_OK);
}

__STATIC_INLINE bool port_io_write(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status;

  status = HAL_UART_Transmit(&UartHandle, buff, count, HAL_MAX_DELAY);

  return (status == HAL_OK);
}

__STATIC_INLINE bool port_io_read(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status;

  status = HAL_UART_Receive(&UartHandle, buff, count, HAL_MAX_DELAY);

  return (status == HAL_OK);
}

#endif

#endif /* NO_STM32_BSP_AI */

#ifdef __cplusplus
}
#endif

#endif /* __AI_STM32_ADAPTOR_H__  */
