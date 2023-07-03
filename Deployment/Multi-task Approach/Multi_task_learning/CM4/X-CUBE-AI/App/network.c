/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sat Jul  1 10:57:38 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "5798d4c26bfdd6f0df72cda62d0627fd"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sat Jul  1 10:57:38 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 640, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_19_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  dense_19_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  dense_20_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  dense_20_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  dense_21_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  dense_21_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 3, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1680, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 560, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_10_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2205, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 560, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 17640, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  pool_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8808, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  dense_19_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 8, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8824, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8808, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  pool_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4392, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4408, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4392, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  pool_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2184, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  pool_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  dense_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  dense_20_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  dense_21_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  dense_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  dense_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)
/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  dense_21_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  nl_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  nl_22_0_conversion_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3, AI_STATIC)
/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016563667450100183f, 0.0028181406669318676f, 0.002626054221764207f, 0.00192452990449965f, 0.0019351301016286016f, 0.0015497681451961398f, 0.002307099988684058f, 0.0018686687108129263f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00314652593806386f, 0.0029374118894338608f, 0.0029526955913752317f, 0.003128229407593608f, 0.0025434361305087805f, 0.0034254975616931915f, 0.0034084422513842583f, 0.0035305789206176996f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0034988338593393564f, 0.004487252794206142f, 0.0032902841921895742f, 0.0029750552494078875f, 0.0038542947731912136f, 0.003246623557060957f, 0.003605401609092951f, 0.005760717671364546f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_19_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005809392314404249f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_23_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008224469609558582f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_24_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006961953826248646f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_20_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008261643350124359f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_21_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011376461014151573f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.16388989984989166f),
    AI_PACK_INTQ_ZP(31)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10209852457046509f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10209852457046509f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10209852457046509f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11817066371440887f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11817066371440887f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11817066371440887f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10592639446258545f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10592639446258545f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01337256096303463f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_19_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011778763495385647f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_23_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008778436109423637f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_24_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07273895293474197f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009807223454117775f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07057344168424606f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_22_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 1, 8), AI_STRIDE_INIT(4, 1, 1, 80, 80),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 1, 8), AI_STRIDE_INIT(4, 1, 8, 24, 24),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 1, 8), AI_STRIDE_INIT(4, 1, 8, 24, 24),
  1, &conv2d_13_weights_array, &conv2d_13_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_13_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  dense_19_weights, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 8, 32, 1, 1), AI_STRIDE_INIT(4, 1, 8, 256, 256),
  1, &dense_19_weights_array, &dense_19_weights_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  dense_19_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_19_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  dense_23_weights, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 32, 32, 1, 1), AI_STRIDE_INIT(4, 1, 32, 1024, 1024),
  1, &dense_23_weights_array, &dense_23_weights_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_23_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_23_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  dense_24_weights, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 1, 1), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &dense_24_weights_array, &dense_24_weights_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  dense_24_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_24_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  dense_20_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 32, 32, 1, 1), AI_STRIDE_INIT(4, 1, 32, 1024, 1024),
  1, &dense_20_weights_array, &dense_20_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  dense_20_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_20_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  dense_21_weights, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 1, 1), AI_STRIDE_INIT(4, 1, 32, 96, 96),
  1, &dense_21_weights_array, &dense_21_weights_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  dense_21_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &dense_21_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 1680, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1680, 1680),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 560, 1, 1), AI_STRIDE_INIT(4, 1, 1, 560, 560),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_10_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 2205), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &serving_default_input_10_output_array, &serving_default_input_10_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_10_output0, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 2205, 1), AI_STRIDE_INIT(4, 1, 1, 1, 2205),
  1, &serving_default_input_10_output_array, &serving_default_input_10_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 560, 1, 1), AI_STRIDE_INIT(4, 1, 1, 560, 560),
  1, &conv2d_13_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 2205, 1), AI_STRIDE_INIT(4, 1, 1, 8, 17640),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output0, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 2205), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  pool_4_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 1101), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &pool_4_output_array, &pool_4_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  pool_4_output0, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1101, 1), AI_STRIDE_INIT(4, 1, 1, 8, 8808),
  1, &pool_4_output_array, &pool_4_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  dense_19_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 2, 2, 16, 16),
  1, &dense_19_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_pad_before_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1103, 1), AI_STRIDE_INIT(4, 1, 1, 8, 8824),
  1, &conv2d_7_pad_before_output_array, &conv2d_7_pad_before_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1101, 1), AI_STRIDE_INIT(4, 1, 1, 8, 8808),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output0, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 1101), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 549), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output0, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 549, 1), AI_STRIDE_INIT(4, 1, 1, 8, 4392),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  dense_23_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_23_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_pad_before_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 551, 1), AI_STRIDE_INIT(4, 1, 1, 8, 4408),
  1, &conv2d_13_pad_before_output_array, &conv2d_13_pad_before_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 549, 1), AI_STRIDE_INIT(4, 1, 1, 8, 4392),
  1, &conv2d_13_output_array, &conv2d_13_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output0, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 549), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &conv2d_13_output_array, &conv2d_13_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  dense_24_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_24_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  pool_16_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 273), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &pool_16_output_array, &pool_16_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  pool_18_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 1, 1, 8, 8),
  1, &pool_18_output_array, &pool_18_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  dense_19_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &dense_19_output_array, &dense_19_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  dense_20_scratch0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_20_scratch0_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  dense_23_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &dense_23_output_array, &dense_23_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  dense_24_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &dense_24_output_array, &dense_24_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  dense_21_scratch0, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_21_scratch0_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_25_output_array, &nl_25_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  dense_20_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &dense_20_output_array, &dense_20_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  dense_21_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3, 3),
  1, &dense_21_output_array, &dense_21_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  dense_21_0_conversion_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &dense_21_0_conversion_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  nl_22_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &nl_22_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  nl_22_0_conversion_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3, 3),
  1, &nl_22_0_conversion_output_array, &nl_22_0_conversion_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_22_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_22_0_conversion_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_22_0_conversion_chain,
  NULL, &nl_22_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_21_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_22_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &nl_22_chain,
  NULL, &nl_22_0_conversion_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_21_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_21_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_21_0_conversion_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &dense_21_0_conversion_chain,
  NULL, &nl_22_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_21_weights, &dense_21_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_21_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_21_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_21_chain,
  NULL, &dense_21_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_20_weights, &dense_20_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_20_layer, 20,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_20_chain,
  NULL, &dense_21_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_25_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -123, -123, -123, -122, -122, -121, -121, -120, -120, -119, -119, -118, -117, -116, -116, -115, -114, -113, -112, -111, -109, -108, -107, -105, -104, -102, -100, -98, -97, -94, -92, -90, -88, -85, -82, -80, -77, -74, -70, -67, -64, -60, -56, -53, -49, -45, -40, -36, -32, -27, -23, -18, -14, -9, -5, 0, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49, 53, 56, 60, 64, 67, 70, 74, 77, 80, 82, 85, 88, 90, 92, 94, 97, 98, 100, 102, 104, 105, 107, 108, 109, 111, 112, 113, 114, 115, 116, 116, 117, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_25_nl_params, AI_ARRAY_FORMAT_S8,
    nl_25_nl_params_data, nl_25_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_25_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_25_chain,
  NULL, &dense_20_layer, AI_STATIC, 
  .nl_params = &nl_25_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_24_weights, &dense_24_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_24_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_24_layer, 24,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_24_chain,
  NULL, &nl_25_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_23_weights, &dense_23_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_23_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_23_layer, 23,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_23_chain,
  NULL, &dense_24_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_19_weights, &dense_19_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_19_layer, 19,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_19_chain,
  NULL, &dense_23_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_18_layer, 18,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_18_chain,
  NULL, &dense_19_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 273), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 273), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_16_layer, 16,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_16_chain,
  NULL, &pool_18_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_13_weights, &conv2d_13_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_13_chain,
  NULL, &pool_16_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_13_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_13_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_13_pad_before_value_data, conv2d_13_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_pad_before_layer, 13,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_13_pad_before_chain,
  NULL, &conv2d_13_layer, AI_STATIC, 
  .value = &conv2d_13_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_10_layer, 10,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_10_chain,
  NULL, &conv2d_13_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_7_chain,
  NULL, &pool_10_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_7_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_7_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_7_pad_before_value_data, conv2d_7_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_pad_before_layer, 7,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_7_pad_before_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .value = &conv2d_7_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_4_layer, 4,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_4_chain,
  NULL, &conv2d_7_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_10_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_1_chain,
  NULL, &pool_4_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 39, 0, 40), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3952, 1, 1),
    3952, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 19320, 1, 1),
    19320, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_25_output, &nl_22_0_conversion_output),
  &conv2d_1_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3952, 1, 1),
      3952, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 19320, 1, 1),
      19320, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_25_output, &nl_22_0_conversion_output),
  &conv2d_1_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    conv2d_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_1_output_array.data = AI_PTR(g_network_activations_map[0] + 1680);
    conv2d_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1680);
    
    pool_4_output_array.data = AI_PTR(g_network_activations_map[0] + 1664);
    pool_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1664);
    
    conv2d_7_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 10472);
    conv2d_7_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10472);
    
    conv2d_7_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_7_output_array.data = AI_PTR(g_network_activations_map[0] + 560);
    conv2d_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 560);
    
    pool_10_output_array.data = AI_PTR(g_network_activations_map[0] + 9368);
    pool_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9368);
    
    conv2d_13_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_13_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_13_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 4408);
    conv2d_13_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 4408);
    
    conv2d_13_output_array.data = AI_PTR(g_network_activations_map[0] + 4968);
    conv2d_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4968);
    
    pool_16_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    pool_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    pool_18_output_array.data = AI_PTR(g_network_activations_map[0] + 2184);
    pool_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2184);
    
    dense_19_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_19_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_19_output_array.data = AI_PTR(g_network_activations_map[0] + 16);
    dense_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16);
    
    dense_23_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 48);
    dense_23_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    
    dense_23_output_array.data = AI_PTR(g_network_activations_map[0] + 112);
    dense_23_output_array.data_start = AI_PTR(g_network_activations_map[0] + 112);
    
    dense_24_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 48);
    dense_24_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    
    dense_24_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_20_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 48);
    dense_20_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    
    dense_20_output_array.data = AI_PTR(g_network_activations_map[0] + 112);
    dense_20_output_array.data_start = AI_PTR(g_network_activations_map[0] + 112);
    
    dense_21_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_21_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_21_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    dense_21_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    
    dense_21_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_21_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_22_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    nl_22_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 640);
    conv2d_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 640);
    
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 672);
    conv2d_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 672);
    
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 864);
    conv2d_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 864);
    
    conv2d_13_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_weights_array.data = AI_PTR(g_network_weights_map[0] + 896);
    conv2d_13_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 896);
    
    conv2d_13_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_bias_array.data = AI_PTR(g_network_weights_map[0] + 1088);
    conv2d_13_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1088);
    
    dense_19_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_19_weights_array.data = AI_PTR(g_network_weights_map[0] + 1120);
    dense_19_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1120);
    
    dense_19_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_19_bias_array.data = AI_PTR(g_network_weights_map[0] + 1376);
    dense_19_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1376);
    
    dense_23_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_23_weights_array.data = AI_PTR(g_network_weights_map[0] + 1504);
    dense_23_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1504);
    
    dense_23_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_23_bias_array.data = AI_PTR(g_network_weights_map[0] + 2528);
    dense_23_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2528);
    
    dense_24_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_24_weights_array.data = AI_PTR(g_network_weights_map[0] + 2656);
    dense_24_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2656);
    
    dense_24_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_24_bias_array.data = AI_PTR(g_network_weights_map[0] + 2688);
    dense_24_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2688);
    
    dense_20_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_20_weights_array.data = AI_PTR(g_network_weights_map[0] + 2692);
    dense_20_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2692);
    
    dense_20_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_20_bias_array.data = AI_PTR(g_network_weights_map[0] + 3716);
    dense_20_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3716);
    
    dense_21_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_21_weights_array.data = AI_PTR(g_network_weights_map[0] + 3844);
    dense_21_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 3844);
    
    dense_21_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_21_bias_array.data = AI_PTR(g_network_weights_map[0] + 3940);
    dense_21_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3940);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1794334,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1794334,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

