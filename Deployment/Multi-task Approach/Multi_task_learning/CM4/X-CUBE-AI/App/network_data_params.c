/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sat Jul  1 10:57:38 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[494] = {
  0xb7325752e6d5f5c6U, 0x9cd034f43d574381U, 0xf95224f0dc49e2f7U, 0xbc0b5ec94e091fcaU,
  0xf10df9b7d40ecfddU, 0x2ccdc02a17abd528U, 0x44ecaa48252ed372U, 0xdfd9fb42f6e44cdeU,
  0x3647fc36535ce2c9U, 0x5cede0acd89eccf9U, 0x9ff3b2b06f4f4b3U, 0x3f2decfc65bdfd9U,
  0xf3eafeb7cac5de02U, 0xd6b0cfbce8f2d303U, 0xf6e80220fee12936U, 0x6cd0d290c36e911U,
  0x2f19c8c24de8e81eU, 0xad0c18271aeeddd8U, 0xc0ce3e22071b01e9U, 0xf018f5c4bdbbc181U,
  0x10ede42330eed927U, 0x48f2b5a49ebd2128U, 0xf40528bbf645437fU, 0x385e57ebdfb1b5acU,
  0x365c28eeb9aff238U, 0x8b6f40ccfce02f6U, 0x1c1205e8f9dfc3fbU, 0x26160ffe28100f16U,
  0x361ee301212eedf8U, 0x718210a271f150aU, 0xf34d55424624efc3U, 0x7f05cdd60096ecf8U,
  0xf81018dc2d1ab22cU, 0x1313e4e72c3ac9faU, 0x203f2af9cb2034a4U, 0xec2c01c5f856212cU,
  0xc409b80309be31f7U, 0x8df82c8a5525c9e4U, 0xee0238fe0403301bU, 0xfa0902fab7cacbd2U,
  0xe29ac30ca8da10d7U, 0xd0e0b6ffd7df1cb6U, 0xd436fbff04da2619U, 0x15ddbc35e16ef16U,
  0x9ecbdcc9220df7e7U, 0xf9bdbfbe81abb5aaU, 0x591de3362a0ebbdcU, 0x49312abc2f509c64U,
  0x6a19c0e6fb559bd2U, 0x3ecfe601a283644U, 0x41ceff5226c2b848U, 0x2dedfbfcc9ca3145U,
  0x9436f40d0f14602cU, 0xffff07c9a3e10cceU, 0x106b28707f67604fU, 0xbb8de0e2ab011719U,
  0xb24f304bde1243c7U, 0xaad22fa2c00a0202U, 0xc1994c01ecfbed7U, 0x1e2a0342c544bcbdU,
  0x584a5f45241ee4c5U, 0xfc3b09ddbfa1b6e1U, 0xd0f31613fc0ed8f9U, 0x492ffebeb3cbcbcdU,
  0x2034c2e9cbde1d43U, 0x1a0a0fe203f9fc15U, 0x6dddbf21a2bf9e6U, 0x262716e50081e101U,
  0xddd0d1cd04150fd2U, 0x2bf4edf0301e3b21U, 0xe5d914501de4d195U, 0x2c482a0cd29db79cU,
  0xd0d3c7d9aac10ddeU, 0xb4c92c4e3c04f6dbU, 0xe100675d7f46cf9aU, 0xc9eaf4f5ddb3b78bU,
  0x133a0f34fec2a1aeU, 0xfb06afd5c80a383eU, 0xd0e4f44438274222U, 0x910e21c322afae5U,
  0xffffff460000016fU, 0xffffff9bffffff6bU, 0xfffffe1fffffffabU, 0xfffffceffffffdcaU,
  0x25f2a7c3fbc4d474U, 0x4399c381dcc4fd20U, 0xf3d4ff1e0994e7bU, 0x672a82c357f328fbU,
  0x3233dbcdf4533724U, 0xfe67acafe3ca81b1U, 0x7de77f5f2a55e9b3U, 0x6604d6d754cc40dbU,
  0x79bf215fc82e11e9U, 0xd5212513f164a28U, 0xb0e4c574c4a49ee2U, 0x9cc7063d6c81af6dU,
  0xb312e60a64127bb7U, 0x4a1209c17f8a9d75U, 0x2bf44aec58025cb9U, 0xb91d5bcba73d6eadU,
  0x813432e8d6d123bcU, 0xb26925a34bececdfU, 0x1e00fcd3cd8d6a56U, 0xd4f79fac50810f53U,
  0x4b994bc34248b413U, 0xb8ac60443fe671cU, 0xac0ecc563835014eU, 0x42d8441539814de0U,
  0xffffff4dfffffe9dU, 0x30fffffdd7U, 0xfffffe8dfffffefdU, 0x3c1ffffffe2U,
  0x291813220c798881U, 0x4fb220ea2b06c8b0U, 0xde49bfbd3572a094U, 0x50e6e6c6270681d7U,
  0x50cdd800e1c59de4U, 0x554a10d7e5add4daU, 0x432163d7bf68f1efU, 0xff25d5fab229e099U,
  0x28e5f5f3810e6717U, 0x1e201d01a65b5836U, 0xda2905af37a77fcaU, 0x42a45625b8e01103U,
  0x5c23a5e7c1eddfdbU, 0xce379bf4f3ac6a4cU, 0x43d50160500e7fe5U, 0x487f9c3c0ab7ce78U,
  0xd75f8f26dd0d592aU, 0x56edba196510331eU, 0x4ed1d8be9643df1bU, 0x62c4fc290dda8e81U,
  0xf3a0804170db59eU, 0x390f14f9d92ab9ccU, 0x6f8f3032911c6b9U, 0xf13de15f91d8132U,
  0x29cffffffabU, 0xffffff05ffffff65U, 0x90000000a3U, 0xffffff34ffffff72U,
  0x2d280abb381fc8e8U, 0xe2f0dcfd1016cee1U, 0xc8c8db150731e4eaU, 0x164b01bd1b149a25U,
  0xa36e3dedf3c4ad1U, 0x26e7f22b25266de4U, 0xfa192c50e2215ee6U, 0xb5d638d4e72d4438U,
  0x17302652fcced740U, 0xdf3beb1cfed3bdc9U, 0xf5363e2424238538U, 0x422fd2fcabf50706U,
  0x8dfd8c9ed37e6deU, 0xc61bc9e929fcf5dcU, 0x470140d64a48da21U, 0xd930dfee474ccc48U,
  0xc3fbfe05d1efcc10U, 0xba064109afcd7f1cU, 0xdcdddf07d1edd3f6U, 0x3d383b15dec80ff4U,
  0xfe52dcdce2ecba46U, 0xcde1c102ff113cefU, 0xd644c824d155da2cU, 0xf5d2ff26fbc92436U,
  0xd7113f20b4ec63d4U, 0x4b13c5d943038b41U, 0xcf3dad31275e001eU, 0xd331c2d2c3c62cfdU,
  0x2d49e9c9def44aedU, 0x9d3e5d3165ddb2eU, 0xe13354919b14fdcU, 0x7e1da0d0501f9f7U,
  0xfffffd70U, 0xfffffe51fffffea0U, 0x7ec000007c4U, 0xdda0000028cU,
  0xffffff7bfffffff1U, 0x434fffff015U, 0x0U, 0x9afffff92cU,
  0x98d00000000U, 0xfffffb3b00000000U, 0x3d5U, 0x396ffffff09U,
  0xffffff0900000701U, 0xcb4U, 0x3a7000001bbU, 0x426U,
  0xe612e4dfeae323e9U, 0xfb0f1309f2f4e4ddU, 0x24251a0d0eddf1f2U, 0xe2f8f30fe6dc1b23U,
  0x13eddf0e1ddd0e3cU, 0x2f24e3f2640325ebU, 0x102dfc6a2005de12U, 0x1e01b4feefc27d3U,
  0x3f1e3f1dff51e17U, 0x16eb08f914070003U, 0xe8f0dbffe2dff80fU, 0xdff5f51e1dddefe7U,
  0xe7dd1b031325deebU, 0xecf8fe0f1c0e0aefU, 0xeedef316fb1a0924U, 0xf900e0e5fbf51707U,
  0xc182310fb19e1d2U, 0xecf3011ac1dafd1aU, 0x250fde90061f04f5U, 0xa26c0ca1a07d42cU,
  0x25f9fbf33521072eU, 0x162425f948eb18ebU, 0xe33ff028f5f8e5ddU, 0xf3ea1a391e2a28f4U,
  0xee1fec1d0cfefa32U, 0x121317e17024f203U, 0xe2f50a620208d60cU, 0x1ff3183b011333f2U,
  0xe3ee10230d21dd43U, 0x15201df836260effU, 0xfbfee065f5dff320U, 0x1cda394cf0fd26d2U,
  0xedee100535ed072dU, 0x6e7def8481a2522U, 0x212bfa430bfbcf0dU, 0xde03345ff061fe7U,
  0xe0dc0b07e617ebebU, 0x11ea1c150bf904f0U, 0xe61be3dd160df0e1U, 0x21e91703fddfefdfU,
  0x24fef1f1141504eaU, 0xe2e11df407dc0ddeU, 0x1609ec13db210c21U, 0x1be721f1fc08fbfaU,
  0xbe61638f224080dU, 0xddf6e50243fafdf1U, 0xf3e72210191d26eaU, 0x23091a36e02013fbU,
  0xeedef0e8ee07e2e8U, 0xfaef0ffbe6e8e316U, 0xfcf4ea0be705fa21U, 0x20e816f90901e3f8U,
  0xe80c1be3ca06f3e7U, 0xda22140dc7e51022U, 0x1b10fada16f5170aU, 0x3ee04caecedbe06U,
  0x25002d0adf00eed2U, 0xb010e15acf4e1ebU, 0x2bd709d6fa1910dbU, 0xed051cdedeffdd25U,
  0xfaf90804e20518dbU, 0xe20b25f5e8e300f7U, 0xcf80ae8091fdd10U, 0xf0191fea2119e7f5U,
  0xe21ee7f2e1ff20f5U, 0x2f6eb2107e11210U, 0xdd031502dcf8fbfaU, 0xed13f310dbf90d06U,
  0x19da1ce212fd21U, 0xf1ecdc1f1714e8fdU, 0xbf5fe25efea011dU, 0x6ff251c13e8f802U,
  0x270519eedcfaffeeU, 0xded0c17b106061dU, 0x18d3eeafe1dd31f4U, 0x152d02af0fdfea3eU,
  0x10f1e8ecdaf4de1eU, 0xddfce9dc04e3001cU, 0xa15030ddf2120f3U, 0x1aebe5061821f7e8U,
  0x2bfb16d9fbefeeb8U, 0xe3f4ebfac4e2e214U, 0x34db04b1f6253e20U, 0x1015ccd303190137U,
  0xebf40cf40b0e141fU, 0x20fddc0512fee9f2U, 0xce3ede4e22205f7U, 0xf31ee4ebed03f1fbU,
  0x1e190f2cf3f5ef18U, 0xece0e217f2dc04faU, 0x1813ec0bf7e8f821U, 0xbf4e8fae7fad4feU,
  0xf103d4173702e219U, 0x18ef0ade5cfee423U, 0x634dc522209dcffU, 0x9050455f00a37b9U,
  0xd8ed1d240512f013U, 0xe1e800f2ead80418U, 0x1f10041c0c0e1fU, 0x1209e3e3f6f419f0U,
  0xd4ede0ee2fece92fU, 0xf12324213a0df816U, 0x1d13124e0a1df5e3U, 0x14d93827f9f435cdU,
  0x1c1823a9fdf9e4caU, 0xe0f5e105a010f9efU, 0x1cf803b4e618210cU, 0xf134b9bf05f1d718U,
  0xf81b100beb18ddddU, 0x21f715e3a8e0deecU, 0x2ff6eae104e12ddeU, 0x181818d1f1f7e017U,
  0xd32fce012df0004U, 0xd4eff00694ed070aU, 0xe6d205ad040b2c0fU, 0xd32e281fafef71bU,
  0xf601e904d3090e04U, 0xde1e06feb8030d27U, 0x2df8e2c6090528dfU, 0x1517d9c9e2dee93fU,
  0xe5e412e52bf1e913U, 0x1817f20847e7041fU, 0xef12f3640418ed05U, 0x10e83a25eb2f28cdU,
  0x21190cd6fa2321e4U, 0xe0ffff14aaf2eff7U, 0x24d7e8b31802061fU, 0x81ad387e7d20442U,
  0xffffffaa00000000U, 0x0U, 0x16000004c1U, 0xfffffeacfffffe0cU,
  0xffffff07U, 0x3e600000000U, 0x48d00000000U, 0xffffffa900000239U,
  0xfffffe45fffffec6U, 0xffffff21000004f8U, 0xffffff6a0000069dU, 0xfffffe8b0000010cU,
  0xfffffcdafffffe19U, 0x4b6000003bfU, 0x4de000003fdU, 0x4aefffffd2aU,
  0x98c6b05c11c59d01U, 0xdf4e2230c7ca0695U, 0xa343d05ffe51cce0U, 0x60be3f7f4e61abcfU,
  0xe8fc22f50000064eU, 0xf2eaf5f80e1d23fbU, 0xe41f14f71dedf821U, 0xf845d7eeeef218e0U,
  0xed1b08dd18fd24dcU, 0xe4c6dd13f9f1043dU, 0xeaf30f0d0a0f20f1U, 0x213ee41ed926ddeaU,
  0xf0f8e0cc1207fefaU, 0xdf28ddf9071cf1c6U, 0x291b1615fb1c1af7U, 0xe2c7013eebe5ebefU,
  0xe6e90c30f731d0baU, 0x20fcfff4f7f70a4dU, 0x121e00fc3011e8efU, 0xec4203061819200aU,
  0xd7e6fc0508cb3e42U, 0xebd1e0ed180c29e6U, 0x4f8340cf6f30509U, 0xd1b0132280afbdaU,
  0xf620153c0df81ee8U, 0x3d13dd12fadc02dfU, 0xd1b0b0bf32ceb0aU, 0xeb0532100eeff754U,
  0x16df0b0823fe2021U, 0x16f5ea12efe40e0fU, 0xed0cea01e8f316f1U, 0xdbe8edf3e008f6e8U,
  0x31e41a451e1df9e5U, 0x590dfe0ff2f0002cU, 0xef0af9ed26ff0defU, 0xf5332eeaf5361459U,
  0x14faf0301fda1e33U, 0x39210e080117f0ceU, 0x924eff5e4f60601U, 0xe2d9f51814ee1f0aU,
  0xdee5f8c21903dffaU, 0xb01be907033518c8U, 0x281d2bf40bef060fU, 0xe9fae02514d0f4e0U,
  0x170024e21501eeccU, 0xfdfb1109f4dee9faU, 0x618fa1f08e1e0eaU, 0x20f6e4ef03051914U,
  0xe206fbe416ef06fdU, 0xed0ff502f4e6e70fU, 0xf407f8f0f2f40800U, 0xe4f8fa0f0b010714U,
  0xd1f7e7e804090710U, 0x9dfe041b1ffdf2e5U, 0x191214180fdddf03U, 0x602e320200aebb8U,
  0xda0efbd71500f6a0U, 0xb5e700e01a250610U, 0xaef2bfb0afae1ffU, 0x112103f80c16e7a7U,
  0xe410ec07040a19bcU, 0xe7befbea1be42231U, 0xe6e41edef30b1cf3U, 0x1d74c6f7d85ae2e3U,
  0x2f7f3480c0a2cfeU, 0x4d02e10924fdf731U, 0xdd24fd12280ef5e8U, 0xff2b2ec71323dd35U,
  0x9eb0208ebd6454fU, 0x1906e31d110f08d0U, 0x12e61b000524f90fU, 0xefbb2dd90106f32cU,
  0xf2e2f0f4dc29e504U, 0xc80ff512ef261aa4U, 0x35fb0715ca16f7f6U, 0x4d8e0fefec7f3aeU,
  0xf4fef8f0e223caceU, 0xa6f1e516ed1b21e6U, 0x2814f504011ef3e6U, 0x1bc6d73a1403e6edU,
  0x10be8d21e36fdb1U, 0xddf0f21c02150509U, 0x151bf51d10f1ed11U, 0xe82bc631e9df1aacU,
  0xb0a1a1aec160abeU, 0xf7e2f2dd04e80b15U, 0xae2dcf31be7e8deU, 0xe7e301e30becdcfcU,
  0xe9141b0a00fee624U, 0xf9fa16f01512f205U, 0x10e61a1120fd0a12U, 0xd0dfa14e4f51fe7U,
  0xece3f8ebe6eafff2U, 0xd7300027f0020a81U, 0x30f0e802d729de19U, 0xb8bf513f5c2eceeU,
  0xde0b22f30f17e5c6U, 0xf5f5e4dc00ddff01U, 0xf4e308f2e7eee222U, 0x1afdf801db101ddfU,
  0x19150df11bef1cffU, 0x70ee304ed08090eU, 0xf6dfde1ffce2e0fdU, 0xf71610feebfc16e5U,
  0x2322fc06f103f60bU, 0x4aeb0222e408e42dU, 0xe6fed318fa28fff7U, 0xf70211d7df34ff3bU,
  0xe9ff11fbf2e83262U, 0xeee10fcdcf6f108U, 0xcebff0508ed24fcU, 0xf4e5121b15f6e122U,
  0x221302041bf6220cU, 0x29ec05ff160b0722U, 0xe30dd41f30250f13U, 0x1a17220ff41ef33aU,
  0x825e6fadde02620U, 0x1ff6121808e1eef8U, 0xe9e4dd08f1e51e25U, 0x21f709e3e70b14eeU,
  0xe60615efef0e1d1eU, 0x91ceb1fe9fedef3U, 0xe82104ebdfe61803U, 0x3f8e8e7f200dfe1U,
  0x1c08ddeeed17f615U, 0xe10ff1fb00f3df1eU, 0xf7e7e5f512e9dddbU, 0xf70808feebe920e3U,
  0x14e7ed21dd0c1ae7U, 0x5807eaf519e30f42U, 0xf403ff160e1af811U, 0xfb2d2ccae61beb29U,
  0x9f210f03f65U, 0xfffffa9e00000964U, 0xa840000055eU, 0xfffff8cdU,
  0xfffffbfa0000011eU, 0x40U, 0x97000000000U, 0x9d00000077eU,
  0xfffff7670000020dU, 0x3060000004cU, 0x6efU, 0xfffff669fffffea8U,
  0xffffff5000000000U, 0x3daU, 0x2b7U, 0xffffff07U,
  0x19dce3d30000022aU, 0x6e8c90d47e911c9U, 0xcfdcf51c392cf8aeU, 0x3b0d3604ec070ae6U,
  0x1fdf28403cea22fbU, 0xe8dff9c0e222f22eU, 0x2f0010def360211dU, 0x7e2e00c00c515eaU,
  0xe115d1cbed01ec16U, 0xede11effc2dc08eaU, 0xa0f4328c5810af1U, 0xbc18a81e1f4c11e8U,
  0xfffffd56c21bfc07U, 0xfffffb64000006f6U,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

