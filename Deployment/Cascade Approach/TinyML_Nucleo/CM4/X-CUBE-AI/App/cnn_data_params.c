/**
  ******************************************************************************
  * @file    cnn_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Wed May 31 15:53:36 2023
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

#include "cnn_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_cnn_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_cnn_weights_array_u64[270] = {
  0x1b14360df0a4e981U, 0x5b26efcc4d7f382cU, 0x4540c67433aecdf6U, 0x263f6d7f3d7fffbbU,
  0x2c005f435483839U, 0x707f1c987f3d905dU, 0x1cdd4b3a182a25f2U, 0x3fbe81f735f2c839U,
  0xf81f34b4f77ffcc9U, 0xfffff97d000004bdU, 0xffffe89afffffc9cU, 0xffffe844fffffcb9U,
  0xfffffab4000002b0U, 0xd2e00baac3d7d781U, 0xd32f27bff623f8f3U, 0xf12df1dde2e1b4d3U,
  0xf10f19af4b005df0U, 0x3f4c30de4642065aU, 0xc2e00ee765ff0e0bU, 0x233c009a36fe142cU,
  0x1c11174959282304U, 0x568041345cc1ed1U, 0xf60af3269cd0ad2cU, 0xa761dc18a1533944U,
  0x23c0f2e0dc032c81U, 0x13bada47fad5f010U, 0x5756417a27fc0d11U, 0x52b2fa4bb4e8e906U,
  0x302fde0d20f71a26U, 0x15492e34c522074aU, 0xe73fe224cdeef26cU, 0xf96504c8d5efd2e4U,
  0xfbdad23f32c32315U, 0xc02fbc07fee00f16U, 0xe6df3fcee0ea0922U, 0x252d3ce61a425205U,
  0xe6c807edecc1ed3cU, 0x2fe142cc4c4df27fU, 0x48a46e1a5f154f1dU, 0x2781e90443b7f226U,
  0x11431a00db1a1075U, 0x1b300049f6dbe17fU, 0xfc22d34807e32620U, 0x1fd7e12df9e7e5f0U,
  0xdacbd531d9001a25U, 0xe315dc0405cf1b11U, 0x1539ed3feb1f2646U, 0x827dd3c14262147U,
  0xf055d60beefe040eU, 0xd3d2419537e52148U, 0x324efe28331e1304U, 0xf40021e204c83c16U,
  0xb042a8139dc375fU, 0xcacf1dec4b07561bU, 0xe9060a2913cddefbU, 0x122e128f1de71b4dU,
  0xc8310ff93208340aU, 0x14b6ed3ec6f0185eU, 0x1efd0441182f24ebU, 0x3c32122eecd936dfU,
  0xdc1a0721f244dadcU, 0xd716e713d846e302U, 0x3fb60a40fb0a0bffU, 0xf7f91624fe4301c8U,
  0xdaf9232de70d1a04U, 0x3906f2600b29d702U, 0xc70b233731faea81U, 0x1644f20400c8ed0cU,
  0xca3803e4c9cbba27U, 0xfa1ef9f50c09c3ffU, 0x24f9c73ee4bfd517U, 0x81f9f8d5d9d5dd0cU,
  0xfbe2fd1910e3ec24U, 0x29f4c612f098ec12U, 0xd1d20dd4ed00c0fbU, 0xe9f6fd1e2fd8d4f8U,
  0xde00235b2214fa2bU, 0x33e9f24cbfdb06cfU, 0xfb83c62a06d2bd81U, 0x24272d7d353cede4U,
  0x4a00350236f038d7U, 0xbee2ef01543d9c8U, 0xf2120b133b0f0aedU, 0x390cda3df1efdf01U,
  0xf3f433380a08d899U, 0x82fffffd6eU, 0x1c1fffffc5bU, 0xffffff1bfffffad8U,
  0xfffffa7600000091U, 0xf390d8e3d4fdf123U, 0xe522ad3738232c3bU, 0x17cad3feb4080427U,
  0xd7b309ffd5e11bf2U, 0x619bf420f244915U, 0xdff4fc29fbf98101U, 0xd627ebdab7eee91cU,
  0x12e0ed1ed60d19ccU, 0x121bc01bddc8daceU, 0xfa811f17d3401437U, 0xe54dcce3dd0612d6U,
  0xca2afffc30e6e722U, 0x29e8e4ff4a500dedU, 0xea29e3ff940cbd0cU, 0xdaa8e613ddbbabe7U,
  0x436a080973e65a47U, 0x394b15121b029abeU, 0xc964e7c600d71a2dU, 0xddf4c836ded0b815U,
  0xef1a02ecd5daf7c6U, 0xe3eaf606bff5e9abU, 0x12f9d80c1cf17ffaU, 0xef0ac9fa3843c73dU,
  0xeec839fa080509U, 0x1f22e4eb9e57d303U, 0x25250629370ede4bU, 0x271ad5040b0b11ccU,
  0xae0f7f9abe0f50aU, 0x17f20e2cbfb12efU, 0x2d340d240d1f2521U, 0xfcd918fde5eadaefU,
  0xf1f9de0620e938e3U, 0x20f1e316290ce713U, 0xd6fc08e7cbe8fa07U, 0xf0c7d0de1c203d19U,
  0x5f20808ee2a04f7U, 0x1517dfe28121d011U, 0x2edd2eeebe9de26U, 0xf82c13edd101b7f2U,
  0x40f61d172b2a35f8U, 0xfbbb0ff303076045U, 0x8c7e12a210a2e3aU, 0x6d0e0eae4ee5be9U,
  0xdffbdc08ed12dfedU, 0xd714f124eddc0ee5U, 0xf902f607c2f9e5f6U, 0xf407d715aaf89210U,
  0xf67f01ebd6efebd6U, 0xd9f7eae9f50c0110U, 0xd81fcef2061a0918U, 0x1524f8f2ee16c5d1U,
  0x446ff21e5f0a9efU, 0xfd07073510230021U, 0x17f5f30cfa01be0fU, 0x1a470effe85810f8U,
  0x227f130bfdc707acU, 0x1fc6f4d7f3fec700U, 0x4f8532dd65a693bU, 0xc42d2cdfe60ef6a5U,
  0xd5f4dd1fa9061b0cU, 0x5f7a770ec34198e6U, 0xaa60e31ea3c1fcbfU, 0xdd339cb6b01f18b8U,
  0x543636263b02541cU, 0xe9fc1af608ec0b2bU, 0x2e6d40e58cf176f8U, 0x2f8156d8fa0df0e3U,
  0x17ee2003d120a9ebU, 0x245f20dbdef42ff0U, 0x17d8ee1048f00a2fU, 0x25ac09f2fdf53bf8U,
  0xe4ead3210cbe1412U, 0xffffffb0ffffff02U, 0xffffffaafffffec5U, 0x70fffffe1bU,
  0xfffffda6fffffdabU, 0x7f5f7dc12e810e6U, 0xebe5dee5fae4f581U, 0x10271629c4003996U,
  0x3837d35723e7e83bU, 0x2cf8b5252437ad0fU, 0xf9e2ca361cdee709U, 0xd7d5edeec600a432U,
  0xdcdf6513b829f02cU, 0xd5ef17148720db91U, 0x2ee2c2c1d42ad5efU, 0xedbfdb00dfc93cffU,
  0xf931263c4aa1ff16U, 0x848c502cb28ba4dU, 0xea091821bc2e060cU, 0xfcf0eafe47c31125U,
  0x2c0282ddf2bdcff7U, 0xe1e2281be3993b11U, 0xfbd67f22d5753958U, 0xee34d90c2ac64b3aU,
  0xf016813b390c2533U, 0xcb37e15368f71376U, 0x1adc2aebfc1f0602U, 0xbf80af99febda0aU,
  0xcb2022dce31fbecbU, 0xff0afa211520cf23U, 0xc005672ce46d072cU, 0xdcd0fbf90645e013U,
  0xed05e93632bb2a5fU, 0xe917cb0df4811b7bU, 0xf63b10e902bfe63bU, 0xf9b7fa0804e5d708U,
  0xe4fb04fbcd19c9c3U, 0xedef25f6df0913beU, 0x165157ce20372208U, 0xee410cd11ff71c1bU,
  0xf7fd361b3cfa164aU, 0xa198090fd915880eU, 0xc9a5d5adba2c0640U, 0x45494803f5d45b1eU,
  0xf103070d090fb5d7U, 0x4dddc0dbef8ca7c6U, 0xf519ad49fcabfcd8U, 0xe38119794862efd8U,
  0x22dfdb31feb809d7U, 0xdefad3f409dfda43U, 0xd6c5c3fd2cdf5b3aU, 0x2f3359e54057eb57U,
  0xb80bf6dea8cae133U, 0xdac2ff00eb18f7e5U, 0xe417fd12ffe9f312U, 0xf60f05e7f8310904U,
  0x17eab8151cb9fbeaU, 0x2ad9c7efed17e581U, 0x41f4ec04a3d2c0cfU, 0x2ed1f9b8fef6f8fcU,
  0x1bd917c081dcfcfeU, 0xd5e40cd2e80401c5U, 0xed0733b6fcdde10cU, 0x7f823e4d5fff514U,
  0xfe0a070502f1e80dU, 0x393e16ea31f71ad7U, 0x411dbb04e3070ee1U, 0xf613ef1039eb0e10U,
  0xdf1e6b6f207c1bcU, 0xdd4e50bbce60020U, 0x48f1de0cfdb2da11U, 0x9ee21db31af8230fU,
  0xdef82bb2d6173fe8U, 0xd70c57d4f01c2fffU, 0x536518d129dd46fbU, 0x3c33a9e92ce90a86U,
  0x3d1d81eb2ce013d6U, 0x5bfffffe7fU, 0x25ffffff93U, 0xffffffef00000113U,
  0xffffff3bffffffd5U, 0xcad80babb7372048U, 0xbf811f09df3b0c50U, 0xc4db015cddf70e17U,
  0x1f5eeacf1205d22cU, 0x2e58ca453e0fdfdeU, 0x3107f00bfae62319U, 0xbb060be4dbeff903U,
  0xa9e4371e461de005U, 0xe9ccd303050a2df6U, 0x1219ed1543ce38f2U, 0x17ffcbc34ae7fae9U,
  0x6fe2bf45a08d1d4U, 0xd70d0e1107252bc6U, 0xeaf14114000b05a5U, 0xcef041190412db9fU,
  0xf81ff41f12fe0cfdU, 0x2c34faf01c0214f5U, 0xcd43818031ad42bU, 0x3c1a031e0bd090d1U,
  0x2b202825aad115baU, 0x120edfa6f4eb2914U, 0xbb5be2cacd90841U, 0xd70d03e4c7a71252U,
  0xe404eaeae0142f3aU, 0x24e4100af80bd1f9U, 0x2cf6eae6c8c7fa04U, 0xc478bbfdd30821e1U,
  0xffffff8cffffff7fU, 0xddU,
};


ai_handle g_cnn_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_cnn_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

