/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// NOTE(Zihao): this is the include file for AOT mode
#pragma once
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using DTypeQ = nv_bfloat16;
using DTypeKV = nv_bfloat16;
using DTypeO = nv_bfloat16;
using IdType = int32_t;

constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

#define ADDITIONAL_FUNC_PARAMS BATCH_MLA_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER BATCH_MLA_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    using Params = MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>; \
    __VA_ARGS__(); \
  })