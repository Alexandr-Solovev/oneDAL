/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

namespace mkl = oneapi::fpk;

template <mkl::job jobz, mkl::uplo ul, typename Float>
sycl::event syevd(sycl::queue& queue,
                  std::int64_t n,
                  Float* a,
                  std::int64_t lda,
                  Float* w,
                  Float* scratchpad,
                  std::int64_t scratchpad_size,
                  const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives