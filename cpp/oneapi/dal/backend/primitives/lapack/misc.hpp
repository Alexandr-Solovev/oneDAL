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

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

namespace mkl = oneapi::fpk;

inline constexpr mkl::job ident_job(mkl::job order) {
    constexpr auto novec = mkl::job::novec;
    constexpr auto vec = mkl::job::vec;
    return (order == novec) ? novec : vec;
}

inline constexpr mkl::jobsvd ident_jobsvd(mkl::jobsvd order) {
    constexpr auto vectors = mkl::jobsvd::vectors;
    constexpr auto somevec = mkl::jobsvd::somevec;
    constexpr auto vectorsina = mkl::jobsvd::vectorsina;
    constexpr auto novec = mkl::jobsvd::novec;

    if (order == vectors)
        return vectors;
    else if (order == somevec)
        return somevec;
    else if (order == vectorsina)
        return vectorsina;
    else
        return novec;
}

} // namespace oneapi::dal::backend::primitives
