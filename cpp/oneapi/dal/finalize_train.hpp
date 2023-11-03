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

#include "oneapi/dal/detail/finalize_train_ops.hpp"
#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/spmd/communicator.hpp"

namespace oneapi::dal {
namespace v1 {

template <typename... Args>
auto finalize_train(Args&&... args) {
    return dal::detail::finalize_train_dispatch(std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
auto finalize_train(sycl::queue& queue, Args&&... args) {
    return dal::detail::finalize_train_dispatch(detail::data_parallel_policy{ queue },
                                                std::forward<Args>(args)...);
}
#endif

} // namespace v1

using v1::finalize_train;

namespace preview {

template <typename... Args>
auto finalize_train(spmd::communicator<spmd::device_memory_access::none>& comm, Args&&... args) {
    return dal::detail::finalize_train_dispatch(
        dal::detail::spmd_policy{ dal::detail::host_policy{}, comm },
        std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
auto finalize_train(spmd::communicator<spmd::device_memory_access::usm>& comm, Args&&... args) {
    return dal::detail::finalize_train_dispatch(
        dal::detail::spmd_policy<dal::detail::data_parallel_policy>{
            dal::detail::data_parallel_policy{ comm.get_queue() },
            comm },
        std::forward<Args>(args)...);
}
#endif

} // namespace preview

} // namespace oneapi::dal
