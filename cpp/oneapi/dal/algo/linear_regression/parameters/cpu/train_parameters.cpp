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

#include <algorithm>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"

#include "oneapi/dal/algo/linear_regression/parameters/cpu/train_parameters.hpp"

namespace oneapi::dal::linear_regression::parameters {

using dal::backend::context_cpu;

template <typename Float>
std::int64_t propose_block_size(const std::int64_t f, const std::int64_t r) {
    constexpr std::int64_t fsize = sizeof(Float);
    std::int64_t proposal = 0x100l * (8 / fsize);
    return std::max<std::int64_t>(128l, proposal);
}

template <typename Float, typename Task>
struct train_parameters_cpu<Float, method::norm_eq, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        const auto& x_train = input.get_data();
        const auto& y_train = input.get_responses();

        const auto f_count = x_train.get_column_count();
        const auto r_count = y_train.get_column_count();

        const auto block = propose_block_size<Float>(f_count, r_count);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_input<Task>& input) const {
        const auto& x_train = input.get_data();
        const auto& y_train = input.get_responses();

        const auto f_count = x_train.get_column_count();
        const auto r_count = y_train.get_column_count();

        const auto block = propose_block_size<Float>(f_count, r_count);

        return params_t{}.set_cpu_macro_block(block);
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const partial_train_result<Task>& input) const {
        // const auto& x_train = input.get_data();
        // const auto& y_train = input.get_responses();

        // const auto f_count = x_train.get_column_count();
        // const auto r_count = y_train.get_column_count();

        const auto block = propose_block_size<Float>(100, 100);

        return params_t{}.set_cpu_macro_block(block);
    }
};
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::norm_eq, task::regression>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::parameters
