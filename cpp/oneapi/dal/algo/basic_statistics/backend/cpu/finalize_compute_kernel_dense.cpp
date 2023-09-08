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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/apply_weights.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/cpu/finalize_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/low_order_moments/moments_online.h>
#include <daal/src/algorithms/low_order_moments/low_order_moments_kernel.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_cpu;
using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_lom_online_kernel_t =
    daal_lom::internal::LowOrderMomentsOnlineKernel<Float, daal_lom::defaultDense, Cpu>;

template <typename Method>
constexpr daal_lom::Method get_daal_method() {
    return daal_lom::defaultDense;
}

template <typename Float>
std::int64_t propose_block_size(std::int64_t row_count, std::int64_t col_count) {
    using idx_t = std::int64_t;
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(col_count > 0);
    constexpr idx_t max_block_mem_size = 16 * 1024 * 1024;
    const idx_t block_of_rows_size = max_block_mem_size / (col_count * sizeof(Float));
    return std::max<idx_t>(std::min<idx_t>(row_count, idx_t(1024l)), block_of_rows_size);
}

template <typename Float>
array<Float> copy_immutable(const array<Float>&& inp) {
    if (inp.has_mutable_data()) {
        return inp;
    }
    else {
        const auto count = inp.get_count();
        auto res = array<Float>::empty(count);
        bk::copy(res.get_mutable_data(), inp.get_data(), count);
        return res;
    }
}

template <typename Float, typename Result, typename Input, typename Parameter>
void alloc_result(Result& result, const Input* input, const Parameter* params, int method) {
    const auto status = result.template allocate<Float>(input, params, method);
    interop::status_to_exception(status);
}

template <typename Float, typename Result, typename Input, typename Parameter>
void initialize_result(Result& result, const Input* input, const Parameter* params, int method) {
    const auto status = result.template initialize<Float>(input, params, method);
    interop::status_to_exception(status);
}

template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel_finalize_compute(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_compute_result<Task>& input) {
    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    auto column_numbers = input.get_partial_min().get_column_count();
    const auto nobs = oneapi::dal::row_accessor<const double>(input.get_nobs()).pull().get_data();
    auto row_count = nobs[0];

    auto daal_partial_obs = interop::convert_to_daal_table<Float>(input.get_nobs());
    auto daal_partial_min = interop::convert_to_daal_table<Float>(input.get_partial_min());
    auto daal_partial_max = interop::convert_to_daal_table<Float>(input.get_partial_max());
    auto daal_partial_sums = interop::convert_to_daal_table<Float>(input.get_partial_sum());
    auto daal_partial_sum_squares =
        interop::convert_to_daal_table<Float>(input.get_partial_sum_squares());
    auto daal_partial_sum_squares_centered =
        interop::convert_to_daal_table<Float>(input.get_partial_sum_squares_centered());

    auto daal_result = daal_lom::Result();

    auto daal_input = daal_lom::Input();
    auto arr_input = array<Float>::zeros(row_count * column_numbers);
    auto daal_input_ =
        interop::convert_to_daal_homogen_table<Float>(arr_input, row_count, column_numbers);
    daal_input.set(daal_lom::InputId::data, daal_input_);
    alloc_result<Float>(daal_result, &daal_input, &daal_parameter, result_ids);

    daal_result.set(daal_lom::ResultId::maximum, daal_partial_max);
    daal_result.set(daal_lom::ResultId::minimum, daal_partial_min);

    daal_result.set(daal_lom::ResultId::sum, daal_partial_sums);
    daal_result.set(daal_lom::ResultId::sumSquares, daal_partial_sum_squares);
    daal_result.set(daal_lom::ResultId::sumSquaresCentered, daal_partial_sum_squares_centered);

    interop::status_to_exception(
        interop::call_daal_kernel_finalize_compute<Float, daal_lom_online_kernel_t>(
            ctx,
            daal_partial_obs.get(),
            daal_partial_sums.get(),
            daal_partial_sum_squares.get(),
            daal_partial_sum_squares_centered.get(),
            daal_result.get(daal_lom::ResultId::mean).get(),
            daal_result.get(daal_lom::ResultId::secondOrderRawMoment).get(),
            daal_result.get(daal_lom::ResultId::variance).get(),
            daal_result.get(daal_lom::ResultId::standardDeviation).get(),
            daal_result.get(daal_lom::ResultId::variation).get(),
            &daal_parameter));

    auto result =
        get_result<Float, task_t>(desc, daal_result).set_result_options(desc.get_result_options());

    return result;
}

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    return call_daal_kernel_finalize_compute<Float, Task>(ctx, desc, input);
}

template <typename Float>
struct finalize_compute_kernel_cpu<Float, method_t, task_t> {
    compute_result<task::compute> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_compute_result<task::compute>& input) const {
        return finalize_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_cpu<float, method_t, task_t>;
template struct finalize_compute_kernel_cpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
