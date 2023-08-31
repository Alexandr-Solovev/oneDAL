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
#include "oneapi/dal/algo/basic_statistics/backend/cpu/partial_compute_kernel.hpp"
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
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
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
result_t call_daal_kernel_with_weights(const context_cpu& ctx,
                                       const descriptor_t& desc,
                                       const partial_compute_input<Task>& input) {
    auto data = input.get_data();
    auto weights = input.get_weights();
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(weights.has_data());

    constexpr bool is_online = true;

    ONEDAL_ASSERT(weights.get_row_count() == data.get_row_count());
    ONEDAL_ASSERT(weights.get_column_count() == std::int64_t(1));

    auto daal_input = daal_lom::Input();
    auto daal_partial = daal_lom::PartialResult();

    auto result = partial_compute_result();
    const auto input_ = input.get_prev();

    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    //apply_weights(ctx, weights_ndarr, data_ndarr);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    daal_input.set(daal_lom::InputId::data, daal_data);
    const bool has_nobs_data = input_.get_nobs().has_data();
    if (has_nobs_data) {
        auto daal_partial_max =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_max());
        auto daal_partial_min =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_min());
        auto daal_partial_sums =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums());
        auto daal_partial_sums_squares =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums_squares());
        auto daal_partial_sums_squares_centered =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums_squares_centered());
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(input_.get_nobs());

        daal_partial.set(daal_lom::PartialResultId::nObservations, daal_nobs);

        daal_partial.set(daal_lom::PartialResultId::partialMaximum, daal_partial_max);
        daal_partial.set(daal_lom::PartialResultId::partialMinimum, daal_partial_min);
        daal_partial.set(daal_lom::PartialResultId::partialSum, daal_partial_sums);
        daal_partial.set(daal_lom::PartialResultId::partialSumSquaresCentered,
                         daal_partial_sums_squares_centered);

        daal_partial.set(daal_lom::PartialResultId::partialSumSquares, daal_partial_sums_squares);

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_lom_online_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       &daal_partial,
                                                                       &daal_parameter,
                                                                       is_online));

        result.set_nobs(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::nObservations)));
        result.set_partial_max(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMaximum)));
        result.set_partial_min(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMinimum)));
        result.set_partial_sums(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSum)));
        result.set_partial_sums_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered)));
        result.set_partial_sums_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquares)));

        return result;
    }
    else {
        alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
        initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_lom_online_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       &daal_partial,
                                                                       &daal_parameter,
                                                                       is_online));

        result.set_nobs(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::nObservations)));
        result.set_partial_max(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMaximum)));
        result.set_partial_min(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMinimum)));
        result.set_partial_sums(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSum)));
        result.set_partial_sums_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered)));
        result.set_partial_sums_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquares)));
        return result;
    }
}

template <typename Float, typename Task>
result_t call_daal_kernel_without_weights(const context_cpu& ctx,
                                          const descriptor_t& desc,
                                          const partial_compute_input<Task>& input) {
    auto data = input.get_data();
    ONEDAL_ASSERT(data.has_data());

    constexpr bool is_online = true;

    auto daal_input = daal_lom::Input();
    auto daal_partial = daal_lom::PartialResult();

    auto result = partial_compute_result();
    const auto input_ = input.get_prev();

    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    daal_input.set(daal_lom::InputId::data, daal_data);
    const bool has_nobs_data = input_.get_nobs().has_data();

    if (has_nobs_data) {
        auto daal_partial_max =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_max());
        auto daal_partial_min =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_min());
        auto daal_partial_sums =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums());
        auto daal_partial_sums_squares =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums_squares());
        auto daal_partial_sums_squares_centered =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sums_squares_centered());
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(input_.get_nobs());

        daal_partial.set(daal_lom::PartialResultId::nObservations, daal_nobs);

        daal_partial.set(daal_lom::PartialResultId::partialMaximum, daal_partial_max);
        daal_partial.set(daal_lom::PartialResultId::partialMinimum, daal_partial_min);
        daal_partial.set(daal_lom::PartialResultId::partialSum, daal_partial_sums);
        daal_partial.set(daal_lom::PartialResultId::partialSumSquaresCentered,
                         daal_partial_sums_squares_centered);

        daal_partial.set(daal_lom::PartialResultId::partialSumSquares, daal_partial_sums_squares);

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_lom_online_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       &daal_partial,
                                                                       &daal_parameter,
                                                                       is_online));

        result.set_nobs(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::nObservations)));
        result.set_partial_max(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMaximum)));
        result.set_partial_min(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMinimum)));
        result.set_partial_sums(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSum)));
        result.set_partial_sums_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered)));
        result.set_partial_sums_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquares)));

        return result;
    }
    else {
        alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
        initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_lom_online_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       &daal_partial,
                                                                       &daal_parameter,
                                                                       is_online));

        result.set_nobs(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::nObservations)));
        result.set_partial_max(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMaximum)));
        result.set_partial_min(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialMinimum)));
        result.set_partial_sums(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSum)));
        result.set_partial_sums_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered)));
        result.set_partial_sums_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_partial.get(daal_lom::PartialResultId::partialSumSquares)));
        return result;
    }
}

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_cpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    if (input.get_weights().has_data()) {
        return call_daal_kernel_with_weights<Float>(ctx, desc, input);
    }
    else {
        return call_daal_kernel_without_weights<Float, Task>(ctx, desc, input);
    }
}

template <typename Float>
struct partial_compute_kernel_cpu<Float, method_t, task_t> {
    partial_compute_result<task::compute> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_compute_input<task::compute>& input) const {
        return partial_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_cpu<float, method_t, task_t>;
template struct partial_compute_kernel_cpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
