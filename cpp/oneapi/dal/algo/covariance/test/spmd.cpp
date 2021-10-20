// /*******************************************************************************
// * Copyright 2021 Intel Corporation
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *******************************************************************************/

// #include "oneapi/dal/algo/covariance/test/fixture.hpp"
// #include "oneapi/dal/test/engine/tables.hpp"
// #include "oneapi/dal/test/engine/io.hpp"

// namespace oneapi::dal::covariance::test {

// namespace te = dal::test::engine;
// namespace la = te::linalg;
// namespace bs = oneapi::dal::covariance;

// template <typename TestType>
// class covariance_spmd_test
//         : public covariance_test<TestType, covariance_spmd_test<TestType>> {
// public:
//     using base_t = covariance_test<TestType, covariance_spmd_test<TestType>>;
//     using float_t = typename base_t::float_t;
//     using input_t = typename base_t::input_t;
//     using result_t = typename base_t::result_t;

//     void set_rank_count(std::int64_t rank_count) {
//         rank_count_ = rank_count;
//     }

//     template <typename... Args>
//     result_t compute_override(Args&&... args) {
//         return this->compute_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
//     }

//     result_t merge_compute_result_override(const std::vector<result_t>& results) {
//         return results[0];
//     }

//     template <typename... Args>
//     std::vector<input_t> split_compute_input_override(std::int64_t split_count, Args&&... args) {
//         const input_t input{ std::forward<Args>(args)... };

//         const auto split_data =
//             te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

//         std::vector<input_t> split_input;
//         split_input.reserve(split_count);

//         for (std::int64_t i = 0; i < split_count; i++) {
//             split_input.push_back( //
//                 input_t{ split_data[i] });
//         }

//         return split_input;
//     }

//     void spmd_general_checks(const te::dataframe& data_fr,
//                              bs::result_option_id compute_mode,
//                              const te::table_id& data_table_id) {
//         CAPTURE(static_cast<std::uint64_t>(compute_mode));
//         const table data = data_fr.get_table(this->get_policy(), data_table_id);

//         const auto cov_desc = base_t::get_descriptor(compute_mode);

//         const auto compute_result = this->compute(cov_desc, data);

//         base_t::check_compute_result(compute_mode, data, compute_result);
//         //base_t::check_for_exception_for_non_requested_results(compute_mode, compute_result);
//     }

// private:
//     std::int64_t rank_count_;
// };

// using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

// TEMPLATE_LIST_TEST_M(covariance_spmd_test,
//                      "covariance common flow",
//                      "[covariance][integration][spmd]",
//                      covariance_types) {
//     SKIP_IF(this->get_policy().is_cpu());
//     SKIP_IF(this->not_float64_friendly());

//     const te::dataframe data =
//         GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 200, 530 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 500, 250 }.fill_normal(0, 1, 7777),
//                            te::dataframe_builder{ 6000, 20 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 6000, 530 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 10000, 200 }.fill_normal(-30, 30, 7777),
//                            te::dataframe_builder{ 1000000, 20 }.fill_normal(-0.5, 0.5, 7777));
//     this->set_rank_count(2);

//     // bs::result_option_id res_min_max = result_options::min | result_options::max;
//     // bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
//     // bs::result_option_id res_all = bs::result_option_id(dal::result_option_id_base(mask_full));

//     // const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

//     const auto data_table_id = this->get_homogen_table_id();

//     this->spmd_general_checks(data, compute_mode, data_table_id);
// }

// } // namespace oneapi::dal::covariance::test
