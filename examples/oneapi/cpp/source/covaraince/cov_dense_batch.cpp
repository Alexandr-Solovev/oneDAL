/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    const auto input_file_name = get_data_path("covcormoments_dense.csv");
    const auto input = dal::read<dal::table>(dal::csv::data_source{ input_file_name });
    
    auto cov_desc = cov::descriptor<float>
        .set_result_options(covariance::result_options::cov_matrix);

    auto result = dal::compute(cov_desc, input);
    auto cov_matrix = result.get_cor();

    return 0;
}
