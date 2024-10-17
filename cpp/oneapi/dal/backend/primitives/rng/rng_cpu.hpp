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

#pragma once

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace oneapi::dal::backend::primitives {

template <engine_list EngineType = engine_list::mt2203>
class daal_engine {
public:
    explicit daal_engine(std::int64_t seed = 777)
            : daal_engine_(initialize_daal_engine(seed)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  daal_engine_.get())) {
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    virtual ~daal_engine() = default;

    void* get_cpu_engine_state() const {
        return impl_->getState();
    }

    auto& get_cpu_engine() {
        return daal_engine_;
    }

private:
    daal::algorithms::engines::EnginePtr initialize_daal_engine(std::int64_t seed) {
        switch (EngineType) {
            case engine_list::mt2203:
                return daal::algorithms::engines::mt2203::Batch<>::create(seed);
            case engine_list::mcg59: return daal::algorithms::engines::mcg59::Batch<>::create(seed);
            case engine_list::mt19937:
                return daal::algorithms::engines::mt19937::Batch<>::create(seed);
            default: throw std::invalid_argument("Unsupported engine type");
        }
    }

    daal::algorithms::engines::EnginePtr daal_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

template <typename Type, typename Size = std::int64_t>
class daal_rng {
public:
    daal_rng() = default;
    ~daal_rng() = default;

    void uniform_cpu(Size count, Type* dst, void* state, Type a, Type b) {
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }

    void uniform_without_replacement_cpu(Size count,
                                         Type* dst,
                                         Type* buffer,
                                         void* state,
                                         Type a,
                                         Type b) {
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }

    template <typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle_cpu(Size count, Type* dst, void* state) {
        Type idx[2];

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }
};

} // namespace oneapi::dal::backend::primitives
