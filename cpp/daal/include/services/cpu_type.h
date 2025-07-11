/* file: cpu_type.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#ifndef __CPU_TYPE_H__
#define __CPU_TYPE_H__

namespace daal
{
/**
 * <a name="DAAL-ENUM-CPUTYPE"></a>
 * Supported types of processor architectures
 */
enum CpuType
{
#if defined(TARGET_X86_64)
    sse2        = 0, /*!< Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2) */
    sse42       = 2, /*!< Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) */
    avx2        = 4, /*!< Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) */
    avx512      = 6, /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    lastCpuType = avx512
#elif defined(TARGET_ARM)
    sve         = 0, /*!< ARM(R) processors based on Arm's Scalable Vector Extension (SVE) */
    lastCpuType = sve
#elif defined(TARGET_RISCV64)
    rv64        = 0,
    lastCpuType = rv64
#endif
};

/**
 * Supported CPU features.
 * The features are defined as bit masks in order to allow for easy combination of features.
 * For example, (avx512_bf16 | avx512_vnni) will return a bit mask that indicates both the avx512_bf16
 * and avx512_vnni features are supported.
 * This allows for easy checking if a specific feature is supported by using a bitwise AND operation.
 * For example, (cpuFeatures & avx512_bf16) will return true if the avx512_bf16 feature is supported.
 */
enum CpuFeature
{
    unknown = 0ULL, /*!< Unknown features */
#if defined(TARGET_X86_64)
    sstep       = (1ULL << 0), /*!< Intel(R) SpeedStep */
    tb          = (1ULL << 1), /*!< Intel(R) Turbo Boost */
    avx512_bf16 = (1ULL << 2), /*!< AVX-512 bfloat16 */
    avx512_vnni = (1ULL << 3), /*!< AVX-512 Vector Neural Network Instructions (VNNI) */
    tb3         = (1ULL << 4), /*!< Intel(R) Turbo Boost Max 3.0 */
#endif
};
} // namespace daal
#endif
