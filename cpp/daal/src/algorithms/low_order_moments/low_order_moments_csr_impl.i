/* file: low_order_moments_csr_impl.i */
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

/*
//++
//  Low order moments of CSR data, computed from the non-zero values only
//--
*/

#ifndef __LOW_ORDER_MOMENTS_CSR_IMPL_I__
#define __LOW_ORDER_MOMENTS_CSR_IMPL_I__

#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
namespace csr
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

/// Per-feature moments of a set of rows, in double whatever the table type, as
/// scikit-learn's mean_variance_axis accumulates them.
template <CpuType cpu>
struct Moments
{
    Status init(size_t nFeatures)
    {
        sum.reset(nFeatures);
        sumSq.reset(nFeatures);
        sumSqCen.reset(nFeatures);
        mean.reset(nFeatures);
        min.reset(nFeatures);
        max.reset(nFeatures);
        DAAL_CHECK_MALLOC(sum.get() && sumSq.get() && sumSqCen.get() && mean.get() && min.get() && max.get());
        return Status();
    }

    size_t nRows = 0;
    TArray<double, cpu> sum;
    TArray<double, cpu> sumSq;
    TArray<double, cpu> sumSqCen;
    TArray<double, cpu> mean;
    TArray<double, cpu> min;
    TArray<double, cpu> max;
};

/// Running moments of one column inside a chunk, packed into one cache line so that a
/// stored value updates a single line.
enum ColumnSlot
{
    slotCount = 0,
    slotMean,     // the shift in the fast kernel
    slotM2,       // the sum of squared shifted values in the fast kernel
    slotShiftSum, // the sum of shifted values, fast kernel only
    slotMin,
    slotMax,
    slotsPerColumn = 8
};

/// Columns merged per task; at namespace scope so that MSVC accepts it as an array size in a lambda.
constexpr size_t columnsPerTask = 256;

/// Splits the rows into chunks with their own accumulators. The count does not depend on the
/// number of threads, and the chunks are merged in order, so the result is the same bit for bit.
///
/// @param[in]  nRows      Number of rows
/// @param[in]  nFeatures  Number of columns
/// @param[in]  nnz        Number of stored values
/// @return     Number of rows per chunk
inline size_t rowsPerChunk(size_t nRows, size_t nFeatures, size_t nnz)
{
    // A chunk initializes and merges slotsPerColumn values per column: 64 stored values per
    // column keep that under a few percent of its work, 2^15 keeps the task overhead small.
    constexpr size_t valuesPerColumn   = 64;
    constexpr size_t minValuesPerChunk = size_t(1) << 15;
    // Enough chunks to occupy any thread count; fixed so that the chunking stays the same.
    constexpr size_t maxChunks = 1024;

    const size_t byColumns = valuesPerColumn * nFeatures;
    const size_t perChunk  = byColumns > minValuesPerChunk ? byColumns : minValuesPerChunk;
    size_t nChunks         = nnz / perChunk;
    nChunks                = nChunks < 1 ? 1 : nChunks;
    nChunks                = nChunks > maxChunks ? maxChunks : nChunks;
    nChunks                = nChunks > nRows ? nRows : nChunks;
    return (nRows + nChunks - 1) / nChunks;
}

/// Merges the moments (cB, meanB, m2B) into (cA, meanA, m2A), Chan et al.; either side may be empty.
inline void mergeMoments(double & cA, double & meanA, double & m2A, double cB, double meanB, double m2B)
{
    const double c     = cA + cB;
    const double inv   = c > 0.0 ? 1.0 / c : 0.0;
    const double delta = meanB - meanA;
    meanA += delta * (cB * inv);
    // A select, not a reordered product: an empty side must not give inf * 0 under the fast FP model.
    const double weight = cA * cB * inv;
    m2A += m2B + (weight > 0.0 ? delta * delta * weight : 0.0);
    cA = c;
}

/// No row has touched the column yet in the per-column scratch of an unsorted row.
constexpr size_t noRow = ~size_t(0);

/// Calls `add(j, x)` once per column a row stores, `j` zero-based and `x` the sum of the row's
/// values of that column, as in the dense row. Rows with sorted indices sum the adjacent repeats;
/// the others go through a per-column scratch.
///
/// @param[in] block      Rows of the table with one-based indices
/// @param[in] rows       Number of rows in `block`
/// @param[in] nFeatures  Number of columns
/// @param[in] add        Receives every observation
/// @return    False if the scratch could not be allocated
template <typename algorithmFPType, CpuType cpu, typename Add>
bool forEachObservation(ReadRowsCSR<algorithmFPType, cpu> & block, size_t rows, size_t nFeatures, const Add & add)
{
    const algorithmFPType * const values = block.values();
    const size_t * const cols            = block.cols();
    const size_t * const rowOffsets      = block.rows();
    TArrayScalable<size_t, cpu> seenRow;
    TArrayScalable<double, cpu> pending;
    for (size_t r = 0; r < rows; r++)
    {
        const size_t begin = rowOffsets[r] - rowOffsets[0];
        const size_t end   = rowOffsets[r + 1] - rowOffsets[0];
        bool sorted        = true;
        for (size_t k = begin + 1; k < end && sorted; k++)
        {
            sorted = cols[k - 1] <= cols[k];
        }
        if (sorted)
        {
            for (size_t k = begin; k < end;)
            {
                const size_t j = cols[k];
                double x       = values[k];
                while (++k < end && cols[k] == j)
                {
                    x += values[k];
                }
                add(j - 1, x);
            }
            continue;
        }
        if (!seenRow.get())
        {
            seenRow.reset(nFeatures);
            pending.reset(nFeatures);
            if (!seenRow.get() || !pending.get()) return false;
            services::internal::service_memset_seq<size_t, cpu>(seenRow.get(), noRow, nFeatures);
        }
        for (size_t k = begin; k < end; k++)
        {
            const size_t j = cols[k] - 1;
            pending[j]     = (seenRow[j] == r) ? pending[j] + values[k] : double(values[k]);
            seenRow[j]     = r;
        }
        for (size_t k = begin; k < end; k++)
        {
            const size_t j = cols[k] - 1;
            if (seenRow[j] != r) continue;
            add(j, pending[j]);
            seenRow[j] = noRow;
        }
    }
    return true;
}

/// Precomputed column sums of `sumCSR`, read from the table's basic statistics.
template <typename algorithmFPType, CpuType cpu>
Status readUserSums(NumericTable * dataTable, size_t nFeatures, TArray<algorithmFPType, cpu> & sums)
{
    NumericTable * const sumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
    if (!sumsTable) return Status(ErrorPrecomputedSumNotAvailable);
    ReadRows<algorithmFPType, cpu> rows(sumsTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(rows);
    sums.reset(nFeatures);
    DAAL_CHECK_MALLOC(sums.get());
    const algorithmFPType * const src = rows.get();
    PRAGMA_OMP_SIMD
    for (size_t j = 0; j < nFeatures; j++)
    {
        sums[j] = src[j];
    }
    return Status();
}

/* One copy of the kernels per value of Parameter::estimatesToCompute, as for the dense methods */

namespace estimates_all
{
#define _MIN_ENABLE_
#define _MAX_ENABLE_
#define _SUM_ENABLE_
#define _SUM2_ENABLE_
#define _SUM2C_ENABLE_
#define _MEAN_ENABLE_
#define _SORM_ENABLE_
#define _VARC_ENABLE_
#define _STDEV_ENABLE_
#define _VART_ENABLE_

#include "src/algorithms/low_order_moments/low_order_moments_estimates_csr.i"

} // namespace estimates_all

namespace estimates_minmax
{
#define _MIN_ENABLE_
#define _MAX_ENABLE_
#undef _SUM_ENABLE_
#undef _SUM2_ENABLE_
#undef _SUM2C_ENABLE_
#undef _MEAN_ENABLE_
#undef _SORM_ENABLE_
#undef _VARC_ENABLE_
#undef _STDEV_ENABLE_
#undef _VART_ENABLE_

#include "src/algorithms/low_order_moments/low_order_moments_estimates_csr.i"

} // namespace estimates_minmax

namespace estimates_meanvariance
{
#undef _MIN_ENABLE_
#undef _MAX_ENABLE_
#undef _SUM_ENABLE_
#undef _SUM2_ENABLE_
#undef _SUM2C_ENABLE_
#define _MEAN_ENABLE_
#undef _SORM_ENABLE_
#define _VARC_ENABLE_
#undef _STDEV_ENABLE_
#undef _VART_ENABLE_

#include "src/algorithms/low_order_moments/low_order_moments_estimates_csr.i"

} // namespace estimates_meanvariance

} // namespace csr
} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
