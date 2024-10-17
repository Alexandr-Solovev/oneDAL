/* file: kdtree_knn_classification_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Common functions for K-Nearest Neighbors predictions calculation
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/services/service_utils.h"
#include "algorithms/algorithm.h"
#include "services/daal_atomic_int.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_math.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/service_sort.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_model_impl.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_impl.i"
#include "src/algorithms/k_nearest_neighbors/knn_heap.h"

#if defined(DAAL_INTEL_CPP_COMPILER)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;
using namespace kdtree_knn_classification::internal;

template <typename algorithmFpType>
struct SearchNode
{
    size_t nodeIndex;
    algorithmFpType minDistance;
};

template <typename algorithmFpType, CpuType cpu>
DAAL_FORCEINLINE bool checkHomogenSOA(const NumericTable & data, services::internal::TArrayScalable<algorithmFpType *, cpu> & soa_arrays)
{
    if (data.getDataLayout() & NumericTableIface::soa)
    {
        if (static_cast<const SOANumericTable &>(data).isHomogeneousFloatOrDouble())
        {
            auto f = (*const_cast<NumericTable &>(data).getDictionary())[0];
            if (daal::data_management::features::getIndexNumType<algorithmFpType>() == f.indexType)
            {
                const size_t xColumnCount = data.getNumberOfColumns();
                soa_arrays.reset(xColumnCount);

                for (size_t i = 0; i < xColumnCount; ++i)
                {
                    soa_arrays[i] = static_cast<algorithmFpType *>(static_cast<SOANumericTable &>(const_cast<NumericTable &>(data)).getArray(i));
                    if (!soa_arrays[i])
                    {
                        return false;
                    }
                }

                return true;
            }
        }
    }
    return false;
}

template <typename algorithmFpType, CpuType cpu>
DAAL_FORCEINLINE const algorithmFpType * getNtData(const bool isHomogenSOA, size_t feat_idx, size_t irow, size_t nrows, const NumericTable & data,
                                                   data_management::BlockDescriptor<algorithmFpType> & xBD,
                                                   services::internal::TArrayScalable<algorithmFpType *, cpu> & soa_arrays)
{
    if (isHomogenSOA)
    {
        return soa_arrays[feat_idx] + irow;
    }
    else
    {
        const_cast<NumericTable &>(data).getBlockOfColumnValues(feat_idx, irow, nrows, readOnly, xBD);
        return xBD.getBlockPtr();
    }
}

template <typename algorithmFpType, CpuType cpu>
DAAL_FORCEINLINE void releaseNtData(const bool isHomogenSOA, const NumericTable & data, data_management::BlockDescriptor<algorithmFpType> & xBD)
{
    if (!isHomogenSOA)
    {
        const_cast<NumericTable &>(data).releaseBlockOfColumnValues(xBD);
    }
}

template <typename algorithmFpType, CpuType cpu>
Status KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::compute(const NumericTable * x, const classifier::Model * m,
                                                                                   NumericTable * y, NumericTable * indices, NumericTable * distances,
                                                                                   const daal::algorithms::Parameter * par)
{
    Status status;

    typedef GlobalNeighbors<algorithmFpType, cpu> Neighbors;
    typedef Heap<Neighbors, cpu> MaxHeap;
    typedef kdtree_knn_classification::internal::Stack<SearchNode<algorithmFpType>, cpu> SearchStack;
    typedef daal::services::internal::MaxVal<algorithmFpType> MaxVal;
    typedef daal::internal::MathInst<algorithmFpType, cpu> Math;

    size_t k;
    size_t nClasses;
    VoteWeights voteWeights       = voteUniform;
    DAAL_UINT64 resultsToEvaluate = classifier::computeClassLabels;

    const auto par3 = dynamic_cast<const kdtree_knn_classification::interface3::Parameter *>(par);
    if (par3)
    {
        k                 = par3->k;
        voteWeights       = par3->voteWeights;
        resultsToEvaluate = par3->resultsToEvaluate;
        nClasses          = par3->nClasses;
    }

    if (par3 == NULL) return Status(ErrorNullParameterNotSupported);

    const Model * const model       = static_cast<const Model *>(m);
    const KDTreeTable & kdTreeTable = *(model->impl()->getKDTreeTable());
    const auto rootTreeNodeIndex    = model->impl()->getRootNodeIndex();
    const NumericTable & data       = *(model->impl()->getData());
    const NumericTable * labels     = nullptr;
    if (resultsToEvaluate != 0)
    {
        labels = model->impl()->getLabels().get();
    }

    const NumericTable * const modelIndices = model->impl()->getIndices().get();

    size_t iSize = 1;
    while (iSize < k)
    {
        iSize *= 2;
    }
    const size_t heapSize = (iSize / 16 + 1) * 16;

    const size_t xRowCount        = x->getNumberOfRows();
    const algorithmFpType base    = 2.0;
    const size_t expectedMaxDepth = (Math::xsLog(xRowCount) / Math::xsLog(base) + 1) * __KDTREE_DEPTH_MULTIPLICATION_FACTOR;
    const size_t stackSize        = Math::xsPowx(base, Math::xsCeil(Math::xsLog(expectedMaxDepth) / Math::xsLog(base)));
    struct Local
    {
        MaxHeap heap;
        SearchStack stack;
    };
    SafeStatus safeStat;
    daal::tls<Local *> localTLS([&]() -> Local * {
        Local * const ptr = service_scalable_calloc<Local, cpu>(1);
        if (ptr)
        {
            if (!ptr->heap.init(heapSize))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if (!ptr->stack.init(stackSize))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                ptr->heap.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
        }
        else
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return ptr;
    });

    DAAL_CHECK_STATUS_OK((status.ok()), status);

    const auto maxThreads     = threader_get_threads_number();
    auto nThreads             = (maxThreads < 1) ? 1 : maxThreads;
    const size_t xColumnCount = x->getNumberOfColumns();
    const auto rowsPerBlock   = (xRowCount + nThreads - 1) / nThreads;
    const auto blockCount     = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;

    services::internal::TArrayScalable<algorithmFpType *, cpu> soa_arrays;
    bool isHomogenSOA = checkHomogenSOA<algorithmFpType, cpu>(data, soa_arrays);

    daal::threader_for(blockCount, blockCount, [&](int iBlock) {
        Local * const local = localTLS.local();
        DAAL_CHECK_MALLOC_THR(local);

        const size_t first = iBlock * rowsPerBlock;
        const size_t last  = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

        if (local)
        {
            const algorithmFpType radius = MaxVal::get();
            data_management::BlockDescriptor<algorithmFpType> xBD;
            const_cast<NumericTable &>(*x).getBlockOfRows(first, last - first, readOnly, xBD);
            const algorithmFpType * const dx = xBD.getBlockPtr();

            data_management::BlockDescriptor<int> indicesBD;
            data_management::BlockDescriptor<algorithmFpType> distancesBD;
            if (indices)
            {
                DAAL_CHECK_STATUS_THR(indices->getBlockOfRows(first, last - first, writeOnly, indicesBD));
            }
            if (distances)
            {
                DAAL_CHECK_STATUS_THR(distances->getBlockOfRows(first, last - first, writeOnly, distancesBD));
            }

            if (labels)
            {
                const size_t yColumnCount = y->getNumberOfColumns();
                data_management::BlockDescriptor<algorithmFpType> yBD;
                DAAL_CHECK_STATUS_THR(y->getBlockOfRows(first, last - first, writeOnly, yBD));
                auto * const dy = yBD.getBlockPtr();

                for (size_t i = 0; i < last - first; ++i)
                {
                    DAAL_CHECK_STATUS_THR(findNearestNeighbors(&dx[i * xColumnCount], local->heap, local->stack, k, radius, kdTreeTable,
                                                               rootTreeNodeIndex, data, isHomogenSOA, soa_arrays));
                    DAAL_CHECK_STATUS_THR(
                        predict(&dy[i * yColumnCount], local->heap, labels, k, voteWeights, modelIndices, indicesBD, distancesBD, i, nClasses));
                }
                DAAL_CHECK_STATUS_THR(y->releaseBlockOfRows(yBD));
            }
            else
            {
                for (size_t i = 0; i < last - first; ++i)
                {
                    DAAL_CHECK_STATUS_THR(findNearestNeighbors(&dx[i * xColumnCount], local->heap, local->stack, k, radius, kdTreeTable,
                                                               rootTreeNodeIndex, data, isHomogenSOA, soa_arrays));
                    DAAL_CHECK_STATUS_THR(predict(nullptr, local->heap, labels, k, voteWeights, modelIndices, indicesBD, distancesBD, i, nClasses));
                }
            }

            if (indices)
            {
                DAAL_CHECK_STATUS_THR(indices->releaseBlockOfRows(indicesBD));
            }

            if (distances)
            {
                DAAL_CHECK_STATUS_THR(distances->releaseBlockOfRows(distancesBD));
            }

            const_cast<NumericTable &>(*x).releaseBlockOfRows(xBD);
        }
    });

    status = safeStat.detach();
    if (!status) return status;

    localTLS.reduce([&](Local * ptr) -> void {
        if (ptr)
        {
            ptr->stack.clear();
            ptr->heap.clear();
            service_scalable_free<Local, cpu>(ptr);
        }
    });
    return status;
}

template <typename algorithmFpType, CpuType cpu>
DAAL_FORCEINLINE void computeDistance(size_t start, size_t end, algorithmFpType * distance, const algorithmFpType * query, const bool isHomogenSOA,
                                      const NumericTable & data, data_management::BlockDescriptor<algorithmFpType> xBD[2],
                                      services::internal::TArrayScalable<algorithmFpType *, cpu> & soa_arrays)
{
    // Initialize the distance array to zero for the range [start, end)
    for (size_t i = start; i < end; ++i)
    {
        distance[i - start] = 0;
    }

    size_t curBDIdx  = 0; // Current block descriptor index
    size_t nextBDIdx = 1; // Next block descriptor index

    const size_t xColumnCount = data.getNumberOfColumns(); // Total number of columns in the data

    const algorithmFpType * dx =
        getNtData(isHomogenSOA, 0, start, end - start, data, xBD[curBDIdx], soa_arrays); // Retrieve data for the first column

    // Iterate over each column to compute squared distances
    for (size_t j = 1; j < xColumnCount; ++j)
    {
        const algorithmFpType * nx =
            getNtData(isHomogenSOA, j, start, end - start, data, xBD[nextBDIdx], soa_arrays); // Retrieve data for the next column

        // Prefetch the next column data to optimize memory access
        DAAL_PREFETCH_READ_T0(nx);
        DAAL_PREFETCH_READ_T0(nx + 16); // Adjust prefetch based on expected access patterns

        // Compute distance contributions from the current column
        for (size_t i = 0; i < end - start; ++i)
        {
            distance[i] += (query[j - 1] - dx[i]) * (query[j - 1] - dx[i]);
        }

        // Release the current block of data to avoid memory leaks
        releaseNtData<algorithmFpType, cpu>(isHomogenSOA, data, xBD[curBDIdx]);

        // Swap block descriptors and pointers for the next iteration
        services::internal::swap<cpu, size_t>(curBDIdx, nextBDIdx);
        services::internal::swap<cpu, const algorithmFpType *>(dx, nx);
    }

    // Handle the last column after the loop
    for (size_t i = 0; i < end - start; ++i)
    {
        distance[i] += (query[xColumnCount - 1] - dx[i]) * (query[xColumnCount - 1] - dx[i]);
    }

    // Release the final block of data
    releaseNtData<algorithmFpType, cpu>(isHomogenSOA, data, xBD[curBDIdx]);
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::findNearestNeighbors(
    const algorithmFpType * query, Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap,
    kdtree_knn_classification::internal::Stack<SearchNode<algorithmFpType>, cpu> & stack, size_t k, algorithmFpType radius,
    const KDTreeTable & kdTreeTable, size_t rootTreeNodeIndex, const NumericTable & data, const bool isHomogenSOA,
    services::internal::TArrayScalable<algorithmFpType *, cpu> & soa_arrays)
{
    heap.reset();
    stack.reset();
    GlobalNeighbors<algorithmFpType, cpu> curNeighbor;
    size_t i;
    SearchNode<algorithmFpType> cur, toPush;
    const KDTreeNode * const nodes = static_cast<const KDTreeNode *>(kdTreeTable.getArray());
    const KDTreeNode * node;
    cur.nodeIndex   = rootTreeNodeIndex;
    cur.minDistance = 0;
    algorithmFpType distance[__KDTREE_LEAF_BUCKET_SIZE + 1];
    size_t start, end;

    data_management::BlockDescriptor<algorithmFpType> xBD[2];

    for (;;)
    {
        node = &nodes[cur.nodeIndex];
        if (node->dimension == __KDTREE_NULLDIMENSION)
        {
            start = node->leftIndex;
            end   = node->rightIndex;
            computeDistance<algorithmFpType, cpu>(start, end, distance, query, isHomogenSOA, data, xBD, soa_arrays);
            // for (i = start; i < end; ++i)
            // {
            //     if (distance[i - start] <= radius)
            //     {
            //         curNeighbor.distance = distance[i - start];
            //         curNeighbor.index    = i;
            //         if (heap.size() < k)
            //         {
            //             heap.push(curNeighbor, k);

            //             if (heap.size() == k)
            //             {
            //                 radius = heap.getMax()->distance;
            //             }
            //         }
            //         else
            //         {
            //             if (heap.getMax()->distance > curNeighbor.distance)
            //             {
            //                 heap.replaceMax(curNeighbor);
            //                 radius = heap.getMax()->distance;
            //             }
            //         }
            //     }
            // }

            if (!stack.empty())
            {
                cur = stack.pop();
            }
            else
            {
                break;
            }
        }
        else
        {
            algorithmFpType val        = query[node->dimension];
            const algorithmFpType diff = val - node->cutPoint;
            if (cur.minDistance <= radius)
            {
                cur.nodeIndex    = (diff < 0) ? node->leftIndex : node->rightIndex;
                toPush.nodeIndex = (diff < 0) ? node->rightIndex : node->leftIndex;
                val -= node->cutPoint;
                toPush.minDistance = cur.minDistance + val * val;
                stack.push(toPush);
            }
            else if (!stack.empty())
            {
                cur = stack.pop();
            }
            else
            {
                break;
            }
        }
    }
    return services::Status();
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::predict(
    algorithmFpType * predictedClass, const Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap, const NumericTable * labels, size_t k,
    VoteWeights voteWeights, const NumericTable * modelIndices, data_management::BlockDescriptor<int> & indices,
    data_management::BlockDescriptor<algorithmFpType> & distances, size_t index, const size_t nClasses)
{
    typedef daal::internal::MathInst<algorithmFpType, cpu> Math;

    const size_t heapSize = heap.size();
    if (heapSize < 1) return services::Status();
    SafeStatus safeStat;
    if (indices.getNumberOfRows() != 0)
    {
        DAAL_ASSERT(modelIndices);

        services::Status s;
        data_management::BlockDescriptor<int> modelIndicesBD;

        const auto nIndices = indices.getNumberOfColumns();
        DAAL_ASSERT(heapSize <= nIndices);

        int * const indicesPtr = indices.getBlockPtr() + index * nIndices;

        for (size_t i = 0; i < heapSize; ++i)
        {
            s |= const_cast<NumericTable *>(modelIndices)->getBlockOfRows(heap[i].index, 1, readOnly, modelIndicesBD);
            DAAL_ASSERT(s.ok());

            indicesPtr[i] = *(modelIndicesBD.getBlockPtr());

            s |= const_cast<NumericTable *>(modelIndices)->releaseBlockOfRows(modelIndicesBD);
            DAAL_ASSERT(s.ok());
        }
    }

    if (distances.getNumberOfRows() != 0)
    {
        services::Status s;

        const auto nDistances = distances.getNumberOfColumns();
        DAAL_ASSERT(heapSize <= nDistances);

        algorithmFpType * const distancesPtr = distances.getBlockPtr() + index * nDistances;
        for (size_t i = 0; i < heapSize; ++i)
        {
            distancesPtr[i] = heap[i].distance;
        }

        Math::xvSqrt(heapSize, distancesPtr, distancesPtr);

        for (size_t i = heapSize; i < nDistances; ++i)
        {
            distancesPtr[i] = -1;
        }
    }

    if (labels)
    {
        DAAL_ASSERT(predictedClass);

        data_management::BlockDescriptor<algorithmFpType> labelBD;
        algorithmFpType * classes =
            static_cast<algorithmFpType *>(daal::services::internal::service_malloc<algorithmFpType, cpu>(heapSize * sizeof(algorithmFpType)));
        DAAL_CHECK_MALLOC(classes)
        algorithmFpType * classWeights =
            static_cast<algorithmFpType *>(daal::services::internal::service_malloc<algorithmFpType, cpu>(nClasses * sizeof(algorithmFpType)));
        DAAL_CHECK_MALLOC(classWeights)

        for (size_t i = 0; i < nClasses; ++i)
        {
            classWeights[i] = 0;
        }

        for (size_t i = 0; i < heapSize; ++i)
        {
            const_cast<NumericTable *>(labels)->getBlockOfColumnValues(0, heap[i].index, 1, readOnly, labelBD);
            classes[i] = *(labelBD.getBlockPtr());
            const_cast<NumericTable *>(labels)->releaseBlockOfColumnValues(labelBD);
        }

        if (voteWeights == voteUniform)
        {
            for (size_t i = 0; i < heapSize; ++i)
            {
                classWeights[(size_t)(classes[i])] += 1;
            }
        }
        else
        {
            DAAL_ASSERT(voteWeights == voteDistance);

            const algorithmFpType epsilon = daal::services::internal::EpsilonVal<algorithmFpType>::get();

            bool isContainZero = false;

            for (size_t i = 0; i < heapSize; ++i)
            {
                if (heap[i].distance <= epsilon)
                {
                    isContainZero = true;
                    break;
                }
            }

            if (isContainZero)
            {
                for (size_t i = 0; i < heapSize; ++i)
                {
                    if (heap[i].distance <= epsilon)
                    {
                        classWeights[(size_t)(classes[i])] += 1;
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < heapSize; ++i)
                {
                    classWeights[(size_t)(classes[i])] += Math::sSqrt(1 / heap[i].distance);
                }
            }
        }

        algorithmFpType maxWeightClass = 0;
        algorithmFpType maxWeight      = 0;
        for (size_t i = 0; i < nClasses; ++i)
        {
            if (classWeights[i] > maxWeight)
            {
                maxWeight      = classWeights[i];
                maxWeightClass = i;
            }
        }
        *predictedClass = maxWeightClass;
        daal_free(classes);
        daal_free(classWeights);
    }

    return services::Status();
}

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
