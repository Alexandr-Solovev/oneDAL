/* file: service_stat_ref.h */
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

/*
//++
//  Template wrappers for common statistic functions.
//--
*/

#ifndef __SERVICE_STAT_REF_H__
#define __SERVICE_STAT_REF_H__

typedef void (*func_type)(DAAL_INT, DAAL_INT, DAAL_INT, void *);

extern "C"
{
#define __DAAL_VSL_SS_MATRIX_STORAGE_COLS           0x00020000
#define __DAAL_VSL_SS_MATRIX_STORAGE_FULL           0x00000000
#define __DAAL_VSL_SS_ED_WEIGHTS                    6
#define __DAAL_VSL_SS_ED_MIN                        16
#define __DAAL_VSL_SS_ED_MAX                        17
#define __DAAL_VSL_SS_ED_SUM                        67
#define __DAAL_VSL_SS_ED_MEAN                       7
#define __DAAL_VSL_SS_ED_2R_MOM                     8
#define __DAAL_VSL_SS_ED_2C_MOM                     11
#define __DAAL_VSL_SS_ED_2C_SUM                     71
#define __DAAL_VSL_SS_ED_VARIATION                  18
#define __DAAL_VSL_SS_ED_CP                         74
#define __DAAL_VSL_SS_ED_CP_STORAGE                 75
#define __DAAL_VSL_SS_CP                            0x0000000100000000
#define __DAAL_VSL_SS_METHOD_FAST                   0x00000001
#define __DAAL_VSL_SS_METHOD_1PASS                  0x00000002
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN         0x00000100
#define __DAAL_VSL_SS_MIN                           0x0000000000000400
#define __DAAL_VSL_SS_MAX                           0x0000000000000800
#define __DAAL_VSL_SS_SUM                           0x0000000002000000
#define __DAAL_VSL_SS_MEAN                          0x0000000000000001
#define __DAAL_VSL_SS_2R_MOM                        0x0000000000000002
#define __DAAL_VSL_SS_2C_MOM                        0x0000000000000010
#define __DAAL_VSL_SS_2C_SUM                        0x0000000020000000
#define __DAAL_VSL_SS_VARIATION                     0x0000000000000200
#define __DAAL_VSL_SS_ED_ACCUM_WEIGHT               23
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN         0x00000100
#define __DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT      0x00000002
#define __DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT 0x00000001
#define __DAAL_VSL_SS_OUTLIERS                      0x0000000000080000
#define __DAAL_VSL_SS_METHOD_BACON                  0x00000020
#define __DAAL_VSL_SS_QUANTS                        0x0000000000010000
#define __DAAL_VSL_SS_ED_QUANT_ORDER_N              24
#define __DAAL_VSL_SS_ED_QUANT_ORDER                25
#define __DAAL_VSL_SS_ED_QUANT_QUANTILES            26
#define __DAAL_VSL_SS_SORTED_OBSERV                 0x0000008000000000
#define __DAAL_VSL_SS_ED_SORTED_OBSERV              78
#define __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE      79
#define __DAAL_VSL_SS_METHOD_RADIX                  0x00100000

#define __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER       -4022
#define __DAAL_VSL_SS_ERROR_INDICES_NOT_SUPPORTED -4085

    typedef void (*threadfuncfor)(DAAL_INT, DAAL_INT, void *, func_type);
    typedef void (*threadfuncforordered)(DAAL_INT, DAAL_INT, void *, func_type);
    typedef void (*threadfuncsection)(DAAL_INT, void *, func_type);
    typedef void (*threadfuncordered)(DAAL_INT, DAAL_INT, DAAL_INT, void *, func_type);
    typedef DAAL_INT (*threadgetlimit)(void);

    struct ThreadingFuncs
    {
        threadfuncfor funcfor;
        threadfuncfor funcforordered;
        threadfuncsection funcsection;
        threadfuncordered funcordered;
        threadgetlimit getlimit;
    };

    // static void _daal_mkl_threader_for_sequential(DAAL_INT n, DAAL_INT threads_request, void * a, func_type func)
    // {
    //     DAAL_INT i;

    //     for (i = 0; i < n; i++)
    //     {
    //         func(i, 0, 1, a);
    //     }
    // }

    // static void _daal_mkl_threader_for_ordered_sequential(DAAL_INT n, DAAL_INT threads_request, void * a, func_type func)
    // {
    //     DAAL_INT i;

    //     for (i = 0; i < n; i++)
    //     {
    //         func(i, 0, 1, a);
    //     }
    // }

    // static void _daal_mkl_threader_sections_sequential(DAAL_INT threads_request, void * a, func_type func)
    // {
    //     func(0, 0, 1, a);
    // }

    // static void _daal_mkl_threader_ordered_sequential(DAAL_INT i, DAAL_INT th_idx, DAAL_INT th_num, void * a, func_type func)
    // {
    //     func(i, th_idx, th_num, a);
    // }

    // static DAAL_INT _daal_mkl_threader_get_max_threads_sequential()
    // {
    //     return 1;
    // }

    // static void _daal_mkl_threader_for(DAAL_INT n, DAAL_INT threads_request, void * a, func_type func)
    // {
    //     //fpk_vsl_serv_threader_for(n, threads_request, a, func);
    // }

    // static void _daal_mkl_threader_for_ordered(DAAL_INT n, DAAL_INT threads_request, void * a, func_type func)
    // {
    //     //fpk_vsl_serv_threader_for_ordered(n, threads_request, a, func);
    // }

    // static void _daal_mkl_threader_sections(DAAL_INT threads_request, void * a, func_type func)
    // {
    //     //fpk_vsl_serv_threader_sections(threads_request, a, func);
    // }

    // static void _daal_mkl_threader_ordered(DAAL_INT i, DAAL_INT th_idx, DAAL_INT th_num, void * a, func_type func)
    // {
    //     //not used. To be implemented if needed.
    // }

    // static DAAL_INT _daal_mkl_threader_get_max_threads()
    // {
    //     return 1;
    // }
}

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct RefStatistics
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct RefStatistics<double, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xcp(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, double * sum, double * crossProduct,
                   __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxcp_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                           double * crossProduct, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxvar_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                            double * sampleVariance, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int x2c_mom(const double * data, const __int64 nFeatures, const __int64 nVectors, double * variance, const __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xoutlierdetection(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const double * baconParams, double * baconWeights)
    {
        int errcode = 0;

        return errcode;
    }

    static int xLowOrderMoments(double * data, __int64 nFeatures, __int64 nVectors, __int64 method, double * sum, double * mean,
                                double * secondOrderRawMoment, double * variance, double * variation)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSumAndVariance(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, __int64 method, double * sum,
                               double * mean, double * secondOrderRawMoment, double * variance)
    {
        int errcode = 0;

        return errcode;
    }

    static int xQuantiles(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const double * quantOrder,
                          double * quants)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSort(double * data, __int64 nFeatures, __int64 nVectors, double * sortedData)
    {
        int errcode = 0;

        return errcode;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct RefStatistics<float, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xcp(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, float * sum, float * crossProduct,
                   __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxcp_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean, float * crossProduct,
                           __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxvar_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean,
                            float * sampleVariance, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int x2c_mom(const float * data, const __int64 nFeatures, const __int64 nVectors, float * variance, const __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xoutlierdetection(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const float * baconParams, float * baconWeights)
    {
        int errcode = 0;

        return errcode;
    }

    static int xLowOrderMoments(float * data, __int64 nFeatures, __int64 nVectors, __int64 method, float * sum, float * mean,
                                float * secondOrderRawMoment, float * variance, float * variation)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSumAndVariance(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, __int64 method, float * sum,
                               float * mean, float * secondOrderRawMoment, float * variance)
    {
        int errcode = 0;

        return errcode;
    }

    static int xQuantiles(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const float * quantOrder,
                          float * quants)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSort(float * data, __int64 nFeatures, __int64 nVectors, float * sortedData)
    {
        int errcode = 0;

        return errcode;
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
