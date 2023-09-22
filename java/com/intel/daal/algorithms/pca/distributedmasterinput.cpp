/* file: distributedmasterinput.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_daal_algorithms_pca_Offline */

#include "daal.h"
#include "com_intel_daal_algorithms_pca_Method.h"
#include "com_intel_daal_algorithms_pca_DistributedStep2MasterInput.h"

#include "com/intel/daal/common_helpers.h"

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2MasterInput_cAddInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                jint id, jlong partialResultAddr, jint method)
{
    if (method == CorrelationDenseValue)
    {
        jniInput<DistributedInput<correlationDense> >::add<Step2MasterInputId, pca::PartialResult<correlationDense> >(inputAddr, id,
                                                                                                                      partialResultAddr);
    }
    else if (method == SVDDenseValue)
    {
        jniInput<DistributedInput<svdDense> >::add<Step2MasterInputId, pca::PartialResult<svdDense> >(inputAddr, id, partialResultAddr);
    }
}