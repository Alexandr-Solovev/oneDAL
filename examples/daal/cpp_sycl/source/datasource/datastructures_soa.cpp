/* file: datastructures_soa.cpp */
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
!  Content:
!    C++ example of using a structure of arrays (SOA)
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_SOA">
 * \example datastructures_soa.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace daal;

using daal::services::internal::SyclExecutionContext;
using daal::data_management::internal::SyclSOANumericTable;
using daal::data_management::internal::SyclSOANumericTablePtr;

const char* toString(data_feature_utils::FeatureType v);
const char* toString(data_feature_utils::InternalNumType v);

int main() {
    std::cout << "Structure of array (SOA) numeric table example" << std::endl << std::endl;

    /*Example of using an SOA numeric table*/
    const size_t nObservations = 10;
    const size_t nFeatures = 4;
    int cDataSOA[nObservations] = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
    float fDataSOA[nObservations] = { 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4.0f };
    double dDataSOA[nObservations] = { 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8 };
    int iDataSOA[nObservations] = { -10, -20, -30, -40, -50, -60, -70, -80, -90, -100 };

    for (const auto& deviceSelector : getListOfDevices()) {
        const auto& nameDevice = deviceSelector.first;
        const auto& device = deviceSelector.second;
        sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        const size_t firstReadRow = 0;
        const size_t nRead = 3;
        size_t readFeatureIdx;

        /* Construct an SYCL SOA numeric table with nObservations rows and nFeatures columns */
        SyclSOANumericTablePtr dataTable = SyclSOANumericTable::create(nFeatures, nObservations);

        checkPtr(dataTable.get());
        dataTable->setArray<int>(cl::sycl::buffer<int>(cDataSOA, sycl::range<1>(nObservations)),
                                 0);
        dataTable->setArray<float>(
            sycl::buffer<float>(fDataSOA, sycl::range<1>(nObservations)),
            1);
        dataTable->setArray<double>(
            sycl::buffer<double>(dDataSOA, sycl::range<1>(nObservations)),
            2);
        dataTable->setArray<int>(cl::sycl::buffer<int>(iDataSOA, sycl::range<1>(nObservations)),
                                 3);

        /* Read a block of rows */
        BlockDescriptor<double> doubleBlock;
        dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, doubleBlock);
        printArray<double>(doubleBlock.getBlockPtr(),
                           nFeatures,
                           doubleBlock.getNumberOfRows(),
                           "Print SOA data structures as double:");
        dataTable->releaseBlockOfRows(doubleBlock);

        /* Read a feature (column) and write a new value into it */
        readFeatureIdx = 0;
        BlockDescriptor<int> intBlock;
        dataTable->getBlockOfColumnValues(readFeatureIdx,
                                          firstReadRow,
                                          nObservations,
                                          readOnly,
                                          intBlock);
        printArray<int>(intBlock.getBlockPtr(),
                        1,
                        intBlock.getNumberOfRows(),
                        "Print the first feature of SOA:");
        dataTable->releaseBlockOfColumnValues(intBlock);

        /* Get the dictionary and the number of features */
        NumericTableDictionaryPtr pDictionary = dataTable->getDictionarySharedPtr();
        std::cout << "Number of features in table: " << pDictionary->getNumberOfFeatures()
                  << std::endl;
        std::cout << std::endl;

        std::cout << "Default type in autogenerated dictionary:" << std::endl;
        for (size_t i = 0; i < nFeatures; i++) {
            data_feature_utils::FeatureType featureType = (*pDictionary)[i].featureType;
            std::cout << "Type of " << i << " feature: ";
            std::cout << toString(featureType) << std::endl;
        }
        std::cout << std::endl;

        /* Modify the dictionary information about data */
        NumericTableFeature& categoricalFeature = (*pDictionary)[0];
        categoricalFeature.featureType = data_feature_utils::DAAL_CATEGORICAL;

        std::cout << "Modified type in the dictionary:" << std::endl;
        for (size_t i = 0; i < nFeatures; i++) {
            data_feature_utils::FeatureType featureType = (*pDictionary)[i].featureType;
            std::cout << "Type of " << i << " feature: ";
            std::cout << toString(featureType) << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

const char* toString(data_feature_utils::FeatureType v) {
    switch (v) {
        case data_feature_utils::DAAL_CATEGORICAL: return "DAAL_CATEGORICAL";
        case data_feature_utils::DAAL_ORDINAL: return "DAAL_ORDINAL";
        case data_feature_utils::DAAL_CONTINUOUS: return "DAAL_CONTINUOUS";
        default: return "[Unknown FeatureType]";
    }
}
