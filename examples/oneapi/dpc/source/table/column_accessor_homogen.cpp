/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <vector>
#include <iostream>
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif



void print_device_info(sycl::device dev) {
    std::cout << "Platfrom: " << dev.get_platform().get_info<sycl::info::platform::name>()
              << std::endl;

    std::cout << "\tName:\t" << dev.get_info<sycl::info::device::name>() << std::endl;

    std::string device = "UNKNOWN";
    switch (dev.get_info<sycl::info::device::device_type>()) {
        case sycl::info::device_type::gpu: device = "GPU"; break;
        case sycl::info::device_type::cpu: device = "CPU"; break;
        case sycl::info::device_type::accelerator: device = "ACCELERATOR"; break;
        case sycl::info::device_type::host: device = "HOST"; break;
        default: break;
    }
    std::cout << "\tType:\t" << device << std::endl;
    std::cout << "\tVendor:\t" << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "\tVersion:\t" << dev.get_info<sycl::info::device::version>() << std::endl;
    std::cout << "\tDriver:\t" << dev.get_info<sycl::info::device::driver_version>() << std::endl;



    std::cout << "\tMax comp units:\t" << dev.get_info<sycl::info::device::max_compute_units>()
              << std::endl;
    std::cout << "\tMax work item dims:\t"
              << dev.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
    std::cout << "\tMax work group size:\t"
              << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "\tGlobal mem size:\t" << dev.get_info<sycl::info::device::global_mem_size>()
              << std::endl;
    std::cout << "\tGlobal mem cache size:\t"
              << dev.get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
    std::cout << std::endl;
}

int main() {
    if (std::getenv("SYCL_DEVICE_FILTER")) {
        std::string env = std::getenv("SYCL_DEVICE_FILTER");

        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;

        while ((pos = env.find(delimiter)) != std::string::npos) {
            token = env.substr(0, pos);
            auto d = sycl::ext::oneapi::filter_selector(token).select_device();
            print_device_info(d);
            env.erase(0, pos + delimiter.length());
        }
        auto d = sycl::ext::oneapi::filter_selector(env).select_device();
        print_device_info(d);
    }
    else {
        for (auto &platform : sycl::platform::get_platforms()) {
            for (auto &device : platform.get_devices()) {
                print_device_info(device);
            }
        }
    }
}
