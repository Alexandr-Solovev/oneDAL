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

#pragma once

#include <cstring>
#include <string>

#include "oneapi/dal/detail/profiler.hpp"
#include "src/services/service_defines.h"

namespace oneapi::dal::backend {

/// Shortens a qualified type name to a profiler task name: the outer type keeps its first and
/// last scopes that are not library namespaces, a template argument keeps only its last one.
///
/// @param[in] type    The type name, as the compiler spells it
/// @param[in] length  Length of `type`
/// @param[in] suffix  Appended to the result
///
/// @return The task name. It is never freed, as the profiler reads it at exit.
inline const char* make_kernel_task_name(const char* type, std::size_t length, const char* suffix) {
    std::string text(type, length);
    for (const char* anonymous :
         { "(anonymous namespace)::", "{anonymous}::", "`anonymous namespace'::" }) {
        for (auto at = text.find(anonymous); at != std::string::npos; at = text.find(anonymous)) {
            text.erase(at, std::strlen(anonymous));
        }
    }
    const auto is_word = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_';
    };
    const auto is_noise = [](const std::string& scope) {
        static const char* const noise[] = { "oneapi",   "dal",        "daal",   "backend",
                                             "internal", "detail",     "v1",     "v2",
                                             "interop",  "algorithms", "method", "task",
                                             "struct",   "class",      "enum",   "parameters" };
        for (const char* n : noise) {
            if (scope == n) {
                return true;
            }
        }
        return false;
    };

    std::string out;
    int depth = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        if (!is_word(text[i])) {
            depth += (text[i] == '<') - (text[i] == '>');
            if (text[i] != ' ') {
                out += text[i];
            }
            ++i;
            continue;
        }
        std::string scopes[2];
        for (;;) {
            std::size_t end = i;
            while (end < text.size() && is_word(text[end])) {
                ++end;
            }
            std::string scope = text.substr(i, end - i);
            i = end;
            while (i < text.size() && text[i] == ' ') {
                ++i;
            }
            // MSVC spells `struct x` and `class x`.
            const bool keyword = scope == "struct" || scope == "class" || scope == "enum";
            if (!is_noise(scope)) {
                if (scopes[0].empty()) {
                    scopes[0] = scopes[1];
                }
                scopes[1] = std::move(scope);
            }
            if (text.compare(i, 2, "::") == 0) {
                i += 2;
                continue;
            }
            if (keyword && i < text.size() && is_word(text[i])) {
                continue;
            }
            break;
        }
        if (depth == 0 && !scopes[0].empty()) {
            out += scopes[0];
            out += '.';
        }
        out += scopes[1];
    }
    out += suffix;

    char* name = new char[out.size() + 1];
    std::memcpy(name, out.c_str(), out.size() + 1);
    return name;
}

/// The default step of a kernel, which adds nothing to its task name.
struct kernel_step_default {
    static constexpr const char* suffix = "";
};

/// The profiler task name of the kernel type `Kernel`, built once per type.
/// The signature spells the type up to a top-level `,`, `;` or `]`, or the closing `>` on MSVC.
template <typename Kernel, typename Step = kernel_step_default>
inline const char* kernel_task_name() {
    static const char* const name = [](const char* signature) {
#ifdef _MSC_VER
        const char* const marker = "kernel_task_name<";
#else
        const char* const marker = "Kernel = ";
#endif
        const char* const found = std::strstr(signature, marker);
        const char* const begin = found ? found + std::strlen(marker) : signature;
        const char* end = begin;
        for (int depth = 0; *end; ++end) {
            if (depth == 0 && (*end == ',' || *end == ';' || *end == ']' || *end == '>')) {
                break;
            }
            depth += (*end == '<') - (*end == '>');
        }
        return make_kernel_task_name(begin, std::size_t(end - begin), Step::suffix);
    }(PRETTY_FUNCTION);
    return name;
}

using kernel_task_name_t = const char* (*)();

/// Starts the task named by `name`; out of line, so a disabled profiler leaves only a load
/// and a branch at the call site.
DAAL_FORCENOINLINE inline daal::internal::profiler_task start_named_kernel_task(
    kernel_task_name_t name) {
    return daal::internal::profiler::start_task(name());
}

/// Starts a profiler task named after the kernel type `Kernel` and its step `Step`.
template <typename Kernel, typename Step = kernel_step_default>
inline daal::internal::profiler_task start_kernel_task() {
    if (DAAL_PROFILER_UNLIKELY(daal::internal::is_profiler_enabled())) {
        return start_named_kernel_task(&kernel_task_name<Kernel, Step>);
    }
    return daal::internal::profiler_task();
}

#ifdef ONEDAL_DATA_PARALLEL
/// Starts the task named by `name` on `queue`, out of line for the same reason.
DAAL_FORCENOINLINE inline detail::queue_sync_guard start_named_kernel_task(
    sycl::queue& queue,
    kernel_task_name_t name) {
    return detail::queue_sync_guard::make(queue, name(), true);
}

/// Starts a profiler task named after the kernel type `Kernel` that covers the device work
/// it submits to `queue`.
template <typename Kernel>
inline detail::queue_sync_guard start_kernel_task(sycl::queue& queue) {
    if (DAAL_PROFILER_UNLIKELY(daal::internal::is_profiler_enabled())) {
        return start_named_kernel_task(queue, &kernel_task_name<Kernel>);
    }
    return detail::queue_sync_guard::make(queue, nullptr, false);
}
#endif

} // namespace oneapi::dal::backend
