/* file: service_profiler.h */
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
//  Profiler for time measurement of kernels
//--
*/
#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <exception>
#include <unordered_map>
#include <thread>
#include "services/library_version_info.h"

#ifdef _WIN32
    #define PRETTY_FUNCTION __FUNCSIG__
#else
    #define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define __SERVICE_PROFILER_H__

#define DAAL_PROFILER_CONCAT2(x, y) x##y
#define DAAL_PROFILER_CONCAT(x, y)  DAAL_PROFILER_CONCAT2(x, y)
#define DAAL_PROFILER_UNIQUE_ID     __LINE__

#define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name, queue)
#define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// HEADER OUTPUT
#define DAAL_PROFILER_PRINT_HEADER()                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        std::cerr << "-----------------------------------------------------------------------------" << '\n'; \
        std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << '\n';                                  \
        if (daal::internal::is_service_debug_enabled())                                                       \
        {                                                                                                     \
            std::cerr << PRETTY_FUNCTION << '\n';                                                             \
        }                                                                                                     \
    } while (0)

// ARGS LOGGING
#define DAAL_PROFILER_LOG_ARGS(task_name, ...)                                  \
    do                                                                          \
    {                                                                           \
        DAAL_PROFILER_PRINT_HEADER();                                           \
        std::cerr << "Profiler task_name: " << #task_name << " Printed args: "; \
        daal::internal::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__);     \
        std::cerr << '\n';                                                      \
    } while (0)

#define DAAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                                                                          \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                                                               \
            }                                                                                                                                 \
            return daal::internal::profiler::start_task(#task_name);                                                                          \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_TASK(...)                                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_THREADING_TASK(task_name)                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            return daal::internal::profiler::start_threading_task(#task_name);                                                                \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_SERVICE_TASK(...)                                                                                                       \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_service_debug_enabled())                                                                                       \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

inline volatile int daal_verbose_val                 = -1;
inline static constexpr int PROFILER_MODE_OFF       = 0;
inline static constexpr int PROFILER_MODE_LOGGER    = 1;
inline static constexpr int PROFILER_MODE_TRACER    = 2;
inline static constexpr int PROFILER_MODE_ANALYZER  = 3;
inline static constexpr int PROFILER_MODE_ALL_TOOLS = 4;
inline static constexpr int PROFILER_MODE_DEBUG     = 5;

namespace daal
{
namespace internal
{

inline static void set_verbose_from_env()
{
    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = PROFILER_MODE_OFF;
    if (verbose_str)
    {
        char * endptr  = nullptr;
        errno          = 0;
        long val       = std::strtol(verbose_str, &endptr, 10);
        bool parsed_ok = (errno == 0 && endptr != verbose_str && *endptr == '\0');
        if (parsed_ok && val >= 0 && val <= 5) newval = static_cast<int>(val);
    }
    daal_verbose_val = newval;
}

inline static int daal_verbose_mode()
{
    if (daal_verbose_val == -1) set_verbose_from_env();
    return daal_verbose_val;
}

inline static std::string format_time_for_output(std::uint64_t time_ns)
{
    std::ostringstream out;
    double time = static_cast<double>(time_ns);
    if (time <= 0)
        out << "0.00s";
    else if (time > 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
    else if (time > 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
    else if (time > 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
    else
        out << static_cast<std::uint64_t>(time) << "ns";
    return out.str();
}

inline void profiler_log_named_args(const char * /*names*/) {}

template <typename T, typename... Rest>
inline void profiler_log_named_args(const char * names, const T & value, Rest &&... rest)
{
    const char * comma = strchr(names, ',');
    std::string name   = comma ? std::string(names, comma) : std::string(names);
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    std::cerr << name << ": " << value << "; ";
    if (comma) profiler_log_named_args(comma + 1, std::forward<Rest>(rest)...);
}

inline static bool is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_DEBUG;
    }();
    return service_debug_value;
}

inline static bool is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return logger_value;
}

inline static bool is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}

inline static bool is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS
               || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

inline static bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

inline void print_header()
{
    if (is_profiler_enabled())
    {
        daal::services::LibraryVersionInfo ver;
        std::cerr << "Major version:          " << ver.majorVersion << '\n';
        std::cerr << "Minor version:          " << ver.minorVersion << '\n';
        std::cerr << "Update version:         " << ver.updateVersion << '\n';
        std::cerr << "Product status:         " << ver.productStatus << '\n';
        std::cerr << "Build:                  " << ver.build << '\n';
        std::cerr << "Build revision:         " << ver.build_rev << '\n';
        std::cerr << "Name:                   " << ver.name << '\n';
        std::cerr << "Processor optimization: " << ver.processor << '\n';
        std::cerr << '\n';
    }
}

struct task_entry
{
    std::int64_t idx;
    std::string name;
    std::uint64_t duration;
    std::int64_t level;
    std::int64_t count;
    bool threading_task;
    /// Per-thread durations for threading tasks (populated during merge)
    std::vector<std::uint64_t> thread_durations;
};

struct task
{
    std::vector<task_entry> kernels;
};

class profiler_task
{
public:
    inline profiler_task(const char * task_name, int idx) : task_name_(task_name), idx_(idx) {}
    inline profiler_task(const char * task_name, int idx, bool thread) : task_name_(task_name), idx_(idx), is_thread_(thread) {}
    inline ~profiler_task();

    profiler_task(const profiler_task &)             = delete;
    profiler_task & operator=(const profiler_task &) = delete;

    profiler_task(profiler_task && other) noexcept : task_name_(other.task_name_), idx_(other.idx_), is_thread_(other.is_thread_)
    {
        other.task_name_ = nullptr;
        other.idx_       = -1;
        other.is_thread_ = false;
    }

    profiler_task & operator=(profiler_task && other) noexcept
    {
        if (this != &other)
        {
            task_name_ = other.task_name_;
            idx_       = other.idx_;
            is_thread_ = other.is_thread_;

            other.task_name_ = nullptr;
            other.idx_       = -1;
            other.is_thread_ = false;
        }
        return *this;
    }

private:
    const char * task_name_ = nullptr;
    int idx_                = -1;
    bool is_thread_         = false;
};

class profiler
{
public:
    inline profiler() { daal_verbose_mode(); }

    inline ~profiler()
    {
        if (is_analyzer_enabled())
        {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            try
            {
#endif
                merge_tasks();
                const auto & tasks_info  = get_instance()->get_task();
                std::uint64_t total_time = 0;
                std::cerr << "Algorithm tree analyzer" << '\n';

                for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
                {
                    const auto & entry = tasks_info.kernels[i];
                    if (entry.level == 0) total_time += entry.duration;
                }

                for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
                {
                    const auto & entry = tasks_info.kernels[i];
                    std::string prefix;
                    for (std::int64_t lvl = 0; lvl < entry.level; ++lvl) prefix += "|  ";
                    prefix += "|-- ";
                    std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << " " << std::fixed
                              << std::setprecision(2) << (total_time > 0 ? (double(entry.duration) / total_time) * 100 : 0.0) << "% " << entry.count
                              << " times in a " << (entry.threading_task ? "parallel" : "sequential") << " region";

                    // Per-thread stats for threading tasks
                    if (entry.threading_task && !entry.thread_durations.empty())
                    {
                        const auto & td = entry.thread_durations;
                        std::uint64_t tmin = *std::min_element(td.begin(), td.end());
                        std::uint64_t tmax = *std::max_element(td.begin(), td.end());
                        std::uint64_t tsum = 0;
                        std::uint64_t active_count = 0;
                        for (auto d : td)
                        {
                            tsum += d;
                            // A thread is "active" if it ran for at least 1% of the max
                            if (d > tmax / 100) active_count++;
                        }
                        double tavg = static_cast<double>(tsum) / td.size();
                        // Imbalance ratio: 1.0 = perfect balance, higher = worse
                        double imbalance = (tavg > 0) ? static_cast<double>(tmax) / tavg : 1.0;

                        std::cerr << '\n' << prefix << "  threads: " << td.size()
                                  << " | min: " << format_time_for_output(tmin)
                                  << " | max: " << format_time_for_output(tmax)
                                  << " | avg: " << format_time_for_output(static_cast<std::uint64_t>(tavg))
                                  << " | imbalance: " << std::fixed << std::setprecision(2) << imbalance << "x"
                                  << " | active: " << active_count << "/" << td.size();

                        if (imbalance > 2.0)
                        {
                            std::cerr << " WARNING: high thread imbalance!";
                        }
                        // Show idle threads (those with < 1% of max)
                        std::uint64_t idle_count = td.size() - active_count;
                        if (idle_count > 0)
                        {
                            std::cerr << " (" << idle_count << " idle threads)";
                        }
                    }
                    std::cerr << '\n';
                }
                std::cerr << "|--(end)" << '\n';
                std::cerr << "DAAL KERNEL_PROFILER: kernels total time " << format_time_for_output(total_time) << '\n';

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            }
            catch (std::exception & e)
            {
                std::cerr << e.what() << std::endl;
            }
#endif
        }
    }

    /// Starts a profiling task with the given task name and returns a profiler_task object
    ///
    /// @param[in] task_name The name of the task to be profiled
    ///
    /// @return A profiler_task object containing the task name and a unique task ID. Returns an invalid
    /// profiler_task (nullptr, -1) if task_name is nullptr
    ///
    /// @note Captures the start time, updates task information, increments the current nesting level and kernel count,
    /// and stores task details (ID, name, start time, level, active status) in the tasks_info.kernels vector.
    /// Invoked by the DAAL_PROFILER_TASK macro.
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        std::lock_guard<std::mutex> lock(global_mutex());
        auto ns_start                = get_time();
        auto & tasks_info            = get_instance()->get_task();
        auto & current_level_        = get_instance()->get_current_level();
        auto & current_kernel_count_ = get_instance()->get_kernel_count();
        std::int64_t tmp             = current_kernel_count_;
        tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, false });
        current_level_++;
        current_kernel_count_++;
        return profiler_task(task_name, tmp);
    }

    /// Starts a threading-specific profiling task with the given task name and returns a profiler_task object
    ///
    /// @param[in] task_name The name of the threading task to be profiled
    ///
    /// @return A profiler_task object containing the task name, a unique task ID, and a threading flag.
    /// Returns an invalid profiler_task (nullptr, -1) if task_name is nullptr
    ///
    /// @note Uses a mutex for thread safety, logs unique task names if logging is enabled, captures the start time,
    /// updates task info, and increments the kernel count. Stores task details in tasks_info.kernels, marking it
    /// as a threading task. Increments the nesting level so that inner threading tasks appear nested correctly
    /// in the analyzer tree. Invoked by the DAAL_PROFILER_THREADING_TASK macro.
    inline static profiler_task start_threading_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        std::lock_guard<std::mutex> lock(global_mutex());
        if (is_logger_enabled())
        {
            auto tid = std::this_thread::get_id();
            if (!is_service_debug_enabled())
            {
                static std::vector<std::string> unique_task_names;
                bool is_new_task = std::find(unique_task_names.begin(), unique_task_names.end(), task_name) == unique_task_names.end();
                if (is_new_task)
                {
                    unique_task_names.push_back(task_name);
                    std::cerr << "-----------------------------------------------------------------------------" << '\n';
                    std::cerr << "THREADING Profiler task started on the main rank: " << task_name << '\n';
                }
            }
            else
            {
                std::cerr << "-----------------------------------------------------------------------------" << '\n';
                std::cerr << "THREADING Profiler task started " << task_name << " [thread " << tid << "]" << '\n';
            }
        }
        auto ns_start                = get_time();
        auto & tasks_info            = get_instance()->get_task();
        auto & current_level_        = get_instance()->get_current_level();
        auto & current_kernel_count_ = get_instance()->get_kernel_count();
        std::int64_t tmp             = current_kernel_count_;
        // Threading tasks all appear at the same level (they're siblings inside the parallel region).
        // Do NOT increment current_level_ here — that's what caused nesting corruption when
        // multiple threads pushed entries concurrently.
        tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, true, {} });
        current_kernel_count_++;
        return profiler_task(task_name, tmp, true);
    }

    /// Terminates a profiling task and records its duration
    ///
    /// @param[in] task_name The name of the task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note If task_name is nullptr, the function returns immediately. Captures the end time,
    /// calculates the task duration, updates the task entry, and decrements the current nesting level.
    /// Logs the task name and duration if tracing is enabled. Uses a mutex for thread safety.
    /// Invoked by macros such as DAAL_PROFILER_TASK.
    inline static void end_task(const char * task_name, int idx_)
    {
        if (!task_name) return;
        const std::uint64_t ns_end = get_time();
        auto & tasks_info          = get_instance()->get_task();

        std::lock_guard<std::mutex> lock(global_mutex());
        auto & entry          = tasks_info.kernels[idx_];
        auto duration         = ns_end - entry.duration;
        entry.duration        = duration;
        auto & current_level_ = get_instance()->get_current_level();
        current_level_--;
        if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
    }

    /// Terminates a threading-specific profiling task and records its duration
    ///
    /// @param[in] task_name The name of the threading task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note If task_name is nullptr or idx_ is invalid, the function returns immediately.
    /// Captures the end time, calculates the task duration, updates the task entry, and decrements
    /// the nesting level. Logs the task name and duration if tracing is enabled.
    /// Uses a mutex for thread safety. Invoked by the DAAL_PROFILER_THREADING_TASK macro.
    inline static void end_threading_task(const char * task_name, int idx_)
    {
        if (!task_name) return;

        std::lock_guard<std::mutex> lock(global_mutex());
        const std::uint64_t ns_end = get_time();
        auto & tasks_info          = get_instance()->get_task();

        if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

        auto & entry   = tasks_info.kernels[idx_];
        auto duration  = ns_end - entry.duration;
        entry.duration = duration;
        // Do NOT decrement current_level_ — threading tasks don't own a level slot.
        // They all sit at the same level as siblings under the parent sequential task.

        if (is_tracer_enabled())
        {
            std::cerr << "THREADING " << task_name << " [thread " << std::this_thread::get_id() << "] finished: "
                      << format_time_for_output(duration) << '\n';
        }
    }

    inline static std::uint64_t get_time()
    {
        auto now = std::chrono::steady_clock::now();
        auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        return static_cast<std::uint64_t>(ns);
    }

    inline static profiler * get_instance()
    {
        static profiler instance;
        return &instance;
    }

    /// Merges tasks at the same nesting level with identical names to improve profiling clarity
    ///
    /// @note Combines tasks with the same name and level in the tasks_info.kernels vector to simplify
    /// profiling output. For non-threading tasks, durations are summed; for threading tasks, the maximum
    /// duration is taken. Updates task counts and removes redundant entries. Skips merging if service
    /// debug mode is enabled to preserve detailed task information.
    inline void merge_tasks()
    {
        if (is_service_debug_enabled())
        {
            return;
        }
        auto & tasks_info = get_instance()->get_task();
        auto & kernels    = tasks_info.kernels;
        size_t i          = 0;
        while (i < kernels.size())
        {
            size_t start      = i;
            int current_level = kernels[i].level;
            size_t end        = start;
            while (end < kernels.size() && kernels[end].level == current_level) ++end;
            for (size_t j = start; j < end; ++j)
            {
                for (size_t k = j + 1; k < end; ++k)
                {
                    if (kernels[j].name == kernels[k].name)
                    {
                        if (kernels[j].threading_task)
                        {
                            // First merge: populate thread_durations from the first entry
                            if (kernels[j].thread_durations.empty())
                            {
                                kernels[j].thread_durations.push_back(kernels[j].duration);
                            }
                            // Collect this thread's duration
                            kernels[j].thread_durations.push_back(kernels[k].duration);
                            // Wallclock = max across threads (parallel region ends when slowest finishes)
                            kernels[j].duration = std::max(kernels[j].duration, kernels[k].duration);
                        }
                        else
                        {
                            kernels[j].duration += kernels[k].duration;
                        }
                        kernels.erase(kernels.begin() + k);
                        --k;
                        --end;
                        kernels[j].count++;
                    }
                }
            }
            i = end;
        }
    }

    inline task & get_task()
    {
        return task_;
    }
    inline std::int64_t & get_current_level()
    {
        return current_level_;
    }
    inline std::int64_t & get_kernel_count()
    {
        return kernel_count_;
    }

private:
    std::int64_t current_level_ = 0;
    std::int64_t kernel_count_  = 0;
    task task_;
    static std::mutex & global_mutex()
    {
        static std::mutex m;
        return m;
    }
};

inline profiler_task::~profiler_task()
{
    if (task_name_)
    {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        try
        {
#endif
            if (is_thread_)
                profiler::end_threading_task(task_name_, idx_);
            else
                profiler::end_task(task_name_, idx_);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        }
        catch (std::exception & e)
        {
            std::cerr << e.what() << std::endl;
        }
#endif
    }
}

/// Memory tracker for profiling allocation/deallocation patterns.
/// Controlled by ONEDAL_MEMTRACK environment variable:
///   0 (default): disabled
///   1: log allocations/deallocations to stderr
///   2: log + summary on exit (peak, current, count)
class memory_tracker
{
public:
    inline static memory_tracker * get_instance()
    {
        static memory_tracker instance;
        return &instance;
    }

    inline static bool is_enabled()
    {
        static const int level = [] {
            const char * val = std::getenv("ONEDAL_MEMTRACK");
            if (!val) return 0;
            char * endptr = nullptr;
            long v        = std::strtol(val, &endptr, 10);
            if (endptr == val || *endptr != '\0' || v < 0 || v > 2) return 0;
            return static_cast<int>(v);
        }();
        return level > 0;
    }

    inline static int tracking_level()
    {
        static const int level = [] {
            const char * val = std::getenv("ONEDAL_MEMTRACK");
            if (!val) return 0;
            char * endptr = nullptr;
            long v        = std::strtol(val, &endptr, 10);
            if (endptr == val || *endptr != '\0' || v < 0 || v > 2) return 0;
            return static_cast<int>(v);
        }();
        return level;
    }

    inline void on_alloc(const char * source, std::size_t bytes, const void * ptr)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        alloc_count_++;
        current_bytes_ += bytes;
        if (current_bytes_ > peak_bytes_) peak_bytes_ = current_bytes_;
        total_allocated_ += bytes;
        if (tracking_level() >= 1)
        {
            std::cerr << "[MEMTRACK] alloc " << format_bytes(bytes) << " via " << source << " -> " << ptr
                      << " (current: " << format_bytes(current_bytes_) << ", peak: " << format_bytes(peak_bytes_) << ")" << '\n';
        }
    }

    inline void on_free(const char * source, const void * ptr, std::size_t bytes)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        free_count_++;
        if (bytes <= current_bytes_)
            current_bytes_ -= bytes;
        else
            current_bytes_ = 0;
        if (tracking_level() >= 1)
        {
            std::cerr << "[MEMTRACK] free " << format_bytes(bytes) << " via " << source << " ptr=" << ptr
                      << " (current: " << format_bytes(current_bytes_) << ")" << '\n';
        }
    }

    inline ~memory_tracker()
    {
        if (tracking_level() >= 2 && (alloc_count_ > 0 || free_count_ > 0))
        {
            std::cerr << "\n[MEMTRACK] === Memory tracking summary ===" << '\n';
            std::cerr << "[MEMTRACK] Total allocations:   " << alloc_count_ << '\n';
            std::cerr << "[MEMTRACK] Total deallocations: " << free_count_ << '\n';
            std::cerr << "[MEMTRACK] Total allocated:     " << format_bytes(total_allocated_) << '\n';
            std::cerr << "[MEMTRACK] Peak usage:          " << format_bytes(peak_bytes_) << '\n';
            std::cerr << "[MEMTRACK] Current (at exit):   " << format_bytes(current_bytes_) << '\n';
            if (current_bytes_ > 0)
            {
                std::cerr << "[MEMTRACK] WARNING: " << format_bytes(current_bytes_) << " not freed at exit" << '\n';
            }
        }
    }

private:
    inline memory_tracker() = default;

    inline static std::string format_bytes(std::size_t bytes)
    {
        std::ostringstream out;
        double b = static_cast<double>(bytes);
        if (b >= 1073741824.0)
            out << std::fixed << std::setprecision(2) << b / 1073741824.0 << " GB";
        else if (b >= 1048576.0)
            out << std::fixed << std::setprecision(2) << b / 1048576.0 << " MB";
        else if (b >= 1024.0)
            out << std::fixed << std::setprecision(2) << b / 1024.0 << " KB";
        else
            out << bytes << " B";
        return out.str();
    }

    std::mutex mtx_;
    std::size_t alloc_count_     = 0;
    std::size_t free_count_      = 0;
    std::size_t current_bytes_   = 0;
    std::size_t peak_bytes_      = 0;
    std::size_t total_allocated_ = 0;
};

#define DAAL_MEMTRACK_ALLOC(source, bytes, ptr)                                    \
    do                                                                             \
    {                                                                              \
        if (daal::internal::memory_tracker::is_enabled())                          \
            daal::internal::memory_tracker::get_instance()->on_alloc(source, bytes, ptr); \
    } while (0)

#define DAAL_MEMTRACK_FREE(source, ptr, bytes)                                     \
    do                                                                             \
    {                                                                              \
        if (daal::internal::memory_tracker::is_enabled())                          \
            daal::internal::memory_tracker::get_instance()->on_free(source, ptr, bytes); \
    } while (0)

} // namespace internal
} // namespace daal
