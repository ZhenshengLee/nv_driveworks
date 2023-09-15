/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DWSHARED_CORE_CONDITIONVARIABLE_HPP_
#define DWSHARED_CORE_CONDITIONVARIABLE_HPP_

#include <chrono>
#include <condition_variable>
#include <atomic>

namespace dw
{
namespace core
{

#ifdef __QNX__
/**
 * A condition variable that counts the waiting threads and only notifies when
 * necessary, preventing unneeded syscalls.
 * Requires that all uses of a ConditionVariable use the same mutex.
 */
class ConditionVariable
{
public:
    ConditionVariable()
        : m_cond(), m_waiting(0) {}

    void wait(std::unique_lock<std::mutex>& lock)
    {
        m_waiting++;
        m_cond.wait(lock);
        m_waiting--;
    }

    template <class T>
    void wait(std::unique_lock<std::mutex>& lock, T predicate)
    {
        m_waiting++;
        m_cond.wait(lock, predicate);
        m_waiting--;
    }

    template <class Clock, class Duration>
    std::cv_status wait_until(std::unique_lock<std::mutex>& lock,
                              const std::chrono::time_point<Clock, Duration>& timeout_time)
    {
        m_waiting++;
        auto res = m_cond.wait_until(lock, timeout_time);
        m_waiting--;
        return res;
    }

    template <class Clock, class Duration, class Pred>
    bool wait_until(std::unique_lock<std::mutex>& lock,
                    const std::chrono::time_point<Clock, Duration>& timeout_time,
                    Pred pred)
    {
        m_waiting++;
        auto res = m_cond.wait_until(lock, timeout_time, pred);
        m_waiting--;
        return res;
    }

    template <class Rep, class Period>
    std::cv_status wait_for(std::unique_lock<std::mutex>& lock,
                            const std::chrono::duration<Rep, Period>& rel_time)
    {
        m_waiting++;
        auto res = m_cond.wait_for(lock, rel_time);
        m_waiting--;
        return res;
    }

    template <class Rep, class Period, class Predicate>
    bool wait_for(std::unique_lock<std::mutex>& lock,
                  const std::chrono::duration<Rep, Period>& rel_time,
                  Predicate pred)
    {
        m_waiting++;
        auto res = m_cond.wait_for(lock, rel_time, pred);
        m_waiting--;
        return res;
    }

    void notify_one() noexcept
    {
        if (m_waiting > 0)
        {
            /*
             * WAR: QNX's libc impl of notify_*() is bad, it unnecessarily does
             * a full system call even if we don't have any waiters which is
             * almost always the case for us. We reached out to QNX they
             * have confirmed our fixing was right but hesitant to modify
             * QNX libc.
             * System calls on QNX are expensive, so doing the optimization in
             * notify function that removing unnecessary system calls.
            */
            m_cond.notify_one();
        }
    }

    void notify_all() noexcept
    {
        if (m_waiting > 0)
        {
            /*
             * WAR: QNX's libc impl of notify_*() is bad, it unnecessarily does
             * a full system call even if we don't have any waiters which is
             * almost always the case for us. We reached out to QNX they
             * have confirmed our fixing was right but hesitant to modify
             * QNX libc.
             * System calls on QNX are expensive, so doing the optimization in
             * notify function that removing unnecessary system calls.
            */
            m_cond.notify_all();
        }
    }

    std::condition_variable::native_handle_type native_handle()
    {
        return m_cond.native_handle();
    }

private:
    std::condition_variable m_cond;
    std::atomic<size_t> m_waiting;
};

#else
// The linux implementations of libc already appear to count waiting threads
// and avoid extra syscalls, so we can fallback to std::condition_variable.
using ConditionVariable = std::condition_variable;
#endif

} // namespace core
} // namespace dw

#endif // DW_CORE_CONDITIONVARIABLE_HPP_
