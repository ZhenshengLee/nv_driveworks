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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief A condition variable that counts the waiting threads and only notifies when
 * necessary, preventing unneeded syscalls. Requires that all uses of a ConditionVariable use the same mutex.
 */
class ConditionVariable
{
public:
    /**
     * @brief Constructor for ConditionVariable. It initializes m_cond and m_waiting variables to NULL.
     * 
     * Access specifier: public
    */
    ConditionVariable()
        : m_cond(), m_waiting(0) {}

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock.
     * It will increase the waiting thread number before taking the lock and decrease it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
    */
    void wait(std::unique_lock<std::mutex>& lock)
    {
        m_waiting++;
        m_cond.wait(lock);
        m_waiting--;
    }

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock, only if the predicate is true.
     * It will increase the waiting thread number before taking the lock and decrease it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
     * @param[in] predicate The condition to check. If true, set thread to wait state.
    */
    template <class T>
    void wait(std::unique_lock<std::mutex>& lock, T predicate)
    {
        m_waiting++;
        m_cond.wait(lock, predicate);
        m_waiting--;
    }

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock and signal it after the specified timeout.
     * It will increase the waiting thread number before taking the lock and decrease it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
     * @param[in] timeout_time The timeout to elapse before releasing the lock.
     * 
     * @retval std::cv_status::timeout if the timeout specified has been reached.
     * @retval std::cv_status::no_timeout if the timeout specified is not reached yet.
    */
    template <class Clock, class Duration>
    std::cv_status wait_until(std::unique_lock<std::mutex>& lock,
                              const std::chrono::time_point<Clock, Duration>& timeout_time)
    {
        m_waiting++;
        std::cv_status res{m_cond.wait_until(lock, timeout_time)};
        m_waiting--;
        return res;
    }

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock and signal it after the specified timeout
     * and only if predicate is true. It will increase the waiting thread number before taking the lock and decrease 
     * it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
     * @param[in] timeout_time The timeout to elapse before releasing the lock.
     * @param[in] pred The condition to check. If true, set thread to wait state.
     *
     * @return the evaluation of the predicate (true or false)
    */
    template <class Clock, class Duration, class Pred>
    bool wait_until(std::unique_lock<std::mutex>& lock,
                    const std::chrono::time_point<Clock, Duration>& timeout_time,
                    Pred pred)
    {
        m_waiting++;
        bool res{m_cond.wait_until(lock, timeout_time, pred)};
        m_waiting--;
        return res;
    }

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock and signal it after the specified relative timeout.
     * It will increase the waiting thread number before taking the lock and decrease it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
     * @param[in] rel_time The relative timeout to elapse before releasing the lock.
     *
     * @retval std::cv_status::timeout if the timeout specified has been reached.
     * @retval std::cv_status::no_timeout if the timeout specified is not reached yet.
    */
    template <class Rep, class Period>
    std::cv_status wait_for(std::unique_lock<std::mutex>& lock,
                            const std::chrono::duration<Rep, Period>& rel_time)
    {
        m_waiting++;
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        std::cv_status res{m_cond.wait_for(lock, rel_time)};
        m_waiting--;
        return res;
    }

    /**
     * @brief Set thread to wait on m_cond variable with the specified lock and signal it after the specified
     * relative timeout and only if predicate is true. It will increase the waiting thread number before taking 
     * the lock and decrease it after the lock is released.
     *
     * Access specifier: public
     *
     * @param[in] lock The lock to wait on for the current thread.
     * @param[in] rel_time The relative timeout to elapse before releasing the lock.
     * @param[in] pred The condition to check. If true, set thread to wait state.
     *
     * @return the evaluation of the predicate (true or false)
    */
    template <class Rep, class Period, class Predicate>
    bool wait_for(std::unique_lock<std::mutex>& lock,
                  const std::chrono::duration<Rep, Period>& rel_time,
                  Predicate pred)
    {
        m_waiting++;
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        bool res{m_cond.wait_for(lock, rel_time, pred)};
        m_waiting--;
        return res;
    }

    /**
     * @brief Notify the threads waiting on m_cond and wake-up one of them. the waiting thread number must be more than 0.
     *
     * Access specifier: public
    */
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

    /**
     * @brief Notify the threads waiting on m_cond and wake-up all of them. the waiting thread number must be more than 0.
     *
     * Access specifier: public
    */
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

    /**
     * @brief Returns the native handle of condition variable m_cond.
     *
     * Access specifier: public
     *
     * @return The native handle of m_cond
    */
    std::condition_variable::native_handle_type native_handle()
    {
        return m_cond.native_handle();
    }

private:
    /// The condition variable used by the class object.
    std::condition_variable m_cond;
    /// Counts the number of threads waiting on m_cond.
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
