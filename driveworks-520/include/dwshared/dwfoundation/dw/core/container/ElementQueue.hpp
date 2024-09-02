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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_ELEMENTQUEUE_HPP_
#define DWSHARED_CORE_ELEMENTQUEUE_HPP_

#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>

#include <functional>
#include <thread>
#include <cstring>
#include <mutex>
#include <dwshared/dwfoundation/dw/core/language/ConditionVariable.hpp>
#include <dwshared/dwfoundation/dw/core/safety/Safety.hpp>

namespace dw
{
namespace core
{

/// ElementQueue functions return status
enum class EElementQueueStatus : uint32_t
{
    Ok,
    Timeout,
    BufferEmpty,
    BufferFull,
    InvalidArgument
};

/**
 * @brief Queue container that allows adding new elements at the back and
 *        getting old elements at the front.
 *
 * @tparam Element_t Type of stored elements
 * @tparam FifoSize  Maximum number of elements the queue can hold
 */
template <class Element_t, size_t FifoSize = 0>
class ElementQueue final // make sure we are not using virtual
{
public:
    using EStatus = EElementQueueStatus; ///< Type of status enum

    /// Construction from queue capacity
    explicit ElementQueue(size_t const fifoSize = FifoSize)
        : m_producerHead(0U)
        , m_consumerHead(0U)
        , m_bufferCapacity(fifoSize)
        , m_mutex()
        , m_sizeChanged()
        , m_dequeuedData(nullptr)
        , m_dequeuedSize(0U)
    {
        // coverity[autosar_cpp14_m0_1_3_violation] RFD Pending: TID-1995
        std::lock_guard<std::mutex> const lock{m_mutex};

        m_buffer = std::make_unique<Element_t[]>(m_bufferCapacity);
    }

    /// Destructor
    ~ElementQueue() = default;

    /// Move constructor
    ElementQueue(ElementQueue&& other)
        : m_producerHead(std::move(other.m_producerHead))
        , m_consumerHead(std::move(other.m_consumerHead))
        , m_bufferCapacity(std::move(other.m_bufferCapacity))
        // TODO(dwplc): FP - no move semantics for mutex
        // coverity[autosar_cpp14_a12_8_4_violation]
        , m_mutex()
        // TODO(dwplc): FP - no move semantics for conditional variable
        // coverity[autosar_cpp14_a12_8_4_violation]
        , m_sizeChanged()
        , m_dequeuedData(std::move(other.m_dequeuedData))
        , m_dequeuedSize(std::move(other.m_dequeuedSize))
        , m_buffer(std::move(other.m_buffer))
    {
        other.clear();
    }

    /// Move operator
    auto operator=(ElementQueue&& other) -> ElementQueue&
    {
        m_producerHead   = std::move(other.m_producerHead);
        m_consumerHead   = std::move(other.m_consumerHead);
        m_bufferCapacity = std::move(other.m_bufferCapacity);
        m_dequeuedData   = std::move(other.m_dequeuedData);
        m_dequeuedSize   = std::move(other.m_dequeuedSize);
        m_buffer         = std::move(other.m_buffer);
        other.clear();

        return *this;
    }

    /// Copy constructor
    ElementQueue(ElementQueue const& other)
        : m_producerHead(other.m_producerHead)
        , m_consumerHead(other.m_consumerHead)
        , m_bufferCapacity(other.m_bufferCapacity)
        , m_mutex()
        , m_sizeChanged()
        , m_dequeuedData(other.m_dequeuedData)
        , m_dequeuedSize(other.m_dequeuedSize)
    {
        m_buffer = std::make_unique<Element_t[]>(other.m_bufferCapacity);
        for (size_t i{0U}; i < other.m_bufferCapacity; i++)
        {
            m_buffer[i] = other.m_buffer[i];
        }
    }

    /// Copy operator
    auto operator=(ElementQueue const& other) -> ElementQueue&
    {
        m_producerHead   = other.m_producerHead;
        m_consumerHead   = other.m_consumerHead;
        m_bufferCapacity = other.m_bufferCapacity;
        m_dequeuedData   = other.m_dequeuedData;
        m_dequeuedSize   = other.m_dequeuedSize;

        m_buffer = std::make_unique<Element_t[]>(other.m_bufferCapacity);
        for (size_t i{0U}; i < other.m_bufferCapacity; i++)
        {
            m_buffer[i] = other.m_buffer[i];
        }

        return *this;
    }

    /// Set size and indices to 0
    void clear()
    {
        // coverity[autosar_cpp14_m0_1_3_violation] RFD Pending: TID-1995
        std::lock_guard<std::mutex> const lock{m_mutex};

        m_dequeuedData = nullptr;
        m_dequeuedSize = 0U;
        m_producerHead = 0U;
        m_consumerHead = 0U;
    }

    /**
     * Get the pointer of first element in the queue.
     * Return null pointer if the queue is empty.
     *
     * @return pointer to the first element, nullptr if queue is empty
     *
     * @note Only single consumer is supported, meaning, only one thread should call this function.
     **/
    auto front(int64_t const timeoutUs = 0) -> Element_t*
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if ((timeoutUs == 0) && emptyUnsafe())
        {
            return nullptr;
        }

        bool const containsElementsAfterWait{waitForSizeChange(
            lock, timeoutUs, [this]() -> bool { return !emptyUnsafe(); })};
        if (!containsElementsAfterWait)
        {
            return nullptr;
        }

        size_t const bufferOffset{m_consumerHead % m_bufferCapacity};
        return &m_buffer[bufferOffset];
    }

    /**
     * Pops first element from the queue.
     * The method will block for given timeout if queue is empty.
     *
     * @param[out] data Pointer to a data element, supporting operator=() where the element will be assigned.
     * @param[in] timeoutUs Timeout in usec to wait if queue is empty
     *
     * @return EStatus::Timeout - if timeout has been reached and queue is still empty <br>
     *         EStatus::BufferEmpty - if timeout was 0 and buffer is empty <br>
     *         EStatus::Ok - operation succeeded <br>
     *
     * @note Only single consumer is supported, meaning, only one thread should call this function.
     **/
    auto pop_front(Element_t* const data, int64_t const timeoutUs = 0) -> EStatus
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if ((timeoutUs == 0) && emptyUnsafe())
        {
            return EStatus::BufferEmpty;
        }

        bool const containsElementsAfterWait{waitForSizeChange(
            lock, timeoutUs, [this]() -> bool { return !emptyUnsafe(); })};
        if (!containsElementsAfterWait)
        {
            return timeoutUs > 0 ? EStatus::Timeout : EStatus::BufferEmpty;
        }

        size_t const bufferOffset{m_consumerHead % m_bufferCapacity};
        if (nullptr != data)
        {
            *data = m_buffer[bufferOffset]; // do not write out, if user requests just an empty pop
        }
        // TODO(dwplc): FP - INT30-C-EX1, m_consumerHead is supporting modulo behavior
        // coverity[cert_int30_c_violation]
        m_consumerHead++;

        m_sizeChanged.notify_all();

        return EStatus::Ok;
    }

    /**
     * Push one element to the back of the queue.
     * The method will block for given timeout if queue is full.
     *
     * @param[in] data Element to be put (copied) into the queue.
     * @param[in] timeoutUs Timeout in usec to wait if queue is full
     *
     * @return EStatus::Timeout - if timeout has been reached and queue is still full <br>
     *         EStatus::BufferFull - if timeout was 0 and buffer is empty <br>
     *         EStatus::Ok - operation succeeded <br>
     *
     * @note Only single producer is supported, meaning, only one thread should call this function.
     **/
    auto push_back(const Element_t& data, int64_t const timeoutUs = 0) -> EStatus
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if ((timeoutUs == 0) && fullUnsafe())
        {
            return EStatus::BufferFull;
        }

        bool const hasSpaceAfterWait{waitForSizeChange(
            lock, timeoutUs, [this]() -> bool { return !fullUnsafe(); })};
        if (!hasSpaceAfterWait)
        {
            return timeoutUs > 0 ? EStatus::Timeout : EStatus::BufferFull;
        }

        size_t const bufferOffset{m_producerHead % m_bufferCapacity};
        m_buffer[bufferOffset] = data;

        // TODO(dwplc): FP - INT30-C-EX1, m_producerHead is supporting modulo behavior
        // coverity[cert_int30_c_violation]
        m_producerHead++;

        m_sizeChanged.notify_all();

        return EStatus::Ok;
    }

    //! Same as push_back(), but copies a set of elements into the queue.
    //! @note Only single producer is supported, meaning, only one thread should call this function.
    auto push_back(const Element_t* const data, size_t const numElements, int64_t const timeoutUs = 0) -> EStatus
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if (numElements == 0U)
        {
            return EStatus::Ok;
        }

        if ((timeoutUs == 0) && !hasSpaceForUnsafe(numElements))
        {
            return EStatus::BufferFull;
        }

        bool const hasSpaceAfterWait{waitForSizeChange(
            lock, timeoutUs, [this, numElements]() -> bool { return hasSpaceForUnsafe(numElements); })};
        if (!hasSpaceAfterWait)
        {
            return timeoutUs > 0 ? EStatus::Timeout : EStatus::BufferFull;
        }

        size_t const bufferOffset{m_producerHead % m_bufferCapacity};
        pushBackInternal(data, numElements, bufferOffset);

        // TODO(dwplc): FP - INT30-C-EX1, m_producerHead is supporting modulo behavior
        // coverity[cert_int30_c_violation]
        m_producerHead += numElements;

        m_sizeChanged.notify_all();

        return EStatus::Ok;
    }

    /**
     * Dequeues as many as possible elements from the front of the queue.
     *
     * @note The difference to pop_front() is that only a pointer to a certain number of elements
     *       is returned. The pointer need to be returned back in dequeueReturn() after consumption.
     * @note Only single consumer is supported, meaning, only one thread should call this function.
     **/
    auto dequeue(const Element_t** const data, size_t* const numElements, int64_t const timeoutUs = 0) -> EStatus
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if ((data == nullptr) || (numElements == nullptr))
        {
            return EStatus::InvalidArgument;
        }

        if (m_dequeuedData != nullptr)
        {
            return EStatus::BufferEmpty;
        }

        if ((timeoutUs == 0) && emptyUnsafe())
        {
            return EStatus::BufferEmpty;
        }

        bool const containsElementsAfterWait{waitForSizeChange(
            lock, timeoutUs, [this]() -> bool { return !emptyUnsafe(); })};
        if (!containsElementsAfterWait)
        {
            return timeoutUs > 0 ? EStatus::Timeout : EStatus::BufferEmpty;
        }

        // Figure out available data and size
        size_t const bufferOffset{m_consumerHead % m_bufferCapacity};
        // TODO(dwplc): FP - bufferOffset is mod result which must less than m_bufferCapacity
        // coverity[cert_int30_c_violation]
        size_t const bufferRemainder{m_bufferCapacity - bufferOffset};

        *data        = &(m_buffer[bufferOffset]);
        *numElements = std::min(bufferRemainder, sizeUnsafe());

        m_dequeuedData = *data;
        m_dequeuedSize = *numElements;

        return EStatus::Ok;
    }

    /**
     * Dequeues required numbers of elements from the front of the queue.
     *
     * @note The difference to pop_front() is that only a pointer to a certain number of elements
     *       is returned. The pointer need to be returned back in dequeueReturn() after consumption.
     * @note Only single consumer is supported, meaning, only one thread should call this function.
     **/
    auto dequeueWithSize(const Element_t** const data, size_t* const numElements, const size_t numElementsRequired, int64_t const timeoutUs = 0) -> EStatus
    {
        std::unique_lock<std::mutex> lock{m_mutex};

        if (numElementsRequired == 0)
        {
            // invalid input numElementsRequired usage
            return EStatus::BufferEmpty;
        }

        if ((data == nullptr) || (numElements == nullptr))
        {
            return EStatus::InvalidArgument;
        }

        if (m_dequeuedData != nullptr)
        {
            return EStatus::BufferEmpty;
        }

        if ((timeoutUs == 0) && emptyUnsafe())
        {
            return EStatus::BufferEmpty;
        }

        bool const containsElementsAfterWait{waitForSizeChange(
            lock, timeoutUs, [this]() -> bool { return !emptyUnsafe(); })};
        if (!containsElementsAfterWait)
        {
            return timeoutUs > 0 ? EStatus::Timeout : EStatus::BufferEmpty;
        }

        // Figure out available data and size
        size_t const bufferOffset{m_consumerHead % m_bufferCapacity};
        // TODO(dwplc): FP - bufferOffset is mod result which must less than m_bufferCapacity
        // coverity[cert_int30_c_violation]
        size_t const bufferRemainder{m_bufferCapacity - bufferOffset};

        if ((sizeUnsafe() < numElementsRequired) || (bufferRemainder < numElementsRequired))
        {
            // no valid data in queue to drain
            return EStatus::BufferEmpty;
        }
        *data        = &(m_buffer[bufferOffset]);
        *numElements = numElementsRequired;

        m_dequeuedData = *data;
        m_dequeuedSize = numElementsRequired;

        return EStatus::Ok;
    }

    //! All dequeued elements must be returned here
    //! @note Only single consumer is supported, meaning, only one thread should call this function.
    bool dequeueReturn(const Element_t* const data)
    {
        // coverity[autosar_cpp14_m0_1_3_violation] RFD Pending: TID-1995
        std::lock_guard<std::mutex> const lock{m_mutex};

        if (data != m_dequeuedData)
        {
            return false; // maybe this should throw
        }

        m_consumerHead += m_dequeuedSize;
        m_dequeuedData = nullptr;
        m_dequeuedSize = 0U;

        m_sizeChanged.notify_all();

        return true;
    }

    //! @see push_back()
    auto enqueue(const Element_t* const data, size_t const numElements, int64_t const timeoutUs = 0) -> EStatus
    {
        return push_back(data, numElements, timeoutUs);
    }

    //! Get queue capacity
    size_t capacity() const { return m_bufferCapacity; }

    //! Get queue size
    size_t size()
    {
        // coverity[autosar_cpp14_m0_1_3_violation] RFD Pending: TID-1995
        std::lock_guard<std::mutex> const lock{m_mutex};

        return sizeUnsafe();
    }

    //! Return how many elements can be inserted until the container is full.
    size_t available()
    {
        size_t const available{capacity() - size()};
        if (available > capacity())
        {
            return 0;
        }
        else
        {
            return available;
        }
    }

    /// True if queue is empty.
    bool empty() { return size() == 0U; }

    /// True if queue is full.
    bool full() { return size() == capacity(); }

    /// True if container can fit numElements more elements.
    bool hasSpaceFor(size_t const numElements) { return available() >= numElements; }

private:
    size_t m_producerHead;   // supporting modulo behavior - m_producerHead % m_bufferCapacity is actual head of producer in queue, type size_t supports valid wrap around at maximum value. (unsigned integer)
    size_t m_consumerHead;   // supporting modulo behavior - m_consumerHead % m_bufferCapacity is actual head of consumer in queue, type size_t supports valid wrap around at maximum value. (unsigned integer)
    size_t m_bufferCapacity; // capacity should be changeable in case assigning one queue to another

    std::mutex m_mutex;
    dw::core::ConditionVariable m_sizeChanged;

    const Element_t* m_dequeuedData;
    size_t m_dequeuedSize;

    std::unique_ptr<Element_t[]> m_buffer;

    size_t sizeUnsafe() const
    {
        if (m_producerHead >= m_consumerHead)
        {
            return m_producerHead - m_consumerHead;
        }
        else
        {
            return std::numeric_limits<decltype(m_producerHead)>::max() - m_consumerHead + m_producerHead;
        }
    }

    size_t availableUnsafe() const
    {
        size_t const available{capacity() - sizeUnsafe()};
        if (available > capacity())
        {
            return 0;
        }
        else
        {
            return available;
        }
    }
    bool emptyUnsafe() const { return sizeUnsafe() == 0U; }
    bool fullUnsafe() const { return sizeUnsafe() == capacity(); }
    bool hasSpaceForUnsafe(size_t const numElements) const { return availableUnsafe() >= numElements; }

    //! Primitive type version
    template <class Q = Element_t, typename std::enable_if<std::is_pod<Q>::value, bool>::type = 0>
    void pushBackInternal(const Q* const data, size_t const numElements, size_t const bufferOffset)
    {
        dw::core::span<Q const> const dataSpan{data, numElements};

        size_t const bufferRemainder{m_bufferCapacity - bufferOffset};

        if (bufferRemainder >= numElements)
        {
            // everything fits beyond bufferOffset
            static_cast<void>(memcpy(&m_buffer[bufferOffset], data, safeMul(numElements, sizeof(Q)).value()));
        }
        else
        {
            // not enough space beyond offset, copy in 2 blocks
            size_t const pushRemainder{numElements - bufferRemainder};

            // fill up to m_bufferCapacity
            static_cast<void>(memcpy(&m_buffer[bufferOffset], data, safeMul(bufferRemainder, sizeof(Q)).value()));

            // copy remainder at buffer beginning
            static_cast<void>(memcpy(m_buffer.get(), &dataSpan[bufferRemainder], safeMul(pushRemainder, sizeof(Q)).value()));
        }
    }

    //! Non-primitive type version
    template <class Q = Element_t, typename std::enable_if<!std::is_pod<Q>::value, bool>::type = 0>
    void pushBackInternal(const Q* data, size_t numElements, size_t bufferOffset)
    {
        size_t bufferRemainder = m_bufferCapacity - bufferOffset;

        if (bufferRemainder >= numElements)
        {
            for (size_t i = 0; i < numElements; i++)
            {
                m_buffer[bufferOffset + i] = data[i];
            }
        }
        else
        {
            size_t pushRemainder = numElements - bufferRemainder;

            for (size_t i = 0; i < bufferRemainder; i++)
            {
                m_buffer[bufferOffset + i] = data[i];
            }
            for (size_t i = 0; i < pushRemainder; i++)
            {
                m_buffer[i] = data[i + bufferRemainder];
            }
        }
    }

    // This function is a replacement for m_sizeChanged.wait_for(...) call and ensures that the
    // timeout value does not cause an overflow when added to the current time. If the addition
    // overflows, we fall back to a `wait` call without a timeout. This wrapper was introduced
    // mainly to handle DW_TIMEOUT_INFINITE which causes an overflow and undefined behavior.
    template <typename Fn>
    bool waitForSizeChange(std::unique_lock<std::mutex>& lock, int64_t timeoutUs, Fn&& stopWaiting)
    {
        // This check is sufficient to ensure that std::chrono::microseconds
        // can hold int64_t without overflowing because the standard already
        // guarantees that `rep` is a signed integer type.
        static_assert(
            sizeof(std::chrono::microseconds::rep) >= sizeof(int64_t),
            "check if microseconds is big enough to hold 64-bit values");

        if (timeoutUs < 0)
        {
            timeoutUs = 0;
        }

        using microseconds = std::chrono::microseconds;
        using steady_clock = std::chrono::steady_clock;
        using time_point   = steady_clock::time_point;
        using duration     = steady_clock::duration;

        // If clock is higher resolution than microseconds then converting from
        // microsecond time to clock's duration type might overflow and vice versa.
        // `common_duration_t` represents the coarser of the two types
        using common_duration_t = std::conditional_t<(microseconds{1} > duration{1}), microseconds, duration>;

        const time_point now{steady_clock::now()};
        const time_point max{time_point::max()};
        const common_duration_t timeout{std::chrono::duration_cast<common_duration_t>(microseconds{timeoutUs})};
        const common_duration_t maxTimeout{std::chrono::duration_cast<common_duration_t>(max - now)};
        if (timeout < maxTimeout)
        {
            return m_sizeChanged.wait_until(lock, now + timeout, std::forward<Fn>(stopWaiting));
        }

        m_sizeChanged.wait(lock, std::forward<Fn>(stopWaiting));
        return true;
    }
};

using ByteQueue = ElementQueue<uint8_t>;

template <size_t Size>
using ByteQueueFixed = ElementQueue<uint8_t, Size>;

} // namespace core
} // namespace dw

#endif // DW_CORE_ELEMENTQUEUE_HPP_
