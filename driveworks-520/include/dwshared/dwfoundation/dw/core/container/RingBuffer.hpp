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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_RINGBUFFER_HPP_
#define DWSHARED_CORE_RINGBUFFER_HPP_

#include "StorageFixed.hpp"
#include "VectorFixed.hpp"

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/language/Optional.hpp>

#include <type_traits>
#include <functional>
#include <cstring>

namespace dw
{
namespace core
{

template <class T>
constexpr auto ternary_operator(bool const condition, T const valueIfTrue, T const valueIfFalse) -> T
{
    return condition ? valueIfTrue : valueIfFalse;
};

/**
 * A ring buffer implementation of variable size.
 * Note that this component is not thread-safe and that synchronisation would
 * be the responsibility of the user.
 */
template <class T, size_t CapacityAtCompileTime_ = 0>
class RingBuffer
{
public:
    using value_type = T; ///< Type of stored elements

    template <class TT, class RingBufferT>
    class iteratorT;
    using iterator       = iteratorT<T, RingBuffer>;             ///< Type of iterator
    using const_iterator = iteratorT<const T, const RingBuffer>; ///< Type of const-iterator

    /// This is the storage capacity of the RingBuffer. If 0, the capacity is defined at runtime.
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t CapacityAtCompileTime{CapacityAtCompileTime_}; // clang-tidy NOLINT(readability-identifier-naming)

private:
    /// Must allocate one extra item to make sure that m_front and m_backEnd don't overlap when full
    static constexpr bool IS_DYNAMICALLY_ALLOCATED{CapacityAtCompileTime == 0U};
    static constexpr size_t CAPACITY_AT_COMPILE_TIME_PLUS_1{CapacityAtCompileTime + 1U};

    static constexpr size_t STATIC_ALLOCATION_SIZE{ternary_operator<size_t>(IS_DYNAMICALLY_ALLOCATED, 0U, CAPACITY_AT_COMPILE_TIME_PLUS_1)};

    using TStorage = StorageFixed<T, STATIC_ALLOCATION_SIZE>;

    /**
     * @tparam  arithmeticType            arithmetic type, expected to be unsigned
     * @return  true if arithmeticType is unsigned and CAPACITY is in its range, false otherwise
     */
    template <typename arithmeticType>
    constexpr static bool isUnsignedWithEnoughRangeForCapacity()
    {
        static_assert(!IS_DYNAMICALLY_ALLOCATED,
                      "Deviation requires capacity to be set at compile-time");
        return std::is_unsigned<arithmeticType>::value &&
               CapacityAtCompileTime <= core::numeric_limits<arithmeticType>::max();
    }

public:
    // -----------------------------------------------------------------------------
    // Constructors / Destructors
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    /// Empty default constructor
    template <typename = void>
    RingBuffer()
        : m_storage{STATIC_ALLOCATION_SIZE}
        , m_frontIdx(0U)
        , m_backEndIdx(0U)
        , m_size(0U)
    {
    }

    // -----------------------------------------------------------------------------
    /// Empty default constructor
    explicit RingBuffer(size_t const capacity) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
        : RingBuffer(safeAdd(capacity, 1U).value(), 0U, 0U, 0U)
    {
        if (CapacityAtCompileTime_ != 0U)
        {
            // TODO(dwplc): FP - ExceptionWithStackTrace is not casting to non-const
            // coverity[cert_str30_c_violation]
            throw InvalidStateException("RingBuffer: this container has been defined as using statically allocated storage (CapacityAtCompileTime!=0). This constructor only applies to dynamic storage. Call one of the other constructors instead, e.g. RingBuffer().");
        }
    }

    // -----------------------------------------------------------------------------
    /// Enable copy-semantics from other instance
    RingBuffer(const RingBuffer& other) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
        : RingBuffer(other.m_storage.size(), 0U, 0U, 0U)
    {
        // copy
        *this = other;
    }

    // -----------------------------------------------------------------------------
    /// Enable move-semantics from other instance
    RingBuffer(RingBuffer&& other) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
        : RingBuffer(STATIC_ALLOCATION_SIZE, 0U, 0U, 0U)
    {
        // move
        *this = std::move(other);
    }

    // -----------------------------------------------------------------------------
    /// Destructor
    ~RingBuffer()
    {
        // Destroy all
        try
        {
            for (auto& item : *this)
            {
                // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
                item.~T();
            }
        }
        catch (const dw::core::ExceptionBase& e)
        {
            LOGSTREAM_ERROR(nullptr) << "~RingBuffer: Exception caught in RingBuffer() destructor: " << e.what() << dw::core::Logger::State::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Member functions
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    ///
    /// Resize ring buffer to given number of elements.
    /// Must be called before adding any elements.
    /// If CapacityAtCompileTime!=0, this method will throw an exception
    /// if the requested size doesn't match the size specified at compile time.
    ///
    void allocate(size_t const capacity)
    {
        if (!empty())
        {
            throw InvalidStateException("RingBuffer: cannot call allocate after buffer has been used");
        }
        m_storage.allocate(safeAdd(capacity, 1U).value());
        m_frontIdx   = 0U;
        m_backEndIdx = 0U;
    }

    // -----------------------------------------------------------------------------
    ///
    /// Ring buffer could be reallocated with external buffer.
    /// Get the required external buffer size for the ring buffer.
    /// This can be called to get the required size
    /// when creating external buffer outside.
    ///
    static size_t getRequiredExternalBufSize(size_t const capacity)
    {
        size_t size{0};
        // Required size for TStorage in ring buffer
        // Note: here one more than capacity for TStorage so that m_backEnd can always point to an invalid element
        size += safeAdd(capacity, 1U).value() * sizeof(T);
        return size;
    }

    // -----------------------------------------------------------------------------
    ///
    /// Reallocate the storage of the ring buffer with an external buffer.
    /// Must be called before adding any elements.
    /// If CapacityAtCompileTime!=0, this method will throw an exception
    /// if the requested size doesn't match the size specified at compile time.
    ///
    void reallocateWithExternalBuf(void* externalBuf, size_t const externalBufSize, size_t const capacity)
    {
        if (externalBuf == nullptr)
        {
            throw InvalidArgumentException("RingBuffer: externalBuf is nullptr");
        }
        if (!empty())
        {
            throw InvalidStateException("RingBuffer: cannot call reallocateWithExternalBuf after buffer has been used");
        }
        if (externalBufSize < getRequiredExternalBufSize(capacity))
        {
            throw InvalidArgumentException("RingBuffer: externalBufSize is smaller than the required size according to capacity");
        }
        m_frontIdx   = 0U;
        m_backEndIdx = 0U;
        // Note: here one more than capacity for TStorage so that m_backEnd can always point to an invalid element
        // Due to storage alignment, storage must reallocate from the start of externalBuf without any offset
        m_storage.reallocateWithExternalBuf(externalBuf, externalBufSize, safeAdd(capacity, 1U).value());
    }

    // -----------------------------------------------------------------------------
    // operator=
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    /// Copy operator
    auto operator=(const RingBuffer<T, CapacityAtCompileTime_>& other) -> RingBuffer<T, CapacityAtCompileTime_>&
    {
        if (this == &other)
        {
            return *this;
        }
        // allow exact copy only
        if (m_storage.size() != other.m_storage.size())
        {
            // TODO(dwplc): FP - ExceptionWithStackTrace is not casting to non-const
            // coverity[cert_str30_c_violation]
            throw BufferFullException("RingBuffer: cannot copy because destination storage "
                                      "size is not equal to the source storage size.");
        }

        // destruct existing elements
        for (T& elem : *this)
        {
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            elem.~T();
        }

        m_frontIdx   = other.m_frontIdx;
        m_backEndIdx = other.m_backEndIdx;
        m_size       = other.m_size;

        // copy construct elements in identical slots
        if (m_frontIdx <= m_backEndIdx)
        {
            // valid range is [m_frontIdx, m_backEndIdx)
            for (size_t i{m_frontIdx}; i < m_backEndIdx; ++i)
            {
                new (&m_storage[i]) T(other.m_storage[i]);
            }
        }
        else
        {
            // valid ranges are [0, m_backIdx) and [m_frontIdx, capacity)
            for (size_t i{0U}; i < m_backEndIdx; ++i)
            {
                new (&m_storage[i]) T(other.m_storage[i]);
            }
            for (size_t i{m_frontIdx}; i < m_storage.size(); ++i)
            {
                new (&m_storage[i]) T(other.m_storage[i]);
            }
        }

        return *this;
    }

    // -----------------------------------------------------------------------------
    /// Move operator
    auto operator=(RingBuffer<T, CapacityAtCompileTime_>&& other) noexcept -> RingBuffer<T, CapacityAtCompileTime_>&
    {
        if (this == &other)
        {
            return *this;
        }

        try
        {
            // destruct existing elements
            for (T& elem : *this)
            {
                // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
                elem.~T();
            }

            m_frontIdx   = other.m_frontIdx;
            m_backEndIdx = other.m_backEndIdx;
            m_size       = other.m_size;

            if (CapacityAtCompileTime_ == 0U)
            {
                // move storage
                m_storage = std::move(other.m_storage);
            }
            else
            {
                // compile-time sized storage can't be moved.
                // move elements to identical slots
                if (m_frontIdx <= m_backEndIdx)
                {
                    // valid range is [m_frontIdx, m_backEndIdx)
                    for (size_t i{m_frontIdx}; i < m_backEndIdx; ++i)
                    {
                        new (&m_storage[i]) T(std::move(other.m_storage[i]));
                    }
                }
                else
                {
                    // valid ranges are [0, m_backIdx) and [m_frontIdx, capacity)
                    for (size_t i{0U}; i < m_backEndIdx; ++i)
                    {
                        new (&m_storage[i]) T(std::move(other.m_storage[i]));
                    }
                    for (size_t i{m_frontIdx}; i < m_storage.size(); ++i)
                    {
                        new (&m_storage[i]) T(std::move(other.m_storage[i]));
                    }
                }
            }
        }
        catch (const dw::core::ExceptionBase& e)
        {
            LOGSTREAM_ERROR(nullptr) << "RingBuffer: Exception caught in RingBuffer move assignment: " << e.what() << dw::core::Logger::State::endl;
        }

        other.m_size       = 0U;
        other.m_frontIdx   = 0U;
        other.m_backEndIdx = 0U;

        return *this;
    }

    // -----------------------------------------------------------------------------
    /// Return number of elements in the buffer
    size_t size() const { return m_size; }

    /**
     * @tparam  sizeType            unsigned integer type
     * @details asserts at compile-time that capacity fits to sizeType
     * @return  size of buffer if capacity fits to sizeType
     */
    template <typename sizeType>
    sizeType narrowSize() const
    {
        static_assert(isUnsignedWithEnoughRangeForCapacity<sizeType>(), "Cast may lead to data loss.");

        // TODO(baltin): write deviation record
        // coverity[cert_int31_c_violation] RFD Accepted: TID-2326
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Accepted: TID-2326
        return static_cast<sizeType>(size());
    }

    // -----------------------------------------------------------------------------
    /// Return frontIdx of the buffer
    size_t frontIdx() const { return m_frontIdx; }

    // -----------------------------------------------------------------------------
    /// Return backEndIdx of the buffer
    size_t backEndIdx() const { return m_backEndIdx; }

    // -----------------------------------------------------------------------------
    /// Set number of elements, frontIdx and backEndIdx in the buffer, this should be called
    /// only when ring buffer is reallocated with external buffer
    void setSizeAndIdx(size_t const sizeIn, size_t const frontIdx, size_t const backEndIdx)
    {
        if (sizeIn > capacity())
        {
            throw InvalidArgumentException("RingBuffer: setSizeAndIdx input sizeIn is larger than capacity.");
        }
        if (frontIdx >= m_storage.size() || backEndIdx >= m_storage.size())
        {
            throw InvalidArgumentException("RingBuffer: setSizeAndIdx input frontIdx or backEndIdx is out of bound.");
        }
        if (((frontIdx <= backEndIdx) && sizeIn != safeSub(backEndIdx, frontIdx).value()) ||
            ((frontIdx > backEndIdx) && sizeIn != (safeSub(safeAdd(safeAdd(backEndIdx, capacity()).value(), 1U).value(), frontIdx).value())))
        {
            throw InvalidArgumentException("RingBuffer: setSizeAndIdx input sizeIn, frontIdx and backEndIdx do not match.");
        }
        m_frontIdx   = frontIdx;
        m_backEndIdx = backEndIdx;
        m_size       = sizeIn;
    }

    // -----------------------------------------------------------------------------
    /// Return number of valid elements currently in the buffer
    size_t available() const { return safeSub(capacity(), m_size).value(); }

    // -----------------------------------------------------------------------------
    /// Return the maximum number of elements this buffer can contain
    /// Note: this is one less than the storage capacity so that m_backEnd can always point to an invalid element
    size_t capacity() const { return m_storage.size() > 0U ? m_storage.size() - 1U : 0U; }

    // -----------------------------------------------------------------------------
    /// Return true if there are no valid elements in the buffer
    /// Note: an empty buffer fulfills both m_size==0 and m_front==m_backEnd
    bool empty() const { return m_size == 0U; }

    // -----------------------------------------------------------------------------
    /// Return true if size() == capacity
    bool full() const { return size() == capacity(); }

    // -----------------------------------------------------------------------------
    /// Clear buffer, i.e. remove all elements.
    void clear()
    {
        for (T& elem : *this)
        {
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            elem.~T();
        }
        m_frontIdx   = 0U;
        m_backEndIdx = 0U;
        m_size       = 0U;
    }

    // -----------------------------------------------------------------------------
    /// Get begin iterator
    // TODO(dwplc): FP - can't make this 'const' for non-const RingBuffer iterator that contains
    //                   non-const reference to *this, compiling error
    // coverity[autosar_cpp14_m9_3_3_violation]
    auto begin() -> iterator { return iterator(this, m_frontIdx); }

    // -----------------------------------------------------------------------------
    /// Get end iterator
    // TODO(dwplc): FP - can't make this 'const' for non-const RingBuffer iterator that contains
    //                   non-const reference to *this, compiling error
    // coverity[autosar_cpp14_m9_3_3_violation]
    auto end() -> iterator { return iterator(this, m_backEndIdx); }

    // -----------------------------------------------------------------------------
    /// Get begin const-iterator
    auto begin() const -> const_iterator { return const_iterator(this, m_frontIdx); }

    // -----------------------------------------------------------------------------
    /// Get end const-iterator
    auto end() const -> const_iterator { return const_iterator(this, m_backEndIdx); }

    // -----------------------------------------------------------------------------
    /// Create an object directly on the storage at the back of the container if the container is not full.
    /// If the container is full, return value is false and no element was added.
    template <typename... Args>
    bool emplace_back_maybe(Args&&... args)
    {
        bool ret{false};

        if (m_size < capacity())
        {
            new (&(span<T>(m_storage.data(), m_backEndIdx + 1U)[m_backEndIdx])) T(std::forward<Args>(args)...);

            m_backEndIdx = incrementIdx(m_backEndIdx);
            m_size++;
            ret = true;
        }

        return ret;
    }

    // -----------------------------------------------------------------------------
    /**
     * Push new element at the end.
     * @return If buffer is full this method will return false and will not add the element.
     **/
    inline bool push_back_maybe(T&& element)
    {
        return emplace_back_maybe(std::move(element));
    }

    // -----------------------------------------------------------------------------
    /// Copy construct new element at the end. Returns false if the container is full.
    // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
    inline bool push_back_maybe(const T& element) { return emplace_back_maybe(element); }

    // -----------------------------------------------------------------------------
    /// Create an object directly on the storage at the back of the container.
    /// Returns a reference to the created object.
    /// If the container is full, the front element is removed.
    template <typename... Args>
    auto emplace_back(Args&&... args) -> T&
    {
        if (capacity() == 0U)
        {
            // TODO(dwplc): FP - ExceptionWithStackTrace is not casting to non-const
            // coverity[cert_str30_c_violation]
            throw BufferFullException("RingBuffer::emplace_back: cannot emplace_back into an RingBuffer "
                                      "with zero capacity. Call allocate() first before pushing "
                                      "objects into the container.");
        }

        if (full())
        {
            // Destroy front
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            m_storage[m_frontIdx].~T();

            new (&(span<T>(m_storage.data(), m_backEndIdx + 1U)[m_backEndIdx])) T(std::forward<Args>(args)...);

            // Update indices
            m_backEndIdx = m_frontIdx; // Same as incrementPtr(m_backEnd);
            m_frontIdx   = incrementIdx(m_frontIdx);
        }
        else
        {
            static_cast<void>(emplace_back_maybe(std::forward<Args>(args)...));
        }

        return back();
    }

    // -----------------------------------------------------------------------------
    /**
     * Push new element at the end. If buffer is full, element in the front
     * will be removed.
     * @return A reference to the created object.
     * @note It is guaranteed that this method always succeeds, however data,
     *       which has not been picked up, will get lost. It is an exception to use these functions on a container
     *       of zero capacity.
     */
    inline auto push_back(T&& element) -> T&
    {
        return emplace_back(std::move(element));
    }

    // -----------------------------------------------------------------------------
    /// Copy construct new element at the end.
    /// If the container is full, the front element is removed.
    // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
    auto push_back(const T& element) -> T& { return emplace_back(element); }

    // -----------------------------------------------------------------------------
    /// Create an object directly on the storage at the front of the container if the container is not full.
    /// If the container is full, return value is false and no element was added.
    template <typename... Args>
    bool emplace_front_maybe(Args&&... args)
    {
        if (m_size == capacity())
        {
            return false;
        }

        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const newFrontIdx = decrementIdx(m_frontIdx);
        new (&m_storage[newFrontIdx]) T(std::forward<Args>(args)...);

        m_frontIdx = newFrontIdx;
        m_size++;

        return true;
    }

    // -----------------------------------------------------------------------------
    /**
     * Push new element at the front.
     * @return If buffer is full this method will return false and will not add the element.
     **/
    inline bool push_front_maybe(T&& element)
    {
        return emplace_front_maybe(std::move(element));
    }

    // -----------------------------------------------------------------------------
    /// Copy construct new element at the front. Returns false if the container is full.
    // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
    inline bool push_front_maybe(const T& element) { return emplace_front_maybe(element); }

    // -----------------------------------------------------------------------------
    /// Create an object directly on the storage at the front of the container.
    /// Returns a reference to the created object.
    /// If the container is full, the back element is removed.
    template <typename... Args>
    auto emplace_front(Args&&... args) -> T&
    {
        if (capacity() == 0U)
        {
            throw BufferFullException("RingBuffer::emplace_front: cannot emplace_front into an RingBuffer "
                                      "with zero capacity. Call allocate() first before pushing "
                                      "objects into the container.");
        }

        if (full())
        {
            m_backEndIdx = decrementIdx(m_backEndIdx);

            // Destroy back
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            m_storage[m_backEndIdx].~T();

            // Create new on front
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto const newFrontIdx = decrementIdx(m_frontIdx);
            new (&m_storage[newFrontIdx]) T(std::forward<Args>(args)...);
            m_frontIdx = newFrontIdx;
        }
        else
        {
            static_cast<void>(emplace_front_maybe(std::forward<Args>(args)...));
        }

        return front();
    }

    // -----------------------------------------------------------------------------
    /**
     * Push new element at the front. If buffer is full, element in the back
     * will be removed.
     * @return A reference to the created object.
     * @note It is guaranteed that this method always succeeds, however data,
     *       which has not been picked up, will get lost. It is an exception to use these functions on a container
     *       of zero capacity.
     */
    inline auto push_front(T&& element) -> T&
    {
        return emplace_front(std::move(element));
    }

    // -----------------------------------------------------------------------------
    /// Copy construct new element at the front.
    /// If the container is full, the back element is removed.
    auto push_front(const T& element) -> T& { return emplace_front(element); }

    // -----------------------------------------------------------------------------
    /**
     * Push a set of elements into the buffer.
     * @return Number of elements pushed to the buffer before buffer was filled.
     **/
    size_t push_back_maybe(const T* const element, size_t const num)
    {
        dw::core::span<T const> const elementSpan{element, num};

        if (capacity() == 0U)
        {
            return 0U;
        }

        size_t const consumed{push_back_contiguous(element, num)};

        if ((consumed == num) || (consumed == 0U))
        {
            return consumed;
        }

        return safeAdd(consumed, push_back_contiguous(&elementSpan[consumed], num - consumed)).value();
    }

    // -----------------------------------------------------------------------------
    /**
     * Pop element from the front
     * @note This method return false if buffer is empty
     */
    inline bool pop_front_maybe(T* const element = nullptr)
    {
        if (empty())
        {
            return false;
        }

        if (element != nullptr)
        {
            *element = std::move(m_storage[m_frontIdx]);
        }

        // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
        m_storage[m_frontIdx].~T();

        m_frontIdx = incrementIdx(m_frontIdx);
        m_size--;

        return true;
    }

    // -----------------------------------------------------------------------------
    /**
     * Pop element from the front
     * @note This method throws an exception if the buffer is empty
     */
    inline void pop_front(T* const element = nullptr)
    {
        if (!pop_front_maybe(element))
        {
            // TODO(dwplc): FP - ExceptionWithStackTrace is not casting to non-const
            // coverity[cert_str30_c_violation]
            throw BufferEmptyException("RingBuffer::pop_front: empty");
        }
    }

    // -----------------------------------------------------------------------------
    /**
     * Pop element from the back
     * @note This method return false if buffer is empty
     */
    inline bool pop_back_maybe(T* const element = nullptr)
    {
        if (empty())
        {
            return false;
        }

        size_t const backIdx{decrementIdx(m_backEndIdx)};
        if (element != nullptr)
        {
            *element = std::move(m_storage[backIdx]);
        }

        // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
        m_storage[backIdx].~T();

        m_backEndIdx = backIdx;
        m_size--;

        return true;
    }

    // -----------------------------------------------------------------------------
    /**
     * Same as @see pop_front(), however pops_a predefined number of elements.
     * Return number of elements popped from the front.
     *
     * @param[in] elements Pointer to contiguous memory capably of holding up-to num elements.
     *            if element == nullptr, the buffer is popped, but no copy takes place
     * @param[in] num Number of elements to extract
     */
    size_t pop_front_maybe(T* const elements, size_t num)
    {
        dw::core::span<T> const elementSpan{elements, num};

        if (num == 0U)
        {
            return 0U;
        }
        if (num > size())
        {
            num = size();
        }

        size_t extracted{pop_front_contiguous(elements, num)};

        if (extracted < num)
        {
            T* const elementsPtr{(elements != nullptr) ? &elementSpan[extracted] : nullptr};
            // TODO(rshih): Fix pending violation. Operation may wrap
            // coverity[cert_int30_c_violation]
            extracted += pop_front_contiguous(elementsPtr, num - extracted);
        }

        return extracted;
    }

    // -----------------------------------------------------------------------------
    /// Returns a reference to the front of the ring.
    /// Throws an exception if empty
    auto front() -> T&
    {
#if DW_RUNTIME_CHECKS()
        if (empty())
        {
            throw BufferEmptyException("RingBuffer::front: empty");
        }
#endif
        return m_storage[m_frontIdx];
    }

    // -----------------------------------------------------------------------------
    /// Returns a reference to the front of the ring.
    /// Throws an exception if empty
    auto front() const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        if (empty())
        {
            throw BufferEmptyException("RingBuffer::front: empty");
        }
#endif
        return m_storage[m_frontIdx];
    }

    // -----------------------------------------------------------------------------
    /// Returns a reference to the back of the ring.
    /// Throws an exception if empty
    auto back() -> T&
    {
#if DW_RUNTIME_CHECKS()
        if (empty())
        {
            // TODO(dwplc): FP - ExceptionWithStackTrace is not casting to non-const
            // coverity[cert_str30_c_violation]
            throw BufferEmptyException("RingBuffer::back: empty");
        }
#endif
        return m_storage[decrementIdx(m_backEndIdx)];
    }

    // -----------------------------------------------------------------------------
    /// Returns a reference to the back of the ring.
    /// Throws an exception if empty
    auto back() const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        if (empty())
        {
            throw BufferEmptyException("RingBuffer::back: empty");
        }
#endif
        return m_storage[decrementIdx(m_backEndIdx)];
    }

    // -----------------------------------------------------------------------------
    /// Access element 0-based, i.e. front at 0, back at size()-1
    inline auto operator[](size_t const idx) -> T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= size())
        {
            core::assertException<OutOfBoundsException>("RingBuffer::operator[]: index out of bounds");
        }
#endif

        size_t const linearIdx{safeAdd(m_frontIdx, idx).value() % m_storage.size()};
        return m_storage[linearIdx];
    }

    // -----------------------------------------------------------------------------
    /// Const access element 0-based, i.e. front at 0, back at size()-1
    inline auto operator[](size_t const idx) const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= size())
        {
            core::assertException<OutOfBoundsException>("RingBuffer::operator[]: index out of bounds");
        }
#endif

        size_t const linearIdx{safeAdd(m_frontIdx, idx).value() % m_storage.size()};
        return m_storage[linearIdx];
    }

    // -----------------------------------------------------------------------------
    /// Get contiguous data ranges as up-to two spans ( @see dw::core::span )
    auto data() const -> StaticVectorFixed<span<const T>, 2>
    {
        StaticVectorFixed<span<const T>, 2> rbData{};

        // compute size of contiguous data blocks
        size_t size1{0U};
        size_t size2{0U};

        if (m_frontIdx <= m_backEndIdx)
        {
            // single continguous array
            size1 = m_backEndIdx - m_frontIdx;
        }
        else
        {
            // front to end of internal array
            size1 = m_storage.size() - m_frontIdx;

            // begin of internal array to end
            size2 = m_backEndIdx;
        }

        if (size1 > 0U)
        {
            static_cast<void>(rbData.emplace_back(&m_storage[m_frontIdx], size1));
        }

        if (size2 > 0U)
        {
            static_cast<void>(rbData.emplace_back(m_storage.data(), size2));
        }

        return rbData;
    }

    // -----------------------------------------------------------------------------
    /// Increment index by signed integer
    size_t incrementIdxSigned(size_t const idx, ssize_t const stepSize) const
    {
        if (stepSize >= 0)
        {
            return incrementIdx(idx, static_cast<size_t>(stepSize));
        }
        else
        {
            // TODO(dwplc): FP - abs(stepSize) ensures the same signedness
            // coverity[cert_int31_c_violation]
            return decrementIdx(idx, static_cast<size_t>(std::abs(stepSize))); // coverity complains if we use -stepSize, so using std::abs instead
        }
    }

    // -----------------------------------------------------------------------------
    /// Decrement index by signed integer
    size_t decrementIdxSigned(size_t const idx, ssize_t const stepSize) const
    {
        if (stepSize >= 0)
        {
            return decrementIdx(idx, static_cast<size_t>(stepSize));
        }
        else
        {
            // TODO(dwplc): FP - abs(stepSize) ensures the same signedness
            // coverity[cert_int31_c_violation]
            return incrementIdx(idx, static_cast<size_t>(std::abs(stepSize))); // coverity complains if we use -stepSize, so using std::abs instead
        }
    }

    // -----------------------------------------------------------------------------
    /// Map index in storage memory to RingBuffer index
    inline size_t computeVirtualIndex(size_t const storageIndex) const
    {
        size_t result{0U};
        if (storageIndex >= m_frontIdx)
        {
            result = storageIndex - m_frontIdx;
        }
        else
        {
            // wrap around at end of buffer
            // TODO(dwplc): FP - m_frontIdx is less than m_storage.size() so this equation won't hit overflow or underflow
            // coverity[cert_int30_c_violation]
            result = m_storage.size() - m_frontIdx + storageIndex;
        }
        return result;
    }

    // -----------------------------------------------------------------------------
    //////////////////////////////////////////////////////
    /// Iterators
    template <class TT, class RingBufferT>
    class iteratorT : public std::iterator<std::random_access_iterator_tag, TT>
    {
    public:
        /// Constructor from RingBuffer pointer and element index
        iteratorT(RingBufferT* const parent, std::size_t const index)
            : m_parent(parent)
            , m_index(index)
        {
        }

        ~iteratorT()                = default;       ///< Destructor
        iteratorT(iteratorT const&) = default;       ///< Copy constructor
        iteratorT(iteratorT&&)      = default;       ///< Move constructor
        iteratorT& operator=(iteratorT&&) = default; ///< Move operator
        iteratorT& operator=(iteratorT&) = default;  ///< Copy operator

        /// Increment
        auto operator++() -> iteratorT&
        {
            m_index = m_parent->incrementIdx(m_index);
            return *this;
        }

        /// Decrement
        auto operator--() -> iteratorT&
        {
            m_index = m_parent->decrementIdx(m_index);
            return *this;
        }

        /// Increment by signed integer
        template <typename TNumerical,
                  typename std::enable_if<std::is_signed<TNumerical>::value, TNumerical>::type* = nullptr>
        // TODO(dwplc): FP - basic numerical type "int" isn't used here
        auto operator+=(TNumerical const n) -> iteratorT&
        {
            m_index = m_parent->incrementIdxSigned(m_index, static_cast<ssize_t>(n));
            return *this;
        }

        /// Decrement by signed integer
        template <typename TNumerical,
                  typename std::enable_if<std::is_signed<TNumerical>::value, TNumerical>::type* = nullptr>
        auto operator-=(TNumerical const n) -> iteratorT&
        {
            m_index = m_parent->decrementIdxSigned(m_index, static_cast<ssize_t>(n));
            return *this;
        }

        /// Increment by unsigned integer
        template <typename TNumerical,
                  typename std::enable_if<std::is_unsigned<TNumerical>::value, TNumerical>::type* = nullptr>
        auto operator+=(TNumerical const n) -> iteratorT&
        {
            m_index = m_parent->incrementIdx(m_index, static_cast<size_t>(n));
            return *this;
        }

        /// Decrement by unsigned integer
        template <typename TNumerical,
                  typename std::enable_if<std::is_unsigned<TNumerical>::value, TNumerical>::type* = nullptr>
        auto operator-=(TNumerical n) -> iteratorT&
        {
            m_index = m_parent->decrementIdx(m_index, static_cast<size_t>(n));
            return *this;
        }

        /// Return new incremented iterator
        template <typename TNumerical>
        auto operator+(TNumerical const n) const -> iteratorT
        {
            iteratorT result{*this};
            result += n;
            return result;
        }

        /// Return new decremented iterator
        template <typename TNumerical>
        auto operator-(TNumerical const n) const -> iteratorT
        {
            iteratorT result{*this};
            result -= n;
            return result;
        }

        /// Iterator difference
        dw::ssize_t operator-(const iteratorT& other) const
        {
            std::size_t const d1{m_parent->computeVirtualIndex(m_index)};
            std::size_t const d2{other.m_parent->computeVirtualIndex(other.m_index)};
            return dw::core::narrow<dw::ssize_t>(d1) - dw::core::narrow<dw::ssize_t>(d2);
        }

        /// Equality operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator==(const iteratorT& lhs,
                               const iteratorT& rhs) noexcept
        {
            return lhs.m_index == rhs.m_index;
        }

        /// Inequality operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator!=(const iteratorT& lhs,
                               const iteratorT& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        /// Less-than operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator<(const iteratorT& lhs, const iteratorT& rhs) noexcept
        {
            std::size_t const d1{lhs.m_parent->computeVirtualIndex(lhs.m_index)};
            std::size_t const d2{rhs.m_parent->computeVirtualIndex(rhs.m_index)};
            return d1 < d2;
        }

        /// Greater-than operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator>(const iteratorT& lhs, const iteratorT& rhs) noexcept
        {
            return rhs < lhs;
        }

        /// Less-or-equal operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator<=(const iteratorT& lhs, const iteratorT& rhs) noexcept
        {
            return lhs == rhs || lhs < rhs;
        }

        /// Greater-or-equal operator
        // TODO(dwplc): FP - no symbol with external linkage declare outside of header file
        // coverity[autosar_cpp14_a3_3_1_violation]
        friend bool operator>=(const iteratorT& lhs, const iteratorT& rhs) noexcept
        {
            return lhs == rhs || rhs < lhs;
        }

        /// Get element the iterator refers to.
        auto operator*() const -> TT&
        {
            return (m_parent->m_storage)[m_index];
        }

    private:
        RingBufferT* m_parent; ///< Ringbuffer this iterator belongs to
        std::size_t m_index;   ///< Element index in RingBuffer
        FRIEND_TEST(RingBuffer_Test, EmplaceWhenFull_L0);
    };

private:
    //////////////////////////////////////////////////////
    /// Private members
    ///

    RingBuffer(size_t const internalStorageSize,
               size_t const frontIdx,
               size_t const backEndIdx,
               size_t const size)
        : m_storage(internalStorageSize)
        , m_frontIdx(frontIdx)
        , m_backEndIdx(backEndIdx)
        , m_size(size)
    {
    }

    // -----------------------------------------------------------------------------
    /**
     * Push a set of elements into the buffer until it reaches the physical end or
     * the tail of the buffer. This guarantees that the elements will be contiguous
     * in memory.
     * @return Number of elements pushed to the buffer before it hit an end.
     **/
    size_t push_back_contiguous(const T* const elements, size_t const num)
    {
        if (full() || (num == 0U))
        {
            return 0U;
        }

        size_t const tocopy{std::min(availableToEnd(), num)};
        size_t const consumed{tocopy};

        // Note: this could be specialized for POD to skip construction and simply memcpy the data
        span<T const> const elementSpan{elements, num};
        for (size_t i{0U}; i < tocopy; ++i)
        {
            // TODO(dwplc): FP or refactor code?
            // coverity[autosar_cpp14_a7_1_1_violation]
            T* endPtr{&(span<T>(m_storage.data(), m_backEndIdx + 1U)[m_backEndIdx])};
            new (endPtr) T(elementSpan[i]);
            ++m_backEndIdx;
        }

        if (m_backEndIdx >= m_storage.size())
        {
            m_backEndIdx = 0U;
        }
        safeIncrement(m_size, consumed);

        return consumed;
    }

    // -----------------------------------------------------------------------------
    /**
     * Pop at most num element from the front in one memcopy
     * @param[in] elements Pointer to contiguous memory capably of holding up-to num elements
     * @param[in] num Number of elements to extract
     * @note This method return number of elements extracted.
     */
    size_t pop_front_contiguous(T* const elements, size_t const num)
    {
        if (empty())
        {
            return 0U;
        }

        size_t tocopy{std::min(sizeToEnd(), num)};
        size_t const consumed{tocopy};

        // Note: this could be specialized for POD to skip construction and simply memcpy the data
        if (elements != nullptr)
        {
            // Move and destroy
            span<T> const elementSpan{elements, num};
            for (size_t i{0U}; i < tocopy; ++i)
            {
                elementSpan[i] = std::move(m_storage[m_frontIdx]);
                // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
                m_storage[m_frontIdx].~T();
                ++m_frontIdx;
            }
        }
        else
        {
            // Only destroy
            while (tocopy > 0U)
            {
                // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
                m_storage[m_frontIdx].~T();
                ++m_frontIdx;
                --tocopy;
            }
        }
        if (m_frontIdx >= m_storage.size())
        {
            m_frontIdx = 0U;
        }

        safeDecrement(m_size, consumed);

        return consumed;
    }

    // -----------------------------------------------------------------------------
    /// Return physical size until end of contiguous memory
    /// Makes sure to leave at least one free space before m_front
    inline size_t availableToEnd() const
    {
        if (m_backEndIdx < m_frontIdx)
        {
            return m_frontIdx - m_backEndIdx - 1U;
        }
        else
        {
            size_t const extraSpace{(m_frontIdx == m_storage.size()) ? static_cast<size_t>(1U) : static_cast<size_t>(0U)}; // static_cast<size_t> necessary for AUTOSAR m5_0_3

            // TODO(dwplc): FP - m_backEndIdx is less than m_storage.size() so this equation won't hit overflow or underflow
            // coverity[cert_int30_c_violation]
            return m_storage.size() - m_backEndIdx - extraSpace;
        }
    }

    // -----------------------------------------------------------------------------
    /// Return number of valid elements from m_front to the end of contiguous memory
    inline size_t sizeToEnd() const
    {
        if (m_frontIdx <= m_backEndIdx)
        {
            return m_backEndIdx - m_frontIdx;
        }
        else
        {
            return safeSub(m_storage.size(), m_frontIdx).value();
        }
    }

    // -----------------------------------------------------------------------------
    size_t incrementIdx(size_t const idx) const
    {
        return incrementIdx(idx, 1U);
    }

    // -----------------------------------------------------------------------------
    size_t decrementIdx(size_t const idx) const
    {
        return decrementIdx(idx, 1U);
    }

    // -----------------------------------------------------------------------------
    size_t incrementIdx(size_t const idx, size_t const stepSize) const
    {
        return safeAdd(idx, stepSize).value() % m_storage.size();
    }

    // -----------------------------------------------------------------------------
    size_t decrementIdx(size_t const idx, size_t stepSize) const
    {
        stepSize = stepSize % m_storage.size();
        if (idx >= stepSize)
        {
            return idx - stepSize;
        }
        else
        {
            stepSize -= idx;
            // TODO(dwplc): FP - stepSize is modulo result with m_storage.size(), so doesn't wrap in this operation
            // coverity[cert_int30_c_violation]
            return m_storage.size() - stepSize;
        }
    }

private:
    /// The storage used by the container.
    TStorage m_storage;

    /// Limits of the buffer
    /// An entry is valid if both are true
    ///   m_storage.data() <= entry < m_storage.end()
    ///   m_front <= entry < m_backEnd

    /// Beginning the Ringbuffer, index into m_storage.
    /// Always points to a valid element unless m_front==m_backEnd
    size_t m_frontIdx; // clang-tidy NOLINT(modernize-use-default-member-init)

    /// back+1 of the Ringbuffer, never points to a valid element
    /// The only time when m_front==m_backEnd is when the buffer is empty
    size_t m_backEndIdx; // clang-tidy NOLINT(modernize-use-default-member-init)

    /// Number of valid elements
    /// Note: could be deduced but is cached for simplicity
    size_t m_size; // clang-tidy NOLINT(modernize-use-default-member-init)
    FRIEND_TEST(RingBuffer_Test, EmplaceWhenFull_L0);
    FRIEND_TEST(RingBuffer_Test, MoveConstructorStatic_L0);
    FRIEND_TEST(RingBuffer_Test, PushFront_L0);
};

/// RingBuffer that is sized at compile time and contains all reserved memory
template <class T, size_t CapacityAtCompileTime_>
using RingBufferFixed = RingBuffer<T, CapacityAtCompileTime_>;

} // namespace core
} // namespace dw

#endif // DW_CORE_RINGBUFFER_HPP_
