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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_HEAPVECTORFIXED_HPP_
#define DW_CORE_HEAPVECTORFIXED_HPP_

#include "BaseVectorFixed.hpp"
#include "../../VectorRef.hpp"

namespace dw
{
namespace core
{

// Forward defines

/// A vector that puts its storage on the heap.
/// Allocation happens at construction only.
template <typename T, bool USE_PINNED>
class HeapVectorFixed;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{

/// Owns storage for size and data. Data is on the heap.
template <typename T, bool USE_PINNED>
class HeapVectorFixedStorage
{
public:
    using TBuffer     = typename std::aligned_storage<sizeof(T), alignof(T)>::type; ///< Buffer type (aligned storage)
    using UniqueSpanT = UniqueHostSpan<TBuffer, USE_PINNED>;                        ///< UniqueSpan type

    HeapVectorFixedStorage() = default;

    /// Constructor from capacity
    explicit HeapVectorFixedStorage(size_t const argCapacity)
    {
        if (argCapacity > 0U)
        {
            m_data = UniqueSpanT::allocate(argCapacity);
        }
    }

    /// Copy construct or assign is not allowed
    HeapVectorFixedStorage(const HeapVectorFixedStorage&) = delete;
    auto operator=(const HeapVectorFixedStorage&) -> HeapVectorFixedStorage& = delete;

    /// Move construct
    HeapVectorFixedStorage(HeapVectorFixedStorage&& other)
        : m_data(std::move(other.m_data))
        , m_size(other.m_size)
    {
        other.m_size = 0U;
    }

    /// Move storage
    auto operator=(HeapVectorFixedStorage&& other) -> HeapVectorFixedStorage&
    {
        if (m_size > 0U)
        {
            throw InvalidStateException("HeapVectorFixedStorage: cannot move with existing items because storage cannot destroy them");
        }
        m_data       = std::move(other.m_data);
        m_size       = other.m_size;
        other.m_size = 0U;
        return *this;
    }

    ~HeapVectorFixedStorage() = default;

    /// Allocates the given capacity
    /// Should only be called once and only if the constructor did not allocate already
    void allocate(size_t const argCapacity)
    {
        if (capacity() == argCapacity)
        {
            return; // No-op
        }

        if (m_data)
        {
            throw InvalidStateException("HeapVectorFixedStorage: cannot call allocate after buffer has been used");
        }

        m_data = UniqueSpanT::allocate(argCapacity);
    }

    /// Get storage capacity
    size_t capacity() const { return m_data->size(); }

    /// Get pointer to data buffer
    auto data() -> T*
    {
        return safeReinterpretCast<T*>(m_data->data()); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Get const pointer to data buffer
    auto data() const -> const T*
    {
        return safeReinterpretCast<const T*>(m_data->data()); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Storage size
    size_t size() const { return m_size; }

    /// Storage size as reference
    size_t& size() { return m_size; }

private:
    typename UniqueSpanT::Type m_data;
    size_t m_size = 0U;
};
}

// See top of file for comments
template <typename T, bool USE_PINNED = false>
class HeapVectorFixed : public BaseVectorFixed<T, detail::HeapVectorFixedStorage<T, USE_PINNED>>
{
public:
    using TStorage = detail::HeapVectorFixedStorage<T, USE_PINNED>; ///< Storage type
    using Base     = BaseVectorFixed<T, TStorage>;                  ///< Base class type

    /// Avoids allocating memory at construction.
    /// Member allocate() must be called to use the vector
    HeapVectorFixed()
        : Base(TStorage{}, nullptr)
    {
    }

    ~HeapVectorFixed()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::ExceptionBase& e)
        {
            LOGSTREAM_ERROR(nullptr) << "~HeapVectorFixed: Exception caught in HeapVectorFixed() destructor: " << e.what() << dw::core::Logger::State::endl;
        }
    }

    /// Allocates the given capacity
    /// Should only be called once and only if the constructor did not allocate already
    void allocate(size_t const argCapacity)
    {
        Base::getStorage().allocate(argCapacity);
    }

    /// Allocates memory but vector will still be empty
    explicit HeapVectorFixed(size_t const argCapacity)
        : Base(TStorage{argCapacity}, nullptr)
    {
    }

    /// create from initializer list
    /// This is not a constructor because it gets confused with Vector(size_t, TValue)
    static auto fromList(const std::initializer_list<T>& val, size_t argCapacity = 0) -> HeapVectorFixed
    {
        if (argCapacity == 0U)
        {
            argCapacity = val.size();
        }
        HeapVectorFixed result{argCapacity};
        result.push_back_range(val);
        return result;
    }

    /// Initializes the vector with size_ copies of T(value)
    template <class TValue = T>
    HeapVectorFixed(size_t const sizeIn, TValue const value)
        : Base(TStorage{sizeIn}, nullptr)
    {
        Base::push_back_range(sizeIn, value);
    }

    /// All items from the other vector will be copied into this
    template <class D>
    explicit HeapVectorFixed(const BaseVectorFixed<T, D>& other)
        : Base(TStorage{other.capacity()}, other)
    {
    }

    /// All items from the other vector will be moved into this
    template <class D>
    explicit HeapVectorFixed(BaseVectorFixed<T, D>&& other)
        : Base(TStorage{other.capacity()}, other.getCapacityHint(), std::move(other))
    {
    }

    /// Copy constructor
    HeapVectorFixed(const HeapVectorFixed& other)
        : Base(TStorage{other.capacity()}, other.getCapacityHint(), other)
    {
    }

    /// Copy assignment
    auto operator=(const HeapVectorFixed& other) -> HeapVectorFixed&
    {
        Base::copyFrom(make_span(other));
        return *this;
    }

    /// Move constructor
    /// Storage is moved, cheap operation
    HeapVectorFixed(HeapVectorFixed&& other)
        // TODO(dwplc): FP - Already use move constructor
        // coverity[autosar_cpp14_a12_8_4_violation]
        : Base(std::move(other.getStorage()), std::move(other.getCapacityHint()))
    {
    }

    /// Initialize from a span by copying each element
    explicit HeapVectorFixed(const span<const T> dataToCopy)
        : Base(TStorage{dataToCopy.size()}, nullptr)
    {
        this->push_back_range(dataToCopy);
    }

    /// Move assignment
    /// Storage is moved, cheap operation
    auto operator=(HeapVectorFixed&& other) -> HeapVectorFixed&
    {
        Base::clear();
        Base::setStorage(other.getStorage());

        return *this;
    }

    /// Use generic move/copy operator
    using Base::operator=;

    /// Get VectorRef that refers to this container. See VectorRef for more details.
    auto toVectorRef() -> VectorRef<T>
    {
        return VectorRef<T>{make_span(Base::getStorage().data(), Base::getStorage().capacity()), Base::getStorage().size()};
    }
};

// TODO(ahu): rename to PinnedVectorFixed
/// HeapVectorFixed using cuda pinned memory
template <typename T>
using VectorPinned = HeapVectorFixed<T, true>;

} // namespace core
} // namespace dw

#endif
