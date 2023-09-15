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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_HASHCONTAINER_HEAPHASHCONTAINER_HPP_
#define DWSHARED_CORE_HASHCONTAINER_HEAPHASHCONTAINER_HPP_

#include "BaseHashContainer.hpp"
#include "HashContainerRef.hpp"
#include "../../VectorFixed.hpp"
#include <functional>

#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

namespace dw
{
namespace core
{

/// A hash container that has all of its storage in the heap.
/// Capacity can be specified at runtime. Moving operator is cheap.
// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class TValue, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class HeapHashMap;

// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class HeapHashSet;

// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class TValue, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
using HashMap = HeapHashMap<TKey, TValue, THash, TKeyEqual>;

// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
using HashSet = HeapHashSet<TKey, THash, TKeyEqual>;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{
/// Storage that keeps items in the heap.
/// Capacity can be specified at runtime. Moving operator is cheap.
template <typename T>
class HeapHashContainerStorage
{
public:
    HeapHashContainerStorage() = default;

    /// Construction with capacity
    explicit HeapHashContainerStorage(size_t const capacity)
    {
        if (capacity > 0U)
        {
            m_data = makeUniqueSpan<HashEntry<T>>(capacity);
        }
    }

    /// Copy construct
    HeapHashContainerStorage(const HeapHashContainerStorage& other)
        : m_data(nullptr)
        , m_size(other.m_size)
    {
        if (other.capacity() > 0U)
        {
            m_data = makeUniqueSpan<HashEntry<T>>(other.capacity());
        }

        // TODO(dwplc): FP - not using '::memcpy' but 'memcpy(dw::core::span<Td> dst, dw::core::span<Ts> src)'
        // coverity[autosar_cpp14_a12_0_2_violation]
        memcpy(m_data.get(), other.m_data.get());
    }

    /// Copy operator
    auto operator=(const HeapHashContainerStorage& other) -> HeapHashContainerStorage&
    {
        memcpy(m_data.get(), other.m_data.get());
        m_size = other.m_size;
        return *this;
    }

    /// Move constructor
    HeapHashContainerStorage(HeapHashContainerStorage&& other)
        : m_data(std::move(other.m_data))
        , m_size(other.m_size)
    {
        other.m_size = 0U;
    }

    /// Move operator
    auto operator=(HeapHashContainerStorage&& other) -> HeapHashContainerStorage&
    {
        if (this != &other)
        {
            m_data       = std::move(other.m_data);
            m_size       = other.m_size;
            other.m_size = 0U;
        }
        return *this;
    }

    /// Destructor
    ~HeapHashContainerStorage() = default;

    /// Allocate capacity on the heap
    void allocate(size_t const capacityI)
    {
        if (capacity() == capacityI)
        {
            return; // No-op
        }

        if (m_data)
        {
            throw InvalidStateException("HeapHashContainerStorage: cannot call allocate after buffer has been used");
        }
        m_data = makeUniqueSpan<HashEntry<T>>(capacityI);
    }

    /// Indexed entry access in storage
    auto operator[](size_t const index) -> HashEntry<T>&
    {
        return m_data[index];
    }

    /// Indexed const entry access in storage
    auto operator[](size_t const index) const -> const HashEntry<T>&
    {
        return m_data[index];
    }

    /// Get span of hash entries
    auto data() -> span<HashEntry<T>>
    {
        return m_data.get();
    }

    /// Get span of const hash entries
    auto data() const -> span<const HashEntry<T>>
    {
        return m_data.get();
    }

    /// Storage capacity
    size_t capacity() const { return m_data->size(); }

    /// Get number of entires currently in storage
    size_t size() const { return m_size; }

    /// Get reference to number of entires currently in storage
    size_t& size() { return m_size; }

private:
    UniqueSpan<HashEntry<T>> m_data;

    size_t m_size = 0U;
};
}

/// A hash map container that has all of its storage in the heap.
/// Capacity can be specified at runtime. Moving operator is cheap.
template <class TKey, class TValue, class THash, class TKeyEqual>
class HeapHashMap : public BaseHashMap<TKey, TValue, THash, TKeyEqual, detail::HeapHashContainerStorage<std::pair<TKey, TValue>>>
{
public:
    using TElement = std::pair<TKey, TValue>;                               ///< Type of stored elements (key-value pairs)
    using TStorage = detail::HeapHashContainerStorage<TElement>;            ///< Type of storage (heap)
    using Base     = BaseHashMap<TKey, TValue, THash, TKeyEqual, TStorage>; ///< Type of base class
    using Base::storage;

    /// Empty default constructor
    HeapHashMap()
        : Base()
    {
    }

    /// Construct with capacity
    explicit HeapHashMap(size_t const capacity)
        : Base(TStorage(capacity))
    {
    }

    /// Destructor
    ~HeapHashMap()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::OutOfBoundsException& e)
        {
            // this would indicate an internal error, not expected.
            LOGSTREAM_ERROR(nullptr) << "~HeapHashMap: Internal error in HashContainer clear: " << e.what() << "\n";
        }
    }

    /// Allocates the given capacity
    /// Should only be called once and only if the constructor did not allocate already
    void allocate(size_t const capacity)
    {
        storage().allocate(capacity);
    }

    /// Construction from initializer list
    HeapHashMap(std::initializer_list<TElement> const ilist)
        : Base(TStorage(static_cast<size_t>(std::ceil(Base::INIT_LIST_LOAD_FACTOR * static_cast<float32_t>(ilist.size())))), ilist)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit HeapHashMap(const BaseHashMap<TKey, TValue, THash, TKeyEqual, D>& other)
        : Base(other)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    HeapHashMap(const HeapHashMap& other)
        : Base(other)
    {
    }

    /// Move constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit HeapHashMap(BaseHashMap<TKey, TValue, THash, TKeyEqual, D>&& other)
        : Base(std::move(other))
    {
    }

    /// Move constructor, cheap
    HeapHashMap(HeapHashMap&& other) = default;

    /// Use generic move/copy operator
    using Base::operator=;

    /// Copy operator
    auto operator=(const HeapHashMap& other) -> HeapHashMap&
    {
        Base::copyFrom(other);
        return *this;
    }

    /// Move operator
    auto operator=(HeapHashMap&& other) -> HeapHashMap&
    {
        storage() = std::move(other.storage());
        return *this;
    }

    /// Get HashMapRef that refers to this HashMap
    auto toHashMapRef() -> HashMapRef<TKey, TValue, THash, TKeyEqual>
    {
        return HashMapRef<TKey, TValue, THash, TKeyEqual>{storage().data(), storage().size()};
    }

    /// Swap with other HeapHashMap
    void swap(HeapHashMap& other)
    {
        std::swap(storage(), other.storage());
    }
};

/// A hash set container that has all of its storage in the heap.
/// Capacity can be specified at runtime. Moving operator is cheap.
template <class TKey, class THash, class TKeyEqual>
class HeapHashSet : public BaseHashSet<TKey, THash, TKeyEqual, detail::HeapHashContainerStorage<TKey>>
{
public:
    using TElement = TKey;                                          ///< Type of stored elements (key only)
    using TStorage = detail::HeapHashContainerStorage<TElement>;    ///< Type of storage (heap)
    using Base     = BaseHashSet<TKey, THash, TKeyEqual, TStorage>; ///< Type of base class
    using Base::storage;

    /// Empty default construction
    HeapHashSet()
        : Base()
    {
    }

    /// Construction with capacity
    explicit HeapHashSet(size_t const capacity)
        : Base(TStorage(capacity))
    {
    }

    /// Destructor
    ~HeapHashSet()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::OutOfBoundsException& e)
        {
            // this would indicate an internal error, not expected.
            LOGSTREAM_ERROR(nullptr) << "~HeapHashSet: Internal error in HashContainer clear: " << e.what() << "\n";
        }
    }

    /// Allocates the given capacity.
    /// Should only be called once and only if the constructor did not allocate already
    void allocate(size_t const capacity)
    {
        storage().allocate(capacity);
    }

    /// Create from initializer list.
    /// Initializer list constructor is not exposed because it gets confused with HeapHashSet(size_t capacity)
    static auto fromList(const std::initializer_list<TElement>& ilist) -> HeapHashSet
    {
        return HeapHashSet(ilist);
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit HeapHashSet(BaseHashSet<TKey, THash, TKeyEqual, D> const& other)
        : Base(other)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    HeapHashSet(const HeapHashSet& other)
        : Base(other)
    {
    }

    /// Move constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit HeapHashSet(BaseHashSet<TKey, THash, TKeyEqual, D>&& other)
        : Base(std::move(other))
    {
    }

    /// Move constructor, cheap
    HeapHashSet(HeapHashSet&& other)
        : Base(std::move(other))
    {
    }

    /// Use generic move/copy operator
    using Base::operator=;

    /// Copy operator
    auto operator=(const HeapHashSet& other) -> HeapHashSet&
    {
        Base::copyFrom(other);
        return *this;
    }

    /// Move operator
    auto operator=(HeapHashSet&& other) -> HeapHashSet&
    {
        storage() = std::move(other.storage());
        return *this;
    }

    /// Get HashSetRef that refers to this HeapHashSet
    auto toHashSetRef() -> HashSetRef<TKey, THash, TKeyEqual>
    {
        return HashSetRef<TKey, THash, TKeyEqual>{storage().data(), storage().size()};
    }

    /// Swap with other HeapHashSet
    void swap(HeapHashSet& other)
    {
        std::swap(storage(), other.storage());
    }

private:
    /// Construction from initializer list
    HeapHashSet(const std::initializer_list<TElement>& ilist)
        : Base(TStorage(static_cast<size_t>(std::ceil(Base::INIT_LIST_LOAD_FACTOR * static_cast<float32_t>(ilist.size())))), ilist)
    {
    }
};
}
}

#endif
