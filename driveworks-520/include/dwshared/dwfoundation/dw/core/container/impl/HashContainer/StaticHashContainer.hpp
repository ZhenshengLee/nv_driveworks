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

#ifndef DWSHARED_CORE_HASHCONTAINER_STATICHASHCONTAINER_HPP_
#define DWSHARED_CORE_HASHCONTAINER_STATICHASHCONTAINER_HPP_

#include "BaseHashContainer.hpp"
#include "HashContainerRef.hpp"
#include "../../VectorFixed.hpp"
#include <functional>

#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

namespace dw
{
namespace core
{

/// A hash container that has all of its storage inside of the object itself.
// Capacity must be specified at compile-time
// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class TValue, size_t CAPACITY, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class StaticHashMap;

// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, size_t CAPACITY, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class StaticHashSet;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{
/// A vector that contains all of its storage inside of the object itself.
/// Capacity must be specified at compile-time
template <typename T, size_t CAPACITY>
class StaticHashContainerStorage
{
public:
    /// Storage capacity
    static constexpr auto capacity() -> size_t { return CAPACITY; }

    /// Indexed element access
    auto operator[](size_t const index) -> HashEntry<T>&
    {
        return m_data[index];
    }

    /// Indexed const element access
    auto operator[](size_t const index) const -> const HashEntry<T>&
    {
        return m_data[index];
    }

    /// Get span of hash entries
    auto data() -> dw::core::span<HashEntry<T>>
    {
        return dw::core::make_span(m_data);
    }

    /// Get span of const hash entries
    auto data() const -> dw::core::span<const HashEntry<T>>
    {
        return dw::core::make_span(m_data);
    }

    /// Get number of entries in storage
    size_t size() const { return m_size; }

    /// Get refererence to number of entries in storage
    size_t& size() { return m_size; }

private:
    HashEntry<T> m_data[CAPACITY];

    size_t m_size = 0U;
};
}

/// A hash container that has all of its storage inside of the object itself.
/// Capacity must be specified at compile-time
template <class TKey, class TValue, size_t CAPACITY, class THash, class TKeyEqual>
class StaticHashMap : public BaseHashMap<TKey, TValue, THash, TKeyEqual, detail::StaticHashContainerStorage<std::pair<TKey, TValue>, CAPACITY>>
{
public:
    using TElement = std::pair<TKey, TValue>;                                ///< Type of stored elements (key-value pair)
    using TStorage = detail::StaticHashContainerStorage<TElement, CAPACITY>; ///< Type of storage (static)
    using Base     = BaseHashMap<TKey, TValue, THash, TKeyEqual, TStorage>;  ///< Type of base class
    using Base::storage;

    StaticHashMap() = default;

    ~StaticHashMap()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::OutOfBoundsException& e)
        {
            // this would indicate an internal error, not expected.
            LOGSTREAM_ERROR(nullptr) << "~StaticHashMap: Internal error in HashContainer clear: " << e.what() << "\n";
        }
    }

    /// Construction from initializer list
    StaticHashMap(std::initializer_list<TElement> const ilist)
        : Base(ilist)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticHashMap(const BaseHashMap<TKey, TValue, THash, TKeyEqual, D>& other)
        : Base(other)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    StaticHashMap(const StaticHashMap& other)
        : Base(other)
    {
    }

    /// Move constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticHashMap(BaseHashMap<TKey, TValue, THash, TKeyEqual, D>&& other)
        : Base(std::move(other))
    {
    }

    /// Move constructor.
    /// Storage is static so it cannot be moved,
    /// instead each element is moved into this one
    /// and other is cleared
    StaticHashMap(StaticHashMap&& other)
        : Base(std::move(other))
    {
    }

    /// Use generic move/copy operator
    using Base::operator=;

    /// Copy operator
    auto operator=(const StaticHashMap& other) -> StaticHashMap&
    {
        Base::copyFrom(other);
        return *this;
    }

    /// Move operator
    auto operator=(StaticHashMap&& other) -> StaticHashMap&
    {
        Base::moveFrom(std::move(other));
        return *this;
    }

    /// Get HashMapRef that refers to this HashMap
    auto toHashMapRef() -> HashMapRef<TKey, TValue, THash, TKeyEqual>
    {
        return HashMapRef<TKey, TValue, THash, TKeyEqual>{storage().data(), storage().size()};
    }
};

/// A hash container that has all of its storage inside of the object itself.
/// Capacity must be specified at compile-time
template <class TKey, size_t CAPACITY, class THash, class TKeyEqual>
class StaticHashSet : public BaseHashSet<TKey, THash, TKeyEqual, detail::StaticHashContainerStorage<TKey, CAPACITY>>
{
public:
    using TElement = TKey;                                                   ///< Type of stored elements (key only)
    using TStorage = detail::StaticHashContainerStorage<TElement, CAPACITY>; ///< Type of storage (static)
    using Base     = BaseHashSet<TKey, THash, TKeyEqual, TStorage>;          ///< Type of base class
    using Base::storage;

    StaticHashSet() = default;

    /// Destructor
    ~StaticHashSet()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::OutOfBoundsException& e)
        {
            // this would indicate an internal error, not expected.
            LOGSTREAM_ERROR(nullptr) << "~StaticHashSet: Internal error in HashContainer clear: " << e.what() << "\n";
        }
    }

    /// Construction from initializer list
    StaticHashSet(std::initializer_list<TElement> const ilist)
        : Base(ilist)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticHashSet(BaseHashSet<TKey, THash, TKeyEqual, D> const& other)
        : Base(other)
    {
    }

    /// Copy constructor.
    /// All items from the other vector will be copied into this
    StaticHashSet(const StaticHashSet& other)
        : Base(other)
    {
    }

    /// Move constructor.
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticHashSet(BaseHashSet<TKey, THash, TKeyEqual, D>&& other)
        : Base(std::move(other))
    {
    }

    /// Move constructor.
    /// Storage is static so it cannot be moved,
    /// instead each element is moved into this one
    /// and other is cleared.
    StaticHashSet(StaticHashSet&& other)
        : Base(std::move(other))
    {
    }

    /// Use generic move/copy operator
    using Base::operator=;

    /// Copy operator
    auto operator=(const StaticHashSet& other) -> StaticHashSet&
    {
        Base::copyFrom(other);
        return *this;
    }

    /// Move operator
    auto operator=(StaticHashSet&& other) -> StaticHashSet&
    {
        Base::moveFrom(std::move(other));
        return *this;
    }

    /// Get HashSetRef that refers to this HashSet
    auto toHashSetRef() -> HashSetRef<TKey, THash, TKeyEqual>
    {
        return HashSetRef<TKey, THash, TKeyEqual>{storage().data(), storage().size()};
    }
};
}
}

#endif
