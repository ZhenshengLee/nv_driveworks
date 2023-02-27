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

#ifndef DWSHARED_CORE_HASHCONTAINER_HASHCONTAINERREF_HPP_
#define DWSHARED_CORE_HASHCONTAINER_HASHCONTAINERREF_HPP_

#include "BaseHashContainer.hpp"
#include "../../VectorFixed.hpp"
#include <functional>

namespace dw
{
namespace core
{

/// A hash container that doesn't own any of its storage.
/// The buffers for elements and size must be specified at construction.
// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class TValue, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class HashMapRef;

// TODO(danielh): switch std::equal_to<TKey> to std::equal_to<> once C++14 is enabled in CUDA
template <class TKey, class THash = DefaultHash<TKey>, class TKeyEqual = std::equal_to<TKey>>
class HashSetRef;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{
/// Storage for HashContainer, it holds only pointers and doesn't own the data.
template <typename T>
class HashContainerRefStorage
{
public:
    /// Empty default construction
    HashContainerRefStorage() // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : HashContainerRefStorage({}, nullptr)
    {
    }

    /// Construction from span and a reference to a size variable
    HashContainerRefStorage(span<HashEntry<T>> data_, size_t& size_) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : HashContainerRefStorage(data_, &size_)
    {
    }

private:
    /// Construction from span and a pointer to a size variable
    HashContainerRefStorage(span<HashEntry<T>> data_, size_t* size_)
        : m_data(std::move(data_))
        , m_size(size_)
    {
        if (m_size == nullptr)
        {
            return;
        }

        if (*m_size > m_data.size())
        {
            // TODO(dwplc): FP - "Using basic numerical type "char" rather than a typedef that includes size and signedness information."
            // coverity[autosar_cpp14_a3_9_1_violation]
            throw InvalidArgumentException("HashContainerRefStorage: the given size (", *m_size, ") is larger than the capacity (", m_data.size(), ")");
        }
    }

public:
    /// Indexed entry access
    auto operator[](size_t index) -> HashEntry<T>&
    {
        return m_data[index];
    }

    /// Indexed entry access
    auto operator[](size_t index) const -> const HashEntry<T>&
    {
        return m_data[index];
    }

    /// Get span of entries
    auto data() -> span<HashEntry<T>>
    {
        return m_data;
    }

    /// Get span of const entries
    auto data() const -> span<const HashEntry<T>>
    {
        return m_data;
    }

    /// Get storage capacity
    size_t capacity() const { return m_data.size(); }

    /// Get current number of entries
    size_t size() const
    {
        if (m_size == nullptr)
        {
            throw InvalidStateException("HashContainerRefStorage: Please initialize the reference with valid data and size");
        }
        return *m_size;
    }

    /// Get reference to current number of entries
    size_t& size()
    {
        if (m_size == nullptr)
        {
            throw InvalidStateException("HashContainerRefStorage: Please initialize the reference with valid data and size");
        }
        return *m_size;
    }

private:
    /// span of enries
    span<HashEntry<T>> m_data;

    /// pointer to size variable
    size_t* m_size;
};
}

/// A hash container that doesn't own any of its storage.
/// The buffers for elements and size must be specified at construction.
template <class TKey, class TValue, class THash, class TKeyEqual>
class HashMapRef : public BaseHashMap<TKey, TValue, THash, TKeyEqual, detail::HashContainerRefStorage<std::pair<TKey, TValue>>>
{
public:
    using TElement = std::pair<TKey, TValue>;                               ///< Type of stored elements (key-value pairs)
    using TStorage = detail::HashContainerRefStorage<TElement>;             ///< Type of storage (reference)
    using Base     = BaseHashMap<TKey, TValue, THash, TKeyEqual, TStorage>; ///< Type of base class
    using Base::storage;

    /// Construction from span and a reference to a size variable
    HashMapRef(span<HashEntry<TElement>> data_, size_t& size_)
        : Base(TStorage{data_, size_})
    {
    }

    /// Empty default construction
    HashMapRef()
        : Base(TStorage())
    {
    }

    /// Returns true if no container is referenced.
    bool isNull() const
    {
        return Base::storage().data().data() == nullptr;
    }
};

/// A hash container that doesn't own any of its storage.
/// The buffers for elements and size must be specified at construction.
template <class TKey, class THash, class TKeyEqual>
class HashSetRef : public BaseHashSet<TKey, THash, TKeyEqual, detail::HashContainerRefStorage<TKey>>
{
public:
    using TElement = TKey;                                          ///< Type of stored elements (key only)
    using TStorage = detail::HashContainerRefStorage<TElement>;     ///< Type of storage (reference)
    using Base     = BaseHashSet<TKey, THash, TKeyEqual, TStorage>; ///< Type of base class
    using Base::storage;

    /// Construction from span and a reference to a size variable
    HashSetRef(span<HashEntry<TElement>> data_, size_t& size_)
        : Base(TStorage{data_, size_})
    {
    }

    /// Empty default construction
    HashSetRef()
        : Base(TStorage())
    {
    }

    /// Returns true if no container is referenced.
    bool isNull() const
    {
        return Base::storage().data().data() == nullptr;
    }
};
}
}

#endif
