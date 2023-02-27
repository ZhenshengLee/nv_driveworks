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

#ifndef DWSHARED_CORE_STORAGEFIXED_HPP_
#define DWSHARED_CORE_STORAGEFIXED_HPP_

#include <cstddef>
#include <type_traits>

#include <dw/core/container/Span.hpp>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/language/cxx14.hpp>

namespace dw
{
namespace core
{

/// StorageFixed: provides fixed size aligned storage for an array of items.
/// It has two versions, one with size determined at runtime (array stored in heap, used when SizeAtCompileTime==0) and
/// one with size known at compile-time (array stored statically in class, used when SizeAtCompileTime!=0).
/// Note that the items returned by this class will not be constructed/destructed by this class.
/// Construction and destruction are the responsibility of the user.
///
/// This default template is the statically sized storage
template <typename T, size_t SizeAtCompileTime = 0>
// TODO(dwplc): FP - it claims that not all 5 special member functions are declared
// coverity[autosar_cpp14_a12_0_1_violation]
class StorageFixed
{
public:
    StorageFixed() = default;

    explicit StorageFixed(size_t size_)
    {
        if (SizeAtCompileTime != size_)
        {
            throw InvalidStateException("StorageFixed: using static storage but requested size is of different size");
        }
    }

    /// Move and copy semantics for statically-sized storage are disabled
    /// because elements need to have constructor/destructors called
    /// and the storage cannot do that.
    StorageFixed(const StorageFixed&) { throw ExceptionWithStackTrace("StorageFixed: cannot copy or move a static storage"); }
    StorageFixed(StorageFixed&&) { throw ExceptionWithStackTrace("StorageFixed: cannot copy or move a static storage"); }

    auto operator   =(const StorageFixed&) -> StorageFixed& { throw ExceptionWithStackTrace("StorageFixed: cannot copy or move a static storage"); }
    auto operator   =(StorageFixed &&) -> StorageFixed& { throw ExceptionWithStackTrace("StorageFixed: cannot copy or move a static storage"); }
    ~StorageFixed() = default;

    /// Provided only to keep a stable API. Throws if the size is different than SizeAtCompileTime.
    void allocate(size_t size_)
    {
        if (size_ != size())
        {
            throw ExceptionWithStackTrace("StorageFixed: cannot allocate different size for static storage");
        }
    }

    auto data() -> T*
    {
        return safeReinterpretCast<T*>(m_data); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }
    auto data() const -> const T*
    {
        return safeReinterpretCast<const T*>(m_data); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    size_t size() const { return SizeAtCompileTime; }

    auto end() -> T* { return &dw::core::span<T>(data(), size())[size()]; }
    auto end() const -> const T* { return data() + size(); }

    auto operator[](size_t const index) -> T& { return dw::core::span<T>(data(), size())[index]; }
    auto operator[](size_t const index) const -> const T& { return dw::core::span<T const>(data(), size())[index]; }

private:
    /// Must make sure storage is aligned (see http://en.cppreference.com/w/cpp/types/aligned_storage)
    using TStorage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    TStorage m_data[SizeAtCompileTime]{};
};

/// Dynamically sized specialization of StorageFixed
template <typename T>
class StorageFixed<T, 0>
{
public:
    explicit StorageFixed(size_t const size)
        : m_size(size)
    {
        if (size > 0U)
        {
            m_data.reset(new TStorage[size]);
        }
    }

    /// Enable move-semantics for dynamically-sized storage
    explicit StorageFixed(const StorageFixed& other) // clang-tidy NOLINT
        : m_size(other.m_size)
    {
    }

    /// Enable move-semantics for dynamically-sized storage
    StorageFixed(StorageFixed&& other) // clang-tidy NOLINT
    {
        *this = std::move(other); // defer to move-assignment operator
    }
    auto operator=(StorageFixed&& other) -> StorageFixed&
    {
        if (&other == this)
        {
            return *this;
        }

        m_data       = std::move(other.m_data);
        m_size       = other.m_size;
        other.m_size = 0U;

        return *this;
    }

    /// Reallocates the storage if the size is different than the current size
    void allocate(size_t const sizeIn)
    {
        /// Reallocate only if we need a bigger buffer
        /// True size of the buffer will be lost if sizeIn < size() but that is ok
        if (sizeIn > size())
        {
            m_data = make_unique<TStorage[]>(sizeIn);
        }
        m_size = sizeIn;
    }

    // TODO(dwplc): FP - can't make 'const', conflicted function overload to the const version of data(),
    //                   compiling error
    // coverity[autosar_cpp14_m9_3_3_violation]
    auto data() -> T*
    {
        return safeReinterpretCast<T*>(m_data.get());
    }

    auto data() const -> const T*
    {
        return safeReinterpretCast<const T*>(m_data.get());
    }

    size_t size() const { return m_size; }

    auto end() -> T* { return &dw::core::span<T>(data(), size())[size()]; }
    auto end() const -> const T* { return &dw::core::span<T const>(data(), size())[size()]; }

    auto operator[](size_t const index) -> T& { return dw::core::span<T>(data(), size())[index]; }
    auto operator[](size_t const index) const -> const T& { return dw::core::span<T const>(data(), size())[index]; }

private:
    /// Must make sure storage is aligned (see http://en.cppreference.com/w/cpp/types/aligned_storage)
    using TStorage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    std::unique_ptr<TStorage[]> m_data;
    size_t m_size;
};

} // namespace core
} // namespace

#endif // DW_CORE_STORAGEFIXED_HPP_
