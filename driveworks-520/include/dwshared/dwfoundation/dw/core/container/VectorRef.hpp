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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_VECTORREF_HPP_
#define DW_CORE_VECTORREF_HPP_

#include "impl/VectorFixed/BaseVectorFixed.hpp"

namespace dw
{
namespace core
{

/// Behaves like a normal vector but doesn't own the data nor size
/// It can be a reference to a StaticVectorFixed or HeapVectorFixed
/// so it can be used to pass vectors without worrying if they are
/// stack or heap allocated. For example:
///     void foo(VectorRef<float32_t> v)
///     {
///         v.push_back(123.f);
///     }
///
///     StaticVectorFixed<float32_t, 1000> s;
///     foo(s.toVectorRef());
///
///     HeapVectorFixed<float32_t> h(1000);
///     foo(h.toVectorRef());
///
/// It can also be a reference to a span and size variable.
///
///     span<float32_t> sp = make_span(somePointer, someCapacity);
///     size_t size = someCurrentSize;
///     foo(VecotrRef{sp, size});
///
template <typename T>
class VectorRef;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{
/// Vector storage that doesn't own size nor data.
/// Both are references and are owned by someone else.
// See top of file for comments
template <typename T>
class VectorRefStorage
{
public:
    using TBuffer = typename std::aligned_storage<sizeof(T), alignof(T)>::type; ///< Type of storage for a data element

    /// Empty default construction
    VectorRefStorage() // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : VectorRefStorage(nullptr)
    {
    }

    /// Empty construction
    explicit VectorRefStorage(std::nullptr_t) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : VectorRefStorage(span<TBuffer>{nullptr}, nullptr)
    {
    }

    /// Construction from data span and reference to a size variable
    VectorRefStorage(span<TBuffer> data_, size_t& size_)
        : m_data(std::move(data_))
        , m_size(&size_)
    {
        if (*m_size > m_data.size())
        {
            throw InvalidArgumentException("VectorRefStorage: the given size (", *m_size, ") is larger than the capacity (", m_data.size(), ")");
        }
    }

    /// Construction from data span and reference to a size variable
    VectorRefStorage(span<T> const dataI, size_t& size_)
        : m_data(safeReinterpretCast<TBuffer*>(dataI.data()), dataI.size()) // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        , m_size(&size_)
    {
        if (*m_size > m_data.size())
        {
            // TODO(dwplc): FP - "Using basic numerical type "char" rather than a typedef that includes size and signedness information."
            // coverity[autosar_cpp14_a3_9_1_violation]
            throw InvalidArgumentException("VectorRefStorage: the given size (", *m_size, ") is larger than the capacity (", m_data.size(), ")");
        }
    }

    VectorRefStorage(const VectorRefStorage<T>& otherStorage) = default;               ///< Copy constructor
    VectorRefStorage& operator=(const VectorRefStorage<T>& otherStorage) = default;    ///< Copy operator
    VectorRefStorage(VectorRefStorage<T>&& otherStorage)                 = default;    ///< Move constructor
    auto operator=(VectorRefStorage<T>&& otherStorage) -> VectorRefStorage& = delete;  ///< No move operator
    ~VectorRefStorage()                                                     = default; ///< Destructor

    /// Get storage capacity
    size_t capacity() const { return m_data.size(); }

    /// Get data pointer
    auto data() -> T*
    {
        return safeReinterpretCast<T*>(m_data.data()); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Get const data pointer
    auto data() const -> const T*
    {
        return safeReinterpretCast<const T*>(m_data.data()); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Get reference to size variable
    size_t& size() const { return *m_size; }

private:
    span<TBuffer> m_data;

    size_t* m_size;

private:
    VectorRefStorage(span<TBuffer> data_, size_t* const sizeI)
        : m_data(std::move(data_))
        , m_size(sizeI)
    {
        if (m_size == nullptr)
        {
            return;
        }

        if (*m_size > m_data.size())
        {
            throw InvalidArgumentException("VectorRefStorage: the given size (", *m_size, ") is larger than the capacity (", m_data.size(), ")");
        }
    }
};
} // namespace detail

// See top of file for comments
template <typename T>
class VectorRef : public BaseVectorFixed<T, detail::VectorRefStorage<T>>
{
public:
    using TStorage = detail::VectorRefStorage<T>;  ///< Type of storage
    using Base     = BaseVectorFixed<T, TStorage>; ///< Type of base class
    using TBuffer  = typename TStorage::TBuffer;   ///< Type of storage for a data element

    /// Empty default construction
    VectorRef()
        : Base(TStorage{nullptr}, nullptr)
    {
    }

    /// Construction from data span and reference to a size variable
    VectorRef(span<TBuffer> data_, size_t& size_)
        : Base(TStorage{data_, size_}, nullptr)
    {
    }

    /// Construction from data span and reference to a size variable
    template <class T2>
    VectorRef(span<T2> data_, size_t& size_)
        : Base(TStorage{std::move(data_), size_}, nullptr)
    {
        static_assert(std::is_same<T, T2>::value, "Template is just for SFINAE, type must match");
        static_assert(std::is_trivially_destructible<T>::value, "The vector assumes that all items after data[size-1] "
                                                                "are garbage and not-constructed. A span<T> would probably have "
                                                                "all items initialized if created from an array or a UniqueSpan. "
                                                                "Thus, passing a span of a non-trivially destructible type might overwrite "
                                                                "some of those items without calling their destructors.");
    }

    /// Copy reference.
    /// Both vectors will now point to the same memory area
    VectorRef(VectorRef&& other)
        // TODO(dwplc): FP - Already use move constructor
        // coverity[autosar_cpp14_a12_8_4_violation]
        : Base(std::move(other.getStorage()), std::move(other.getCapacityHint()))
    {
    }

    /// Copy constructor
    VectorRef(const VectorRef& other)
        : Base(TStorage(other.getStorage()), other.getCapacityHint())
    {
    }

    /**
     *  @brief      Construct from a type with a public auto toVectorRef() -> VectorRef<T> method
     *  @tparam     VectorFixed Non-reference type that implements the toVectorRef method.
     *  @param[in]  vectorFixed Object of type VectorFixed to be constructed from
     */
    template <class VectorFixed,
              std::enable_if_t<!std::is_same<VectorRef, std::decay_t<VectorFixed>>::value>* = nullptr>
    VectorRef(VectorFixed& vectorFixed) // clang-tidy NOLINT(google-explicit-constructor)
        : VectorRef(vectorFixed.toVectorRef())
    {
        static_assert(!std::is_reference<VectorFixed>::value, "Type cannot be a reference type.");
    }

    /// Destructor
    ~VectorRef() = default;

    /// Copy operator
    auto operator=(const VectorRef& other) -> VectorRef&
    {
        Base::setStorage(other.getStorage());
        return *this;
    }

    /// Move operator
    auto operator=(VectorRef&& other) -> VectorRef&
    {
        Base::setStorage(std::move(other.getStorage()));
        return *this;
    }

    /// Return true if no Vector is referenced
    bool isNull() const
    {
        return Base::getStorage().data() == nullptr;
    }
};

} // namespace core
} // namespace dw

#endif
