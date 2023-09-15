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

#ifndef DW_CORE_STATICVECTORFIXED_HPP_
#define DW_CORE_STATICVECTORFIXED_HPP_

#include "BaseVectorFixed.hpp"
#include "../../VectorRef.hpp"

namespace dw
{
namespace core
{

/// A vector that contains all of its storage inside of the object itself.
/// Capacity must be specified at compile-time
template <typename T, size_t CAPACITY>
class StaticVectorFixed;

////////////////////////////////////////////////////////////////////////////////

namespace detail
{
/// A vector that contains all of its storage inside of the object itself.
/// Capacity must be specified at compile-time
template <typename T, size_t CAPACITY>
class StaticVectorFixedStorage
{
public:
    StaticVectorFixedStorage()  = default; // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
    ~StaticVectorFixedStorage() = default;

    /// Move constructor
    StaticVectorFixedStorage(StaticVectorFixedStorage&& other)
        : m_size(std::move(other.m_size))
    {
        if (other.m_size != 0)
        {
            throw InvalidStateException("Cannot move/copy static storage. Methods are a noop and can only be called with empty vectors.");
        }
    }

    StaticVectorFixedStorage(const StaticVectorFixedStorage& other) = delete;
    auto operator=(const StaticVectorFixedStorage& other) -> StaticVectorFixedStorage& = delete;
    auto operator=(StaticVectorFixedStorage&& other) -> StaticVectorFixedStorage& = delete;

    /// Get storage capacity
    static constexpr size_t capacity() { return CAPACITY; }

    /// Get pointer to data
    auto data() -> T*
    {
        return safeReinterpretCast<T*>(m_data); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Get const pointer to data
    auto data() const -> const T*
    {
        return safeReinterpretCast<const T*>(m_data); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Size
    size_t size() const { return m_size; }

    /// Size as reference
    size_t& size() { return m_size; }

private:
    using TBuffer            = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
    TBuffer m_data[CAPACITY] = {};

    size_t m_size = 0U;
};
}

// See top of file for comments
template <typename T, size_t CAPACITY>
class StaticVectorFixed : public BaseVectorFixed<T, detail::StaticVectorFixedStorage<T, CAPACITY>>
{
public:
    using TStorage = detail::StaticVectorFixedStorage<T, CAPACITY>; ///< Storage type
    using Base     = BaseVectorFixed<T, TStorage>;                  ///< Base class type

    StaticVectorFixed()
        : Base()
    {
    }

    ~StaticVectorFixed()
    {
        try
        {
            Base::clear();
        }
        catch (const dw::core::ExceptionBase& e)
        {
            LOGSTREAM_ERROR(nullptr) << "~StaticVectorFixed: Exception caught in StaticVectorFixed() destructor: " << e.what() << dw::core::Logger::State::endl;
        }
    }

    /// This constructor is a dummy to match the API of HeapVectorFixedFixed
    explicit StaticVectorFixed(size_t const argCapacity)
        : Base()
    {
        if (argCapacity != CAPACITY)
        {
            // TODO(dwplc): FP - "Using basic numerical type "char" rather than a typedef that includes size and signedness information."
            // coverity[autosar_cpp14_a3_9_1_violation]
            throw InvalidArgumentException("StaticVectorFixed: the capacity requested at run-time (",
                                           argCapacity,
                                           ") does not match the capacity requested at compile-time (",
                                           strip_constexpr(CAPACITY), ").");
        }
    }

    /// Initializes the vector with size_ copies of T(value)
    template <class TValue = T>
    StaticVectorFixed(size_t const argSize, TValue const value)
        : Base()
    {
        Base::push_back_range(argSize, value);
    }

    /// create from initializer list
    /// This is not a constructor because it gets confused with Vector(size_t, TValue)
    static auto fromList(std::initializer_list<T> const val) -> StaticVectorFixed
    {
        StaticVectorFixed result{};
        result.push_back_range(val);
        return result;
    }

    /// Copy constructor
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticVectorFixed(const BaseVectorFixed<T, D>& other)
        : Base(other)
    {
    }

    /// Copy constructor
    /// All items from the other vector will be copied into this
    StaticVectorFixed(const StaticVectorFixed& other)
        : Base(other)
    {
    }

    /// Move constructor
    /// All items from the other vector will be copied into this
    template <class D>
    explicit StaticVectorFixed(BaseVectorFixed<T, D>&& other)
        : Base(std::move(other))
    {
    }

    /// Move constructor
    /// Storage is static so it cannot be moved,
    /// instead each element is moved into this one
    /// and other is cleared
    StaticVectorFixed(StaticVectorFixed&& other)
        : Base(std::move(other))
    {
    }

    /// Initialize from a span by copying each element
    explicit StaticVectorFixed(const span<const T> dataToCopy)
        : Base()
    {
        this->push_back_range(dataToCopy);
    }

    /// Use generic move/copy operator
    using Base::operator=;

    /// Copy
    auto operator=(const StaticVectorFixed& other) -> StaticVectorFixed&
    {
        Base::copyFrom(other);
        return *this;
    }

    /// Move
    auto operator=(StaticVectorFixed&& other) -> StaticVectorFixed&
    {
        Base::moveFrom(other);
        return *this;
    }

    /// Get VectorRef that refers to this container. See VectorRef for more details.
    auto toVectorRef() -> VectorRef<T>
    {
        return VectorRef<T>{make_span(Base::getStorage().data(), CAPACITY), Base::getStorage().size()};
    }
};

} // namespace core
} // namespace dw

#endif
