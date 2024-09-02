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

#ifndef DW_CORE_ARRAY_HPP_
#define DW_CORE_ARRAY_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx14.hpp>
#include <dwshared/dwfoundation/dw/core/ConfigChecks.h>
#include <dwshared/dwfoundation/dw/core/language/Math.hpp>

#include <cstddef>

namespace dw
{
namespace core
{

// non-inlined helper functions for bounds-error reporting

/// throw OutOfBoundsException
[[noreturn]] void throwArrayIndexOutOfBounds();

/// throw OutOfBoundsException
[[noreturn]] void throwMatrixIndexOutOfBounds();

/// throw OutOfBoundsException
[[noreturn]] void throwSpanIndexOutOfBounds();
[[noreturn]] void throwVectorIndexOutOfBounds();

/// Type replacing std::array because std::array isn't supported in CUDA.
/// Once CUDA supports std::array, this could be removed.
/// Provides a reduced interface of std::array, and represents an aggregate type.
///
/// This type is similar to dw::math::Vector in that a static and fixed sized number of generic elements can
/// be allocated. It differs in semantics based on its aggregate initialization, and can be used in
/// 'constexpr' contexts, which is not supported by 'dw::Vector'. It should be most commonly used for any
/// 'T[N]' array instance that only requires pure-member access (but no, e.g., linear algebra operations),
/// like size indications of multidimenstional datastructures ('dw::core::span' etc.)
///
/// Also, this is a basic core type that does not provide / support nor require any math-related
/// functionality, so that separation of non-math-related functionality using this type is possible (e.g.,
/// separation of 'dwfoundation' and (upcoming) 'dwmath').
///
/// Explicitly as a pure public class to allow for aggregate type. Note: using struct is a autosar a11_0_2 violation
template <class T, size_t N>
class Array // clang-tidy NOLINT(readability-braces-around-statements)
{
public:
    using value_type = T; ///< Type of elements in Array

    // TODO(dwplc): RFD - std::array cannot yet be used with CUDA, so we use a C-style array here.
    // coverity[autosar_cpp14_a18_1_1_violation]
    T _elems[N]; ///< non-private member to enable full aggregate initialization

    // No explicit construct/copy/destroy for aggregate type

    /// Array size
    /// Array is a fixed size datastructure, so this is the same as max_size()
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    static constexpr size_t size() noexcept { return N; }

    /// Array max size
    /// Array is a fixed size datastructure, so this is the same as size()
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr size_t max_size() const noexcept { return N; }

    /// Returns true if size() is 0
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr bool empty() const noexcept { return size() == 0; }

    /// Indexed element access
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr auto operator[](size_t const i) -> T&
    {
        // TODO(dwplc)(AVDWPLC-715): There should be bounds checking here, however that triggers a QNX compiler bug,
        // and trying to work around the compiler bug causes other issues. Add bounds checking here whenever QNX moves
        // to a newer version of nvcc.
        // coverity[cert_ctr50_cpp_violation]
        return _elems[i];
    }

    /// Indexed const element access
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr auto operator[](size_t const i) const -> const T&
    {
        // TODO(dwplc)(AVDWPLC-715): There should be bounds checking here, however that triggers a QNX compiler bug,
        // and trying to work around the compiler bug causes other issues. Add bounds checking here whenever QNX moves
        // to a newer version of nvcc.
        // coverity[cert_ctr50_cpp_violation]
        return _elems[i];
    }

    /// Indexed element access with additional out of bounds check.
    /// For Non-CUDA code it throws an out of bounds exception if index is out of range.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto at(size_t const i) -> T&
    {
        if (i >= N)
        {
#if !defined(__CUDACC__)
            throwArrayIndexOutOfBounds();
#else
            core::assertException<OutOfBoundsException>("Array access index out of range");
#endif
        }

        return _elems[i];
    }

    /// Indexed const element access with additional out of bounds check.
    /// For Non-CUDA code it throws an out of bounds exception if index is out of range.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr auto at(size_t const i) const -> T const&
    {
// TODO(janickm): make exception more explicit once C++14 is available

// C++11 constexpr requires single return statement, and result of conditional expression must be an
// lvalue so use boolean ? lvalue : (throw-expr, lvalue)
// TODO(dwplc): Remove this when we can use C++14 constexpr functionality
#if !defined(__CUDACC__)
        // coverity[autosar_cpp14_m5_18_1_violation]
        return i < N ? _elems[i] : (throwArrayIndexOutOfBounds(), _elems[0]);
#else
        // coverity[autosar_cpp14_m5_18_1_violation]
        return i < N ? _elems[i] : (core::assertException<OutOfBoundsException>("Array access index out of range"), _elems[0]);
#endif
    }

    /// Fill array with given value;
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE DW_CONSTEXPR_CXX14 void fill(const T& value)
    {
        for (std::size_t idx{0U}; idx < N; ++idx)
        {
            _elems[idx] = value;
        }
    }

    /// Access first element.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto front() noexcept -> T& { return _elems[0]; }

    /// Access first element.

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr auto front() const noexcept -> T const& { return _elems[0]; }

    /// Access last element.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto back() noexcept -> T& { return _elems[N - 1]; }

    /// Access last element.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    constexpr auto back() const noexcept -> T const& { return _elems[N - 1]; }

    /// Get pointer to beginning of the Array data.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto data() noexcept -> T* { return _elems; }

    /// Get const pointer to beginning of the Array data.
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto data() const noexcept -> T const* { return _elems; }

    // Iterators

    /// Iterator to first element
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto begin() noexcept -> T* { return data(); }

    /// Iterator to first element
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto begin() const noexcept -> T const* { return data(); }

    /// End iterator
    // TODO(dwplc): Replace with safe iterator when available
    // coverity[autosar_cpp14_m5_0_15_violation]
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto end() noexcept -> T* { return data() + N; }

    /// End iterator
    // TODO(dwplc): Replace with safe iterator when available
    // coverity[autosar_cpp14_m5_0_15_violation]
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE
    auto end() const noexcept -> T const* { return data() + N; }

    // Non-standard methods

    /// Create new Array containing the first M elements.
    template <size_t M>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto getFirst() const noexcept -> Array<T, M>
    {
        static_assert(M <= N, "Invalid range requested");
        Array<T, M> out{};
        for (size_t i{0U}; i < M; ++i)
        {
            out[i] = _elems[i];
        }
        return out;
    }
};

/// Specialization for dummy zero-sized array
template <class T>
class Array<T, 0U>
{
public:
    using value_type          = T; ///< Type of elements in Array
    static constexpr size_t N = 0; ///< Size 0

    // clang-format off

// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE constexpr size_t size() const noexcept     { return 0U; }   ///< 0 size()
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE constexpr size_t max_size() const noexcept { return 0U; }   ///< 0 size()
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE constexpr bool empty() const noexcept      { return true; } ///< always empty

    // dereferencing a zero-sized array for an element is undefined behaviour
    // TODO(dwplc): FP - Coverity wants this operator to be marked as static because it doesn't reference any members,
    //                   but operators cannot be static.
    // coverity[autosar_cpp14_m9_3_3_violation]
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto       operator[](size_t) noexcept -> T&      { return *data(); } ///< Invalid data access on empty Array
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator[](size_t) const noexcept -> const T&{ return *data(); } ///< Invalid data access on empty Array

    /// Invalid data access on empty Array
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto at(size_t) -> T&
    {
        core::assertException<OutOfBoundsException>("Array of zero size used");
        return *data();
    }

// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto front() noexcept    -> T&   { return *data(); } ///< Invalid data access on empty Array
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto back() noexcept   -> T&     { return *data(); } ///< Invalid data access on empty Array
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto data() noexcept   -> T*     { return nullptr; } ///< Always returns nullptr
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto begin() noexcept   -> T*    { return nullptr; } ///< Always returns nullptr
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto end() noexcept  -> T*       { return nullptr; } ///< Always returns nullptr

    // TODO(dwplc): RFD - Coverity complains because this fill implementation doesn't have any side effects,
    //                    but that is the desired behavior for the zero sized array specialization.
    // coverity[autosar_cpp14_m0_1_8_violation]
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE DW_CONSTEXPR_CXX14 static void fill(const T&) { } ///< Does nothing on empty Array
    // clang-format on

    /// Empty array cannot return a sub-Array.
    template <size_t M>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto getFirst() const noexcept -> Array<T, M>
    {
        static_assert(M <= N, "Invalid range requested");
        return *this;
    }
};

/// Same as Array, but it gets allocated with requested aligment
template <class T, size_t N, size_t ALIGNMENT>
struct alignas(ALIGNMENT) AlignedArray : public Array<T, N>
{
};

// Array comparisons
template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return core::equal(first.begin(), first.end(), second.begin());
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return !(first == second);
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return core::lexicographicalCompare(first.begin(), first.end(), second.begin(), second.end());
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return second < first;
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<=(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return !(first > second);
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>=(Array<T, N> const& first, Array<T, N> const& second) noexcept
{
    return !(first < second);
}

// This type instances are already explicitly instantiated to verify via static analysis
// (explicitly use an instances unlikely used in practice)
extern template class Array<void*, 0>;
extern template class Array<void*, 1>;
extern template class Array<void*, 2>;

// Commonly used types
/// 2 dimentional array with each dimension in uint32_t type
using Array2ui = Array<uint32_t, 2>;

namespace detail
{

/// Helper template for generic statically sized integer_sequence type
template <typename T, std::size_t Start, std::size_t End, T Value, typename Enable = void, T... Sequence>
struct fill_sequence; // clang-tidy NOLINT(readability-identifier-naming)

/// Helper template for generic statically sized integer_sequence type
template <typename T, std::size_t End, T Value, T... Sequence>
struct fill_sequence<T, End, End, Value, void, Sequence...>
{
    /// integer_sequence type
    using type = integer_sequence<T, Sequence...>;
};

/// Helper template for generic statically sized integer_sequence type
template <typename T, std::size_t Current, std::size_t End, T Value, T... Sequence>
struct fill_sequence<T, Current, End, Value, typename std::enable_if<(Current < End)>::type, Sequence...>
{
    /// fill_sequence type
    using type = typename fill_sequence<T, Current + 1, End, Value, void, Sequence..., Value>::type;
};

template <typename T, T... Value>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr auto makeFilledArrayImpl(integer_sequence<T, Value...>) -> Array<T, sizeof...(Value)>
{
    return Array<T, sizeof...(Value)>{Value...};
}

/**
 * @brief Fill a sequence with N value=Value data
 * e.g. fill_integer_sequence<uint32_t, 5, 11>::type = integer_sequence<uint32_t, 11, 11, 11, 11, 11>
 *      fill_integer_sequence<uint32_t, 3, 0>::type = integer_sequence<uint32_t, 0, 0, 0>
 *
 * @tparam T  integer data type
 * @tparam N  element number
 * @tparam Value filling value
 */
template <typename T, std::size_t N, T Value>
using fill_integer_sequence = typename fill_sequence<T, std::size_t(0), N, Value>::type;
}

/// Create an @ref Array filled with the given @a value.
template <typename T, std::size_t N, T Value>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr auto makeFilledArray() -> Array<T, N>
{
    return detail::makeFilledArrayImpl<T>(detail::fill_integer_sequence<T, N, Value>());
}

} // namespace core
} // namespace dw

#endif // DW_CORE_ARRAY_HPP_
