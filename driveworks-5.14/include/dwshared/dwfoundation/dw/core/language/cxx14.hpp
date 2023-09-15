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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_CXX14_HPP_
#define DWSHARED_CORE_CXX14_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>

#include <memory>
#include <type_traits>

/// @def DW_CONSTEXPR_CXX14
/// Mark a function as @c constexpr if compiling with C++14 or higher, but do not mark it as such when compiling without
/// C++14 support. This is useful for functions which can be written as @c constexpr with C++14 rules, but not with
/// C++11.
///
/// Once all compilers fully support C++14 extended definition of @c constexpr, all sites can be replaced with
/// @c constexpr and this macro can be removed.
#if __cplusplus < 201402L
#define DW_CONSTEXPR_CXX14
#else
#define DW_CONSTEXPR_CXX14 constexpr
#endif

namespace dw
{
namespace core
{

namespace detail
{

/// @{
/// This overload set is used to determine which @c make_unique overloads are enabled based on the type. By giving each
/// member object a different name, the overloads should be fairly easy to distinguish.
template <typename T>
struct CheckMakeUniqueFor
{
    using SingleObject = std::unique_ptr<T>;
};

template <typename T>
struct CheckMakeUniqueFor<T[]>
{
    using UnsizedArray = std::unique_ptr<T[]>;
};

template <typename T, std::size_t N>
struct CheckMakeUniqueFor<T[N]>
{
    struct SizedArray
    {
    };
};
/// @}
}

/// Create a single object using the provided @a args.
///
/// @returns A @c unique_ptr to the created object.
template <typename T, typename... TArgs>
auto make_unique(TArgs&&... args) -> typename detail::CheckMakeUniqueFor<T>::SingleObject
{
    return std::unique_ptr<T>(new T(std::forward<TArgs>(args)...));
}

/// Create a unique pointer to an unsized array with default-constructed objects.
///
/// @param sz The size of the array to construct.
///
/// @returns A @c unique_ptr to the created array.
// coverity[autosar_cpp14_a13_3_1_violation] -- FP: this overload is never simultaneously enabled with the TArgs&& one
template <typename TArray>
auto make_unique(std::size_t const sz) -> typename detail::CheckMakeUniqueFor<TArray>::UnsizedArray
{
    using ElementType = typename std::remove_extent<TArray>::type;
    return std::unique_ptr<TArray>(new ElementType[sz]);
}

/// The @c make_unique overload cannot be used with sized arrays.
template <typename TArray, typename... TArgs>
auto make_unique(TArgs&&...) -> typename detail::CheckMakeUniqueFor<TArray>::SizedArray = delete;

/// @c std::integer_sequence
template <typename T, T... Ints>
class integer_sequence // clang-tidy NOLINT(readability-identifier-naming)
{
    using value_type = T;

    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

/// @c std::index_sequence
template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>; // clang-tidy NOLINT(readability-identifier-naming)

namespace detail
{

template <typename T, T Current, T End, typename Enable = void, T... Sequence>
struct build_integer_sequence; // clang-tidy NOLINT(readability-identifier-naming)

template <typename T, T End, T... Sequence>
struct build_integer_sequence<T, End, End, void, Sequence...>
{
    using type = integer_sequence<T, Sequence...>;
};

template <typename T, T Current, T End, T... Sequence>
struct build_integer_sequence<T, Current, End, typename std::enable_if<(Current < End)>::type, Sequence...>
{
    using type = typename build_integer_sequence<T, Current + 1, End, void, Sequence..., Current>::type;
};
}

/// @c std::make_integer_sequence
template <typename T, T N>
using make_integer_sequence = typename detail::build_integer_sequence<T, T(0), N>::type; // clang-tidy NOLINT(readability-identifier-naming)

/// @c std::make_index_sequence
template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>; // clang-tidy NOLINT(readability-identifier-naming)

/// Unit type intended for use as a well-behaved empty alternative to 'void' (replicates C++17's 'std::monostate')
struct Monostate
{
};

/// Monostates are always *equal*
CUDA_BOTH_INLINE constexpr bool operator==(Monostate, Monostate) noexcept
{
    return true;
}

CUDA_BOTH_INLINE constexpr bool operator!=(Monostate, Monostate) noexcept
{
    return false;
}

CUDA_BOTH_INLINE constexpr bool operator<(Monostate, Monostate) noexcept
{
    return false;
}

CUDA_BOTH_INLINE constexpr bool operator>(Monostate, Monostate) noexcept
{
    return false;
}

CUDA_BOTH_INLINE constexpr bool operator<=(Monostate, Monostate) noexcept
{
    return true;
}

CUDA_BOTH_INLINE constexpr bool operator>=(Monostate, Monostate) noexcept
{
    return true;
}

/// Trait to determine if a given type is a specialization of a template class
template <typename Test, template <typename...> class Ref>
struct IsSpecialization : std::false_type
{
};

template <template <typename...> class Ref, typename... Args>
struct IsSpecialization<Ref<Args...>, Ref> : std::true_type
{
};

/// Convert constexpr to value
template <typename T>
auto strip_constexpr(T const v) -> T
{
    return v;
}

} // namespace dw::core
} // namespace dw

#endif
