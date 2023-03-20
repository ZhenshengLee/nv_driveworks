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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_TOADDRESS_HPP_
#define DW_CORE_UTILITY_TOADDRESS_HPP_

#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/language/cxx17.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace dw
{
namespace core
{
namespace detail
{

template <typename T, typename = void_t<>>
struct pointer_has_custom_to_address_traits : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T>
struct pointer_has_custom_to_address_traits<T, void_t<decltype(std::pointer_traits<T>::to_address(std::declval<T const&>()))>>
    : std::true_type
{
};

template <typename T>
constexpr bool pointer_has_custom_to_address_traits_v = pointer_has_custom_to_address_traits<T>::value; // clang-tidy NOLINT(readability-identifier-naming)

} // namespace dw::core::detail

/**
 * \defgroup toaddress_group Pointer conversion Group of Functions
 * @{
 */

/// @{
/// The @c to_address family of functions extract a raw pointer from a fancy pointer (for example: @c unique_ptr,
/// @c shared_ptr, etc), an iterator, or a raw pointer.
///
/// Get the address of the pointer @a p, which is simply @a p.
///
/// @tparam T The type this pointer refers to. If @c T is a function, the program is ill-formed and a @c static_assert
///         will be triggered.
template <typename T>
CUDA_BOTH_INLINE constexpr auto to_address(T* const p) noexcept -> T* // clang-tidy NOLINT(readability-identifier-naming)
{
    static_assert(!std::is_function<T>::value, "Can not get the address of a function");

    return p;
}

/// Get the address of fancy pointer @c TPtr if it has custom @c pointer_traits with a @c to_address function.
template <typename TPtr, std::enable_if_t<detail::pointer_has_custom_to_address_traits_v<TPtr>, bool> = true>
CUDA_BOTH_INLINE constexpr auto to_address(TPtr const& p) noexcept -> decltype(std::pointer_traits<TPtr>::to_address(p)) // clang-tidy NOLINT(readability-identifier-naming)
{
    return std::pointer_traits<TPtr>::to_address(p);
}

/// Get the address of fancy pointer or iterator @c TPtr by using its @c operator->.
///
/// @note
/// Unlike @c std::to_address, this overload only participates in cases where @c TPtr::operator-> directly returns a
/// pointer. While @c std::to_address recursively follows @c operator->, this implementation will not work for cases
/// where the fancy pointer's @c operator-> returns another fancy pointer. This trade-off is required because there is
/// no mechanism in C++ to write the return type or the conditional @c noexcept qualifier required by AUTOSAR with
/// recursion. If this behavior is required, then @c std::pointer_traits should be specialized with a @c to_address
/// implementation.
template <typename TPtr, std::enable_if_t<!detail::pointer_has_custom_to_address_traits_v<TPtr> && std::is_pointer<decltype(std::declval<TPtr const&>().operator->())>::value, bool> = false>
CUDA_BOTH_INLINE constexpr auto to_address(TPtr const& p) noexcept(noexcept(std::declval<TPtr const&>().operator->())) -> decltype(std::declval<TPtr const&>().operator->()) // clang-tidy NOLINT(readability-identifier-naming)
{
    return to_address(p.operator->());
}
/// @}

/**@}*/

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_UTILITY_TOADDRESS_HPP_*/
