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

#ifndef DWSHARED_CORE_TUPLE_HPP_
#define DWSHARED_CORE_TUPLE_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include "cxx20.hpp"

#include <type_traits>
#include <utility>

namespace dw
{
namespace core
{

template <typename... T>
struct Tuple
{
};

/**
 * \defgroup tuplestruct_group Tuple Struct Group
 * @{
 */

/// An alternative std::tuple implementation since the implementation on QNX
/// uses extensive template recursion which quickly reaches the default compiler
/// limits.
template <typename THead, typename... TTail>
struct Tuple<THead, TTail...>
{
    /// Tuple constructor
    CUDA_BOTH_INLINE constexpr explicit Tuple(const THead& head, const TTail&... tail)
        : m_head(head)
        , m_tail(tail...)
    {
    }
    /// The tuple head
    THead m_head;
    /// The tuple tail
    Tuple<TTail...> m_tail;
};

/**@}*/

/// @c std::tuple_size
template <class T>
struct tuple_size;

/// @c std::tuple_size
template <class... Types>
struct tuple_size<Tuple<Types...>>
    : std::integral_constant<std::size_t, sizeof...(Types)>
{
};

/// @c std::tuple_element
template <std::size_t I, class T>
struct tuple_element;

/// @c std::tuple_element
template <std::size_t Index, typename THead, typename... TTail>
struct tuple_element<Index, Tuple<THead, TTail...>>
    : tuple_element<Index - 1, Tuple<TTail...>>
{
};

/// @c std::tuple_element
template <typename THead, typename... TTail>
struct tuple_element<0, Tuple<THead, TTail...>>
{
    using type = THead;
};

/// @c std::tuple_element
template <std::size_t Index>
struct tuple_element<Index, Tuple<>>
{
    static_assert(Index < tuple_size<Tuple<>>::value, "Tuple index is out-of-range");
};

/// @c std::tuple_element_t
template <std::size_t Index, class T>
using tuple_element_t = typename tuple_element<Index, T>::type;

/**
 * \defgroup tuple_group Tuple Group of Functions
 * @{
 */

/// Create a tuple
template <typename... Args>
CUDA_BOTH_INLINE constexpr auto make_tuple(Args&&... args)
{
    return Tuple<dw::core::unwrap_ref_decay_t<Args>...>(std::forward<Args>(args)...);
}

/// Return a const tuple, when index is zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index == 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr const auto& get(const Tuple<THead, TTail...>& t)
{
    return t.m_head;
}

/// Return a const tuple, when index is not zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index != 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr const auto& get(const Tuple<THead, TTail...>& t)
{
    return get<Index - 1>(t.m_tail);
}

/// Return a non-const tuple, when index is zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index == 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr auto& get(Tuple<THead, TTail...>& t)
{
    return t.m_head;
}

/// Return a non-const tuple, when index is not zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index != 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr auto& get(Tuple<THead, TTail...>& t)
{
    return get<Index - 1>(t.m_tail);
}

/// Return a non-const tuple by reference, when index is zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index == 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr auto&& get(Tuple<THead, TTail...>&& t)
{
    return std::move(t.m_head);
}

/// Return a non-const tuple by reference, when index is not zero, same behavior of std::get(std::tuple)
template <
    std::size_t Index,
    typename THead, typename... TTail,
    typename std::enable_if_t<Index != 0, void>* = nullptr>
CUDA_BOTH_INLINE constexpr auto&& get(Tuple<THead, TTail...>&& t)
{
    // Note that this only allows moving from the requested value
    // and other values are not mutated
    return get<Index - 1>(std::move(t.m_tail));
}

/**@}*/

} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_CORE_TUPLE_HPP_*/
