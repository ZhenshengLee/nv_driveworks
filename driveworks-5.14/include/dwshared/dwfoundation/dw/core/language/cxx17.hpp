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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_CXX17_HPP_
#define DWSHARED_CORE_CXX17_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

namespace dw
{
namespace core
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::in_place_______ Utilities                                                                                     //
// https://en.cppreference.com/w/cpp/utility/in_place                                                                 //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if __cplusplus < 201703L

/// See @c std::in_place_t.
struct in_place_t // clang-format NOLINT(readability-identifier-naming)
{
    explicit in_place_t() = default;
};

/// See @c std::in_place.
// coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3959720
constexpr in_place_t in_place{}; // clang-format NOLINT(readability-identifier-naming)

/// See @c std::in_place_type_t.
template <typename T>
struct in_place_type_t // clang-format NOLINT(readability-identifier-naming)
{
    explicit in_place_type_t() = default;
};

/// See @c std::in_place_type.
template <typename T>
in_place_type_t<T> in_place_type{}; // clang-format NOLINT(readability-identifier-naming)

/// See @c std::in_place_index_t
template <std::size_t Idx>
struct in_place_index_t // clang-format NOLINT(readability-identifier-naming)
{
    explicit in_place_index_t() = default;
};

/// See @c std::in_place_index.
template <std::size_t Idx>
in_place_index_t<Idx> in_place_index; // clang-format NOLINT(readability-identifier-naming)

/// See @c std::void_t.
template <typename...>
using void_t = void; // clang-format NOLINT(readability-identifier-naming)

#else

using in_place_t = std::in_place_t; // clang-format NOLINT(readability-identifier-naming)

inline constexpr in_place_t in_place = std::in_place; // clang-format NOLINT(readability-identifier-naming)

template <typename T>
using in_place_type_t = std::in_place_type_t<T>; // clang-format NOLINT(readability-identifier-naming)

template <typename T>
inline constexpr in_place_type_t<T> in_place_type{}; // clang-format NOLINT(readability-identifier-naming)

template <std::size_t Idx>
using in_place_index_t = std::in_place_index_t<Idx>; // clang-format NOLINT(readability-identifier-naming)

template <std::size_t Idx>
inline constexpr in_place_index_t<Idx> in_place_index{}; // clang-format NOLINT(readability-identifier-naming)

using std::void_t;

#endif

} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_CORE_CXX17_HPP_*/
