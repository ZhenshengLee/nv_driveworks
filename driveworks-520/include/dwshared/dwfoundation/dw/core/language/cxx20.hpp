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
// SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_CXX20_HPP_
#define DWSHARED_CORE_CXX20_HPP_

#include <functional>
#include <type_traits>

namespace dw
{
namespace core
{

#if __cplusplus < 202002L

/// @c std::unwrap_reference
template <typename T>
struct unwrap_reference // clang-format NOLINT(readability-identifier-naming)
{
    using type = T;
};

/// @c std::unwrap_reference
template <typename U>
struct unwrap_reference<std::reference_wrapper<U>> // clang-format NOLINT(readability-identifier-naming)
{
    using type = U&;
};

/// @c std::unwrap_reference_t
template <typename T>
using unwrap_reference_t = typename unwrap_reference<T>::type; // clang-format NOLINT(readability-identifier-naming)

/// @c std::unwrap_ref_decay
template <typename T>
struct unwrap_ref_decay : unwrap_reference<std::decay_t<T>> // clang-format NOLINT(readability-identifier-naming)
{
};

/// @c std::unwrap_ref_decay_t
template <typename T>
using unwrap_ref_decay_t = typename unwrap_ref_decay<T>::type; // clang-format NOLINT(readability-identifier-naming)

#else

template <typename T>
using unwrap_reference = std::unwrap_reference<T>; // clang-format NOLINT(readability-identifier-naming)

template <typename T>
using unwrap_reference_t = std::unwrap_reference_t<T>; // clang-format NOLINT(readability-identifier-naming)

template <typename T>
using unwrap_ref_decay = std::unwrap_ref_decay<T>; // clang-format NOLINT(readability-identifier-naming)

template <typename T>
using unwrap_ref_decay_t = std::unwrap_ref_decay_t<T>; // clang-format NOLINT(readability-identifier-naming)

#endif

} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_CORE_CXX20_HPP_*/
