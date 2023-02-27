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
// SPDX-FileCopyrightText: Copyright (c) 2015-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_BASE2_HPP_
#define DW_CORE_UTILITY_BASE2_HPP_

#include <dw/core/language/BasicTypes.hpp>
#include <climits>

namespace dw
{
namespace core
{
namespace util
{

/// True if x is an exact power of two
constexpr bool isPowerOfTwo(size_t const x)
{
    return x && ((x & (x - 1)) == 0);
}

/// If v is a power of two, returns v
/// Else it returns the next highest power of two
constexpr uint32_t nextPowerOf2(uint32_t const v)
{
    return v > 1 ? static_cast<uint32_t>(1 << (sizeof(uint32_t) * CHAR_BIT - static_cast<uint32_t>(__builtin_clz(v - 1U)))) : v;
}

/// For a power of two, return it's base-2 logarithm, otherwise return -1
constexpr int32_t ilog2(uint32_t const x)
{
    return isPowerOfTwo(x) ? static_cast<int32_t>(sizeof(uint32_t) * CHAR_BIT -
                                                  static_cast<uint32_t>(__builtin_clz(x)) - 1U)
                           : -1;
}

} // namespace util
} // namespace core
} // namespace dw

#endif
