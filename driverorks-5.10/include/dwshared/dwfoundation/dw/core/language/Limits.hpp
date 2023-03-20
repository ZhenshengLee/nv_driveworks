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

#ifndef DW_CORE_LANGUAGE_LIMITS_HPP_
#define DW_CORE_LANGUAGE_LIMITS_HPP_

#include "BasicTypes.hpp"

#include <limits>
#include <climits>
#include <type_traits>
#include <cfloat>

namespace dw
{
namespace core
{

///Numeric limits

// TODO(dwplc): RFD - Required for template specialization below. This generic version is intentionally undefined, non-specialized instances shall result in a "undefined" compile error.
// coverity[autosar_cpp14_m3_2_3_violation]
template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float32_t>
{
    CUDA_BOTH_INLINE static constexpr float32_t min()
    {
        return FLT_MIN;
    }
    CUDA_BOTH_INLINE static constexpr float32_t lowest()
    {
        return -FLT_MAX;
    }
    CUDA_BOTH_INLINE static constexpr float32_t max()
    {
        return FLT_MAX;
    }
    CUDA_BOTH_INLINE static constexpr float32_t epsilon()
    {
        return FLT_EPSILON;
    }
    CUDA_BOTH_INLINE static constexpr float32_t infinity()
    {
        return __builtin_huge_valf();
    }
};
template <>
struct numeric_limits<float64_t>
{
    CUDA_BOTH_INLINE static constexpr float64_t min()
    {
        // The c-style cast (double)(...) is inside a system header file.
        // coverity[autosar_cpp14_a5_2_2_violation]
        return DBL_MIN;
    }
    CUDA_BOTH_INLINE static constexpr float64_t lowest()
    {
        // The c-style cast (double)(...) is inside a system header file.
        // coverity[autosar_cpp14_a5_2_2_violation]
        return -DBL_MAX;
    }
    CUDA_BOTH_INLINE static constexpr float64_t max()
    {
        // The c-style cast (double)(...) is inside a system header file.
        // coverity[autosar_cpp14_a5_2_2_violation]
        return DBL_MAX;
    }
    CUDA_BOTH_INLINE static constexpr float64_t epsilon()
    {
        // The c-style cast (double)(...) is inside a system header file.
        // coverity[autosar_cpp14_a5_2_2_violation]
        return DBL_EPSILON;
    }
    CUDA_BOTH_INLINE static constexpr float64_t infinity()
    {
        return __builtin_huge_val();
    }
};

template <>
struct numeric_limits<int8_t>
{
    CUDA_BOTH_INLINE static constexpr int8_t min()
    {
        return INT8_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int8_t lowest()
    {
        return INT8_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int8_t max()
    {
        return INT8_MAX;
    }
    CUDA_BOTH_INLINE static constexpr int8_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<uint8_t>
{
    CUDA_BOTH_INLINE static constexpr uint8_t min()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint8_t lowest()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint8_t max()
    {
        // Macro declared inside a system header file, missing "U" suffix.
        // coverity[autosar_cpp14_m2_13_3_violation]
        return UINT8_MAX;
    }
    CUDA_BOTH_INLINE static constexpr uint8_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<int16_t>
{
    CUDA_BOTH_INLINE static constexpr int16_t min()
    {
        return INT16_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int16_t lowest()
    {
        return INT16_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int16_t max()
    {
        return INT16_MAX;
    }
    CUDA_BOTH_INLINE static constexpr int16_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<uint16_t>
{
    CUDA_BOTH_INLINE static constexpr uint16_t min()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint16_t lowest()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint16_t max()
    {
        // Macro declared inside a system header file, missing "U" suffix.
        // coverity[autosar_cpp14_m2_13_3_violation]
        return UINT16_MAX;
    }
    CUDA_BOTH_INLINE static constexpr uint16_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<int32_t>
{
    CUDA_BOTH_INLINE static constexpr int32_t min()
    {
        return INT_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int32_t lowest()
    {
        return INT_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int32_t max()
    {
        return INT_MAX;
    }
    CUDA_BOTH_INLINE static constexpr int32_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<uint32_t>
{
    CUDA_BOTH_INLINE static constexpr uint32_t min()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint32_t lowest()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint32_t max()
    {
        return UINT_MAX;
    }
    CUDA_BOTH_INLINE static constexpr uint32_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<int64_t>
{
    CUDA_BOTH_INLINE static constexpr int64_t min()
    {
        return LONG_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int64_t lowest()
    {
        return LONG_MIN;
    }
    CUDA_BOTH_INLINE static constexpr int64_t max()
    {
        return LONG_MAX;
    }
    CUDA_BOTH_INLINE static constexpr int64_t epsilon()
    {
        return 0;
    }
};

template <>
struct numeric_limits<uint64_t>
{
    CUDA_BOTH_INLINE static constexpr uint64_t min()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint64_t lowest()
    {
        return 0U;
    }
    CUDA_BOTH_INLINE static constexpr uint64_t max()
    {
        return ULONG_MAX;
    }
    CUDA_BOTH_INLINE static constexpr uint64_t epsilon()
    {
        return 0U;
    }
};

} // namespace core
} // namespace dw

#endif // DW_CORE_LANGUAGE_LIMITS_HPP_
