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

#ifndef DW_CORE_CUDAMATH_HPP_
#define DW_CORE_CUDAMATH_HPP_

#include <cuda_runtime.h>
#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/language/Math.hpp>

#include <cmath>

namespace dw
{

namespace cuda
{

/**
 * @brief Return absolute value of @p value
 * @param value input value
 * @return Absolute value of @p value
 */
// TODO(dwplc): FP - std::abs is not overriden since this is dw::core namespace
// coverity[autosar_cpp14_m17_0_3_violation]
CUDA_BOTH_INLINE float32_t abs(float32_t value)
{
#if defined(__CUDA_ARCH__) || defined(__CUDA__)
    return fabs(value);
#else
    return std::abs(value);
#endif // defined(__CUDA_ARCH__) || defined(__CUDA__)
}
/**
 * @brief Return the square root of given input @p v.
 *        Note @p v is not of type float64_t
 *
 * @tparam T data type of @p v
 *
 * @param v input value
 * @return Square root of input @p v
 */
template <typename T>
CUDA_BOTH_INLINE auto cuda_sqrt(T const v) -> T
{
#ifdef __CUDA_ARCH__
    return sqrtf(v);
#else
    return std::sqrt(v);
#endif
}

/**
 * @brief Return the square root of given input @p v.
 *        Note @p v must be of type float64_t
 *
 * @param v input value
 * @return Square root of input @p v
 */
CUDA_BOTH_INLINE float64_t cuda_sqrt(float64_t const v)
{
#ifdef __CUDA_ARCH__
    return sqrt(v);
#else
    return std::sqrt(v);
#endif
}

/**
 * @brief Returns the value of a number rounded to the nearest integer.
 * @param value input value
 * @return Rounded value of @p value
 */
CUDA_BOTH_INLINE float32_t round(float32_t value)
{
#if defined(__CUDA_ARCH__) || defined(__CUDA__)
    return roundf(value);
#else
    return std::round(value);
#endif // defined(__CUDA_ARCH__) || defined(__CUDA__)
}

/**
 * @brief Compute next power of two number closest to given @p v.
 *
 * @note This function runs on device(GPU). For host, use
 *       utils::nextPowerOf2 which is constexpr
 *
 * @param v input value
 * @return Next power of two number closest to @p v
 */
__device__ __forceinline__
    uint32_t
    nextPowerOf2(uint32_t v)
{
    // from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    v--;
    v |= v >> 1U;
    v |= v >> 2U;
    v |= v >> 4U;
    v |= v >> 8U;
    v |= v >> 16U;
    return v + 1;
}

/**
 * @brief Returns rounded up value of (@p a + @p b - 1) / @p b.
 *        This is a helper function to round up when computing grid size for a kernel.
 *        Example:
 *        <tt>dim3 blockSize(128); dim3 gridSize(cuda::roundUpDiv(itemCount, blockSize.x));</tt>
 *
 * @tparam Ta Data type of @p a. Should be of integral types
 * @tparam Tb Data type of @p b. Should be of integral types
 *
 * @param a input value
 * @param b input value
 * @return Rounded up value of (@p a + @p b - 1) / @p b.
 */
template <class Ta, class Tb, class Tc = decltype((std::declval<Ta>() + std::declval<Tb>()) / std::declval<Tb>())>
CUDA_BOTH_INLINE constexpr auto roundUpDiv(Ta a, Tb b) -> Tc
{
    static_assert(std::is_integral<Ta>::value, "This function only makes sense for integral types");
    static_assert(std::is_integral<Tb>::value, "This function only makes sense for integral types");
    return (a + b - Ta(1)) / b;
}

} // namespace cuda
} // namespace dw

#endif // DW_CORE_CUDAMATH_HPP_
