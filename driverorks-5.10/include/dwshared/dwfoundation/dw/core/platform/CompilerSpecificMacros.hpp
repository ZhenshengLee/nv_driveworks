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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/** @file
* Info about compiler and compiler specific macros.
* Some info have been taken from this article: http://nadeausoftware.com/articles/2012/10/c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
*/

#ifndef DW_CORE_PLATFORM_COMPILER_SPECIFIC_MACROS_HPP_
#define DW_CORE_PLATFORM_COMPILER_SPECIFIC_MACROS_HPP_

#include <type_traits>

// clang-format off

#define DW_STRINGIFY(text) #text

#if defined(__GNUC__) || defined(__GNUG__)
    #define DW_COMPILER_IS_GCC

    #define DW_COMPILER_NAME "GNU GCC/G++"
    #define DW_COMPILER_VERSION_STRING_LONG __VERSION__

    #define DW_FUNCTION_NAME __PRETTY_FUNCTION__
    #define DW_THREAD_LOCAL thread_local

    #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100) >= 70000 // gcc >= 7
        // Implicit fallthrough attributes (used to explicitly indicate a missing break in a switch case statement)
        #define DW_FALLTHROUGH [[fallthrough]]
    #else
        #define DW_FALLTHROUGH  // qnx7 uses gcc 5.4
    #endif
#else
    static_assert(false, "Unsupported compier");
#endif // host compiler (GNU / MSC)

// Aligned keyword
#ifdef __CUDACC__
    #define DW_RESTRICT         __restrict__
    #define DW_ALIGNED(BYTES)   __align__(BYTES)
    #define CUDA_BOTH           __host__ __device__
    #define CUDA_BOTH_INLINE    __forceinline__ __host__ __device__
    #define CUDA_INLINE         __forceinline__ __device__
#else
    // Empty defines when not using NVCC so IDE doesn't complain
    #define __launch_bounds__(...)
    #define __syncthreads()

    // CUDA_BOTH is defined as empty for non-nvcc so that we can define a CUDA_BOTH function in a header
    // and include in both nvcc- and gcc-compiled files
    // Deviation rationale: Required for CUDA related compiler keywords
    // coverity[autosar_cpp14_a16_0_1_violation]
    #define CUDA_BOTH

    // CUDA_INLINE is intentionally not defined here because such a function should never be
    // included in a gcc-compiled file

    // Deviation rationale: Compiler abstraction - restrict keyword
    // coverity[autosar_cpp14_a16_0_1_violation]
    #define DW_RESTRICT         __restrict

    #if defined(DW_COMPILER_IS_GCC)
        // Deviation rationale: Compiler Abstraction - allignment instructions
        // coverity[autosar_cpp14_a16_0_1_violation]
        #define DW_ALIGNED(BYTES)   alignas(BYTES)
        // Deviation rationale: Required for CUDA related compiler keywords - inline instruction
        // coverity[autosar_cpp14_a16_0_1_violation]
        #define CUDA_BOTH_INLINE    __inline__
    #else
        // Deviation rationale: Check for unsuported compilers
        // coverity[autosar_cpp14_a16_0_1_violation]
        static_assert(false, "Unsupported compier");
    #endif
#endif // __CUDACC__

// clang-format on

#endif // DW_CORE_PLATFORM_COMPILER_SPECIFIC_MACROS_HPP_
