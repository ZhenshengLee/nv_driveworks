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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_PLATFORM_SIMD_HPP_
#define DW_CORE_PLATFORM_SIMD_HPP_

/**
 * @brief Conditionally compile NEON/SIMD code based on the compiler and platform being used.
 *        Utilizes GCC compiler's preprocessor directives to control the compiler's warning behavior
 *        and selectively ignore specific warning options as following:<br>
 *        - "-Wunused-parameter" : detects unused function parameters.
 *        - "-Wold-style-cast" : detects old-style type casts.
 *        - "-Wfloat-equal": detects floating-point comparisons.
 *        - "-Wconversion" : detects implicit type conversions.
 *        Use @a DW_ENABLE_ARM_SIMD to gate conditional compile of NEON/SIMD code.<br>
 *        Must push/pop warning config - arm_neon.h has some conflicts with DW default build settings.
 *
 *        WAR: coverity-qnx builds don't work with arm_neon.h yet, so disable SIMD on coverity builds
 */
#if !defined(__CUDACC__) && defined(__GNUC__) && defined(__aarch64__) && !defined(__COVERITY__)

// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic push
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wunused-parameter"
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wold-style-cast"
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wfloat-equal"
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wconversion"
#include <arm_neon.h>
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic pop

#define DW_ENABLE_ARM_SIMD 1 // use this symbol to gate conditional compile of NEON/SIMD code

#else

#define DW_ENABLE_ARM_SIMD 0 //disable SIMD

#endif

#endif // DW_CORE_PLATFORM_SIMD_HPP_
