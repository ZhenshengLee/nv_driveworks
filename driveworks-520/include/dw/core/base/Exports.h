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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Core Exports</b>
 *
 * @b Description: This file defines DW core exports.
 */

/**
 * @defgroup core_exports_group Core Exports
 * @ingroup core_group
 *
 * Defines DW core exports.
 * @{
 */

#ifndef DW_CORE_EXPORTS_H_
#define DW_CORE_EXPORTS_H_

// clang-format off
#if defined(DW_EXPORTS) && !defined(__COVERITY__)
  // coverity[autosar_cpp14_a1_1_1_violation] RFD Accepted: TID-1873
  #define DW_API_PUBLIC __attribute__ ((visibility ("default")))
  // coverity[autosar_cpp14_a1_1_1_violation] RFD Accepted: TID-1873
  #define DW_API_LOCAL  __attribute__ ((visibility ("hidden")))
#else
  #define DW_API_PUBLIC
  #define DW_API_LOCAL
#endif

#if defined(__GNUC__) && !defined(__COVERITY__)
  #define DW_DEPRECATED(msg) __attribute__((deprecated(msg)))
  #if ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100) >= 60000 /* gcc >= 6 */ || defined DW_FORCE_DEPRECATED_ENUM || __clang_major__ >= 10)
     #define DW_DEPRECATED_ENUM(msg) __attribute__ ((deprecated(msg)))
  #else
     #define DW_DEPRECATED_ENUM(msg)
  #endif
#else
  #define DW_DEPRECATED(msg)
  #define DW_DEPRECATED_ENUM(msg)
#endif
// clang-format on

/** @} */
#endif // DW_CORE_EXPORTS_H_
