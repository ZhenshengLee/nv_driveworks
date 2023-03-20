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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Core Version Information </b>
 *
 * @b Description: This file defines DriveWorks version information and query methods.
 */

/**
 * @defgroup core_version_group Core Version
 * Defines version information and query methods.
 *
 * Unless explicitly specified, all errors returned by DriveWorks APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_CORE_BASE_VERSION_H_
#define DW_CORE_BASE_VERSION_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Status.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __dwVersion
{
    int32_t major; // DriveWorks major version, changes with every product
                   // release with a large new feature set.
    int32_t minor; // DriveWorks minor version, changes with every minor
                   // release containing some features.
    int32_t patch; // DriveWorks patch version, changes with every bugfix
                   // not containing new features.

    char hash[41];  // Globally unique identifier of the DriveWorks sources. (git hash == 40 chars)
    char extra[16]; // Additional string to be appended to version number, e.g., -rc5, -full.
} dwVersion;

/**
 * Query the current DriveWorks library version.
 *
 * @param[out] version A pointer to a dwVersion struct to be filled with
               the current DriveWorks library version.
 *
 * @return DW_INVALID_ARGUMENT if version pointer is null. Provide a valid version pointer.<br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC dwStatus dwGetVersion(dwVersion* const version);

#ifdef __cplusplus
}
#endif
/** @} */

#include <dw/core/base/VersionCurrent.h> // configured file

#endif // DW_CORE_BASE_VERSION_H_
