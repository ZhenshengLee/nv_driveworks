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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Core Object (Extra) Methods</b>
 *
 * @b Description: This file defines the core methods of the SDK.
 */

/**
 * @defgroup core_group Core Interface
 *
 * @brief Defines the core methods of the SDK.
 *
 * Unless explicitly specified, all errors returned by DW API are non recoverable and the user application should transition to fail safe mode.
 *
 */

#ifndef DW_CORE_OBJECT_EXTRA_H_
#define DW_CORE_OBJECT_EXTRA_H_

#include <dw/core/context/Context.h>
#include <dw/core/health/HealthSignals.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DW_OBJECT_MAX_NAME_LEN 128

/**
 * Set the module name for a DW Object
 *
 * @param[in] name Pointer to an C string to contain the name to set the object to
 * @param[in] handle Object handle
 *
 * @return DW_INVALID_ARGUMENT if one of input parameters is NULL.<br>
 *         DW_INVALID_HANDLE if given context handle is not an actual handle.<br>
 *         DW_SUCCESS Successfully set name for an object.
 *
 * @note the maximum length of the name is DW_OBJECT_MAX_NAME_LEN
 */
DW_API_PUBLIC
dwStatus dwObject_setObjectName(char8_t const* const name, dwModuleHandle_t const handle);

#ifdef __cplusplus
}
#endif

#endif // DW_CORE_OBJECT_EXTRA_H_
