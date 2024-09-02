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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Extended NvMedia Includes</b>
 *
 * @b Description: This file extends NvMedia API on Drive platforms.
 */

/**
 * @defgroup extended_core_nvmedia_group Core NvMedia
 * @ingroup extended_core_group\
 *
 * @{
 */

#ifndef DW_CORE_NVMEDIA_EXT_H_
#define DW_CORE_NVMEDIA_EXT_H_

#include "NvMedia.h"

#if defined(VIBRANTE) && VIBRANTE_PDK_DECIMAL < 6000400
#include <nvmedia_dla.h>
#else
/// Dummy definition for non NvMedia supported platforms
/// exclude the nvmedia_dla.h.
#ifndef NVMEDIA_DLA_H
#define NVMEDIA_DLA_H
typedef void NvMediaDla;
#endif // #ifndef NVMEDIA_DLA_H
#endif // #if defined(VIBRANTE) && VIBRANTE_PDK_DECIMAL < 6000400
/** @} */
#endif // DW_CORE_NVMEDIA_EXT_H_
