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
 * <b>NVIDIA DriveWorks API: NvMedia Includes</b>
 *
 * @b Description: This file defines NvMedia API on Drive platforms.
 */

#ifndef DW_CORE_NVMEDIA_H_
#define DW_CORE_NVMEDIA_H_

#include <dw/core/base/Config.h>
#include <sys/time.h>

#if (defined(VIBRANTE) && (VIBRANTE_PDK_DECIMAL < 6000400))
#include <nvmedia_core.h>
#include <nvmedia_surface.h>
#include <nvmedia_2d.h>
#endif

#if (defined(VIBRANTE) && (VIBRANTE_PDK_DECIMAL >= 6000400)) || defined(LINUX_AND_EMU)
#include <nvmedia_6x/nvmedia_core.h>
#include <nvmedia_6x/nvmedia_2d.h>
#ifndef DW_IS_SAFETY
#include <nvmedia_6x/nvmedia_parser.h>
#include <nvmedia_6x/nvmedia_ide.h>
#if (defined(VIBRANTE) && (VIBRANTE_PDK_DECIMAL <= 6000600)) || defined(LINUX_AND_EMU)
#include <nvmedia_6x/nvmedia_nvscibuf.h>
#endif
#include <nvmedia_6x/nvmedia_ijpe.h>
#endif
#include <nvmedia_6x/nvmedia_iep.h>
#include <nvscibuf.h>
#endif

#ifndef DW_IS_SAFETY
// NvMedia deprecated nvmedia_vop.h from DOS6.
#if (defined(VIBRANTE) && (VIBRANTE_PDK_DECIMAL < 6000000))
#include <nvmedia_vop.h>
#endif

#if (defined(VIBRANTE) && (VIBRANTE_PDK_DECIMAL < 6000400))
#include <nvmedia_video.h>
#include <nvmedia_vmp.h>
#include <nvmedia_vep.h>
#include <nvmedia_viddec.h>
#endif
#endif // DW_IS_SAFETY

#else
/// Dummy definition for non NvMedia supported platforms
typedef void NvMediaIPPManager;
#endif // DW_CORE_NVMEDIA_H_
