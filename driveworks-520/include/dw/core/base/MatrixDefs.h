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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_MATRIXDEFS_H_
#define DW_CORE_MATRIXDEFS_H_

#include "MatrixTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Identity for dwMatrix2f
static const dwMatrix2f DW_IDENTITY_MATRIX2F = {{(float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)1}};
/// Identity for dwMatrix3f
static const dwMatrix3f DW_IDENTITY_MATRIX3F = {{(float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1}};
/// Identity for dwMatrix3d
static const dwMatrix3d DW_IDENTITY_MATRIX3D = {{(float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1}};
/// Identity for dwMatrix4f
static const dwMatrix4f DW_IDENTITY_MATRIX4F = {{(float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1}};
/// Identity for dwMatrix6f
static const dwMatrix6f DW_IDENTITY_MATRIX6F = {{(float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1}};
/// Identity for dwTransformation2f
static const dwTransformation2f DW_IDENTITY_TRANSFORMATION2F = {{1.f, 0.f, 0.f,
                                                                 0.f, 1.f, 0.f,
                                                                 0.f, 0.f, 1.f}};
/// Identity for dwTransformation3f
static const dwTransformation3f DW_IDENTITY_TRANSFORMATION3F = {{1.f, 0.f, 0.f, 0.f,
                                                                 0.f, 1.f, 0.f, 0.f,
                                                                 0.f, 0.f, 1.f, 0.f,
                                                                 0.f, 0.f, 0.f, 1.f}};

#ifdef __cplusplus
}
#endif

#endif // DW_CORE_MATRIXDEFS_H_
