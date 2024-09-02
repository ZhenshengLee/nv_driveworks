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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_GPUPROPERTIES_H_
#define DW_CORE_GPUPROPERTIES_H_
#include <stdint.h>

/**
 * @brief GPU device properties.
 *
 * Redefinition of cudaDeviceProp for the safety build.
 */
typedef struct dwGPUDeviceProperties
{
    /// Major version
    int32_t major;
    /// Minor version
    int32_t minor;
    /// Memory bus width
    int32_t memoryBusWidth;
    /// L2 cache size
    int32_t l2CacheSize;
    /// Max number of textures 1D
    int32_t maxTexture1D;
    /// Max number of textures 2D
    int32_t maxTexture2D[2];
    /// Max number of textures 3D
    int32_t maxTexture3D[3];
    /// Total memory
    int32_t totalConstMem;
    /// Total shared memory per block
    int32_t sharedMemPerBlock;
    /// Registers per block
    int32_t regsPerBlock;
    /// Warp size
    int32_t warpSize;
    /// Max threads per processor
    int32_t maxThreadsPerMultiProcessor;
    /// Max threads per block
    int32_t maxThreadsPerBlock;
    /// Max thread size
    int32_t maxThreadsDim[3];
    /// Max grid size
    int32_t maxGridSize[3];
    /// Memory pitch
    int32_t memPitch;
    /// Texture alignment
    int32_t textureAlignment;
    /// Device overlap
    int32_t deviceOverlap;
    /// Number of async engines
    int32_t asyncEngineCount;
    /// Kernel timeout enabled flag
    int32_t kernelExecTimeoutEnabled;
    /// Integrated GPU flag
    int32_t integrated;
    /// Can map host memory flag
    int32_t canMapHostMemory;
    /// ECC enabled flag
    int32_t eccEnabled;
    /// Unified addressing flag
    int32_t unifiedAddressing;
    /// The PCI domain ID
    int32_t pciDomainID;
    /// The PCI bus ID
    int32_t pciBusID;
    /// The PCI device ID
    int32_t pciDeviceID;
    /// Number of concurrent kernels
    int32_t concurrentKernels;
    /// Compute mode flag
    int32_t computeMode;
    /// Pitch alignment supported by texture unit
    int32_t devAttrTexturePitchAlignment;
} dwGPUDeviceProperties;

#endif //DW_CORE_GPUPROPERTIES_H_
