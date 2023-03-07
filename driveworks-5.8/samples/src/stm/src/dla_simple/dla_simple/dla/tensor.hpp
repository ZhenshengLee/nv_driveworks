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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _TENSOR_H_
#define _TENSOR_H_

#if VIBRANTE_PDK_DECIMAL < 6000400
#include <string>

#include "nvmedia_core.h"
#include "nvmedia_tensor.h"
#include "nvmedia_tensor_nvscibuf.h"

//! Class for create and destroy NvMedia Tensor from NvSciBuf
class Tensor
{
public:
    static NvMediaStatus FillNvSciBufTensorAttrs(
        NvMediaDevice* device,
        NvMediaTensorAttr tensorAttrs[],
        uint32_t numAttrs,
        NvSciBufAttrList attr_h);

    Tensor(NvMediaDevice* device);

    NvMediaStatus Create(NvSciBufObj bufObj, uint8_t initValue);

    NvMediaTensor* GetPtr() const;

    NvMediaStatus ExpectStatus(NvMediaStatus expectedStatus);

    virtual ~Tensor();

protected:
    NvMediaDevice* m_pDevice;

    NvMediaTensor* m_pTensor;
};

#endif
#endif // end of _TENSOR_H_
