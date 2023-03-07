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

#if VIBRANTE_PDK_DECIMAL < 6000400
#include <cstring>
#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor(NvMediaDevice* device)
    : m_pDevice(device), m_pTensor(nullptr)
{
}

Tensor::~Tensor()
{
    m_pDevice = nullptr;

    if (m_pTensor)
    {
        NvMediaTensorDestroy(m_pTensor);
    }
}

NvMediaStatus Tensor::FillNvSciBufTensorAttrs(
    NvMediaDevice* device,
    NvMediaTensorAttr tensorAttrs[],
    uint32_t numAttrs,
    NvSciBufAttrList attr_h)
{
    return NvMediaTensorFillNvSciBufAttrs(device, tensorAttrs, numAttrs, 0, attr_h);
}

NvMediaStatus Tensor::Create(NvSciBufObj bufObj, uint8_t initValue)
{
    NvMediaTensorSurfaceMap tensorMap{};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaTensorCreateFromNvSciBuf(m_pDevice,
                                             bufObj,
                                             &m_pTensor);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaTensorCreateFromNvSciBuf");

    status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &tensorMap);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaTensorLock");

    memset(tensorMap.mapping, initValue, tensorMap.size);
    NvMediaTensorUnlock(m_pTensor);

    return status;
}

NvMediaTensor* Tensor::GetPtr() const
{
    return m_pTensor;
}

NvMediaStatus Tensor::ExpectStatus(NvMediaStatus expectedStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaTensorTaskStatus taskStatus{};

    status = NvMediaTensorGetStatus(m_pTensor, 1, &taskStatus);
    if (status != NVMEDIA_STATUS_OK)
        return status;

    std::cout << "Operation duration: " << taskStatus.durationUs << " us." << std::endl;
    if (taskStatus.status != expectedStatus)
        throw std::runtime_error("Expected " + std::to_string(expectedStatus) + " but found " + std::to_string(taskStatus.status) + ".");

    return status;
}
#endif
