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
#include <iostream>

#include "dla.hpp"
#include "drivenet.hpp"

Dla::~Dla()
{
    for (auto m_vLoadable : m_vLoadables)
    {
        if (m_vLoadable)
        {
            NvMediaDlaLoadableDestroy(m_pDla, m_vLoadable);
        }
    }

    m_vLoadables.clear();

    if (m_pDla)
    {
        NvMediaDlaDestroy(m_pDla);
        m_pDla = nullptr;
    }
}

Dla::Dla(NvMediaDla* dla)
    : m_pDla(dla)
{
}

std::unique_ptr<Dla> Dla::Create()
{
    NvMediaDla* dla = NvMediaDlaCreate();
    if (dla == nullptr)
        throw std::runtime_error("NvMediaDlaCreate");

    return std::unique_ptr<Dla>(new Dla(dla));
}

NvMediaStatus Dla::Init(uint32_t dlaId, uint32_t numTasks)
{
    auto status = NvMediaDlaInit(m_pDla, dlaId, numTasks);

    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaInit");

    return status;
}

NvMediaStatus Dla::AddLoadable(uint32_t& loadableIndex)
{
    NvMediaDlaBinaryLoadable binaryLoadable{};
    NvMediaStatus status{NVMEDIA_STATUS_OK};
    NvMediaDlaLoadable* loadable{nullptr};

    binaryLoadable.loadable     = drivenet_nvdla;
    binaryLoadable.loadableSize = sizeof(drivenet_nvdla);

    // Create loadable handle
    status = NvMediaDlaLoadableCreate(m_pDla, &loadable);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaLoadableCreate");

    // Append loadable
    status = NvMediaDlaAppendLoadable(m_pDla, binaryLoadable, loadable);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaAppendLoadable");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, loadable);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    // Load loadable
    status = NvMediaDlaLoadLoadable(m_pDla);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaLoadLoadable");

    m_vLoadables.push_back(loadable);

    loadableIndex = m_vLoadables.size() - 1;
    return status;
}

NvMediaStatus Dla::GetDesc(
    uint32_t loadableIndex,
    std::vector<NvMediaDlaTensorDescriptor>& vInputTensorDesc,
    std::vector<NvMediaDlaTensorDescriptor>& vOutputTensorDesc)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaDlaTensorDescriptor tensorDesc{};
    int32_t num{0};

    if (!(vInputTensorDesc.size() == 0 && vOutputTensorDesc.size() == 0))
        throw std::runtime_error("Check descriptor argument");

    if (loadableIndex >= m_vLoadables.size())
        throw std::runtime_error("Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    // Input tensor
    status = NvMediaDlaGetNumOfInputTensors(m_pDla, &num);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaGetNumOfInputTensors");

    for (auto i = 0; i < num; i++)
    {
        status = NvMediaDlaGetInputTensorDescriptor(m_pDla, i, &tensorDesc);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("NvMediaDlaGetInputTensorDescriptor");
        status = PrintTensorDesc(&tensorDesc);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("PrintTensorDesc of input tensor");
        vInputTensorDesc.push_back(tensorDesc);
    }

    // Output tensor
    status = NvMediaDlaGetNumOfOutputTensors(m_pDla, &num);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaGetNumOfOutputTensors");

    for (auto i = 0; i < num; i++)
    {
        status = NvMediaDlaGetOutputTensorDescriptor(m_pDla, i, &tensorDesc);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("NvMediaDlaGetOutputTensorDescriptor");
        status = PrintTensorDesc(&tensorDesc);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("PrintTensorDesc of output tensor");
        vOutputTensorDesc.push_back(tensorDesc);
    }

    return status;
}

NvMediaStatus Dla::DataRegister(
    uint32_t loadableIndex,
    Tensor* tensor)
{
    NvMediaDlaData dlaData{};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (loadableIndex >= m_vLoadables.size())
        throw std::runtime_error("Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    dlaData.type           = NVMEDIA_DLA_DATA_TYPE_TENSOR;
    dlaData.pointer.tensor = tensor->GetPtr();
    status                 = NvMediaDlaDataRegister(m_pDla, &dlaData, 0);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaDataRegister");

    return status;
}

NvMediaStatus Dla::DataUnregister(
    uint32_t loadableIndex,
    Tensor* tensor)
{
    NvMediaDlaData dlaData{};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (loadableIndex >= m_vLoadables.size())
        throw std::runtime_error("Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    dlaData.type           = NVMEDIA_DLA_DATA_TYPE_TENSOR;
    dlaData.pointer.tensor = tensor->GetPtr();
    status                 = NvMediaDlaDataUnregister(m_pDla, &dlaData);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaDataUnregister");

    return status;
}

NvMediaStatus Dla::RemoveLoadable(uint32_t loadableIndex)
{
    auto status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    status = NvMediaDlaRemoveLoadable(m_pDla);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaRemoveLoadable");

    return status;
}

NvMediaStatus Dla::Submit(
    uint32_t loadableIndex,
    std::vector<Tensor*>& vpInputTensor,
    std::vector<Tensor*>& vpOutputTensor)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaDlaArgs inputArgs{};
    NvMediaDlaArgs outputArgs{};

    if (!(vpInputTensor.size() < MAX_NUM_OF_DLA_DATA && vpOutputTensor.size() < MAX_NUM_OF_DLA_DATA))
        throw std::runtime_error("Check input args");

    if (loadableIndex >= m_vLoadables.size())
        throw std::runtime_error("Check loadable index argument");

    // input tensor
    for (auto i = 0u; i < vpInputTensor.size(); i++)
    {
        m_aInputDlaData[i].type           = NVMEDIA_DLA_DATA_TYPE_TENSOR;
        m_aInputDlaData[i].pointer.tensor = vpInputTensor[i]->GetPtr();
    }
    inputArgs.dlaData = m_aInputDlaData.data();
    inputArgs.numArgs = vpInputTensor.size();

    // output tensor
    for (auto i = 0u; i < vpOutputTensor.size(); i++)
    {
        m_aOutputDlaData[i].type           = NVMEDIA_DLA_DATA_TYPE_TENSOR;
        m_aOutputDlaData[i].pointer.tensor = vpOutputTensor[i]->GetPtr();
    }
    outputArgs.dlaData = m_aOutputDlaData.data();
    outputArgs.numArgs = vpOutputTensor.size();

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSetCurrentLoadable");

    status = NvMediaDlaSubmit(m_pDla, &inputArgs, NULL, &outputArgs, NVMEDIA_DLA_DEFAULT_TASKTIMEOUT);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaDlaSubmit");

    return status;
}

NvMediaStatus Dla::PrintTensorDesc(NvMediaDlaTensorDescriptor* tensorDesc)
{
    uint32_t i = 0;
    std::cout << "Tensor descripor" << std::endl;
    std::cout << "\t name = " << tensorDesc->name << std::endl;

    for (i = 0; i < tensorDesc->numAttrs; i++)
    {
        switch (tensorDesc->tensorAttrs[i].type)
        {
        case NVM_TENSOR_ATTR_DATA_TYPE:
            std::cout << "\t Data type = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_BITS_PER_ELEMENT:
            std::cout << "\t Bits per element = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_DIMENSION_ORDER:
            std::cout << "\t dimension order = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_CPU_ACCESS:
            std::cout << "\t CPU access = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_ALLOC_TYPE:
            std::cout << "\t Alloc type = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_4D_N:
            std::cout << "\t N = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_4D_C:
            std::cout << "\t C = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_4D_H:
            std::cout << "\t H = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_4D_W:
            std::cout << "\t W = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_4D_X:
            std::cout << "\t X = " << tensorDesc->tensorAttrs[i].value << std::endl;
            break;
        case NVM_TENSOR_ATTR_MAX:
        default:
            return NVMEDIA_STATUS_ERROR;
        }
    }

    return NVMEDIA_STATUS_OK;
}
#endif
