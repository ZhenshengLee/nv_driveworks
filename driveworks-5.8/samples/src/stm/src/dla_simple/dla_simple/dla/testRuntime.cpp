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
#include <iterator>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "testRuntime.hpp"

TestRuntime::TestRuntime(uint32_t dlaId, uint32_t numTasks)
    : m_dlaId(dlaId), m_numTasks(numTasks), m_loadableIndex(0)
{
}

TestRuntime::~TestRuntime()
{
#if defined(VIBRANTE) && VIBRANTE_PDK_DECIMAL < 6000400
    if (m_device)
    {
        NvMediaDeviceDestroy(m_device);
    }
#endif //#if defined(VIBRANTE) && VIBRANTE_PDK_DECIMAL < 6000400

    for (auto i = 0u; i < m_vupInputTensor.size(); i++)
    {
        m_upDla->DataUnregister(m_loadableIndex, m_vupInputTensor[i].get());
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++)
    {
        m_upDla->DataUnregister(m_loadableIndex, m_vupOutputTensor[i].get());
    }

    m_upDla->RemoveLoadable(m_loadableIndex);

    m_upDla.reset();

    for (auto i = 0u; i < m_pInputTensorScibuf.size(); i++)
    {
        if (m_pInputTensorScibuf[i])
        {
            NvSciBufObjFree(m_pInputTensorScibuf[i]);
        }
    }

    for (auto i = 0u; i < m_vupInputTensor.size(); i++)
    {
        m_vupInputTensor[i].reset();
    }

    for (auto i = 0u; i < m_pOutputTensorScibuf.size(); i++)
    {
        if (m_pOutputTensorScibuf[i])
        {
            NvSciBufObjFree(m_pOutputTensorScibuf[i]);
        }
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++)
    {
        m_vupOutputTensor[i].reset();
    }

    DeinitNvSciBuf();
}

NvMediaStatus TestRuntime::SetUp()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    std::vector<NvMediaDlaTensorDescriptor> vInputTensorDesc;
    std::vector<NvMediaDlaTensorDescriptor> vOutputTensorDesc;
    NvSciBufObj sciBufObj;

    status = InitNvSciBuf();
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("InitNvSciBuf");

#if defined(VIBRANTE) && VIBRANTE_PDK_DECIMAL < 6000400
    m_device = NvMediaDeviceCreate();
    if (m_device == nullptr)
        throw std::runtime_error("NvMediaDeviceCreate failed");
#else
    m_device = nullptr;
#endif

    m_upDla = Dla::Create();
    if (m_upDla == nullptr)
        throw std::runtime_error("Create");

    status = m_upDla->Init(m_dlaId, m_numTasks);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("Init");

    status = m_upDla->AddLoadable(m_loadableIndex);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("AddLoadable");

    status = m_upDla->GetDesc(m_loadableIndex, vInputTensorDesc, vOutputTensorDesc);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("GetDesc");

    // input tensor allocation
    for (auto i = 0u; i < vInputTensorDesc.size(); i++)
    {
        status = ReconcileAndAllocSciBufObj(vInputTensorDesc[i].tensorAttrs, vInputTensorDesc[i].numAttrs, &sciBufObj);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("ReconcileAndAllocSciBufObj");

        m_pInputTensorScibuf.push_back(sciBufObj);

        std::unique_ptr<Tensor> upTensor(new Tensor(m_device));

        status = upTensor->Create(sciBufObj, 0);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("Tensor Create");

        status = m_upDla->DataRegister(m_loadableIndex, upTensor.get());

        m_vupInputTensor.push_back(std::move(upTensor));
    }

    // output tensor allocation
    for (auto i = 0u; i < vOutputTensorDesc.size(); i++)
    {
        status = ReconcileAndAllocSciBufObj(vOutputTensorDesc[i].tensorAttrs, vOutputTensorDesc[i].numAttrs, &sciBufObj);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("ReconcileAndAllocSciBufObj");

        m_pOutputTensorScibuf.push_back(sciBufObj);

        std::unique_ptr<Tensor> upTensor(new Tensor(m_device));

        status = upTensor->Create(sciBufObj, 0);
        if (status != NVMEDIA_STATUS_OK)
            throw std::runtime_error("Tensor Create");

        status = m_upDla->DataRegister(m_loadableIndex, upTensor.get());

        m_vupOutputTensor.push_back(std::move(upTensor));
    }

    return status;
}

NvMediaStatus TestRuntime::RunTest()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    std::vector<Tensor*> vInputTensor;
    std::vector<Tensor*> vOutputTensor;

    for (auto i = 0u; i < m_vupInputTensor.size(); i++)
    {
        vInputTensor.push_back(m_vupInputTensor[i].get());
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++)
    {
        vOutputTensor.push_back(m_vupOutputTensor[i].get());
    }

    status = m_upDla->Submit(m_loadableIndex, vInputTensor, vOutputTensor);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("Submit");

    return status;
}

void TestRuntime::ExpectFinished()
{

    NvMediaStatus status = m_vupOutputTensor[0]->ExpectStatus(NVMEDIA_STATUS_OK);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("Unexpected return status -- expect finished: " + std::to_string(status));
}

void TestRuntime::ExpectUnfinished()
{

    NvMediaStatus status = m_vupOutputTensor[0]->ExpectStatus(NVMEDIA_STATUS_TIMED_OUT);
    if (status != NVMEDIA_STATUS_TIMED_OUT)
        throw std::runtime_error("Unexpected return status -- expect unfinished: " + std::to_string(status));
}

NvMediaStatus TestRuntime::InitNvSciBuf(void)
{
    NvSciError err       = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    err = NvSciBufModuleOpen(&m_NvscibufModule);
    if (err != NvSciError_Success)
        throw std::runtime_error("NvSciBufModuleOpen");

    status = NvMediaTensorNvSciBufInit();
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("NvMediaTensorNvSciBufInit");

    return status;
}

void TestRuntime::DeinitNvSciBuf()
{
    NvSciBufModuleClose(m_NvscibufModule);

    NvMediaTensorNvSciBufDeinit();
}

NvMediaStatus TestRuntime::ReconcileAndAllocSciBufObj(
    NvMediaTensorAttr tensorAttrs[],
    uint32_t numAttrs,
    NvSciBufObj* sciBuf)
{
    NvMediaStatus status                         = NVMEDIA_STATUS_OK;
    NvSciError err                               = NvSciError_Success;
    NvSciBufAttrValAccessPerm access_perm        = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrList unreconciled_attrlistTensor = NULL;
    NvSciBufAttrList reconciled_attrlist         = NULL;
    NvSciBufAttrList conflictlist                = NULL;

    NvSciBufAttrKeyValuePair attr_kvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                         &access_perm,
                                         sizeof(access_perm)};

    err = NvSciBufAttrListCreate(m_NvscibufModule, &unreconciled_attrlistTensor);
    if (err != NvSciError_Success)
        throw std::runtime_error("NvSciBufAttrListCreate");

    err = NvSciBufAttrListSetAttrs(unreconciled_attrlistTensor, &attr_kvp, 1);
    if (err != NvSciError_Success)
        throw std::runtime_error("NvSciBufAttrListSetAttrs");

    status = Tensor::FillNvSciBufTensorAttrs(m_device, tensorAttrs, numAttrs, unreconciled_attrlistTensor);
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("GetNvSciBufTensorAttrs");

    err = NvSciBufAttrListReconcile(&unreconciled_attrlistTensor,
                                    1,
                                    &reconciled_attrlist,
                                    &conflictlist);
    if (err != NvSciError_Success)
        throw std::runtime_error("NvSciBufAttrListReconcile");

    err = NvSciBufObjAlloc(reconciled_attrlist,
                           sciBuf);
    if (err != NvSciError_Success)
        throw std::runtime_error("NvSciBufAttrListReconcile");

    if (unreconciled_attrlistTensor)
    {
        NvSciBufAttrListFree(unreconciled_attrlistTensor);
    }
    if (reconciled_attrlist)
    {
        NvSciBufAttrListFree(reconciled_attrlist);
    }
    if (conflictlist)
    {
        NvSciBufAttrListFree(conflictlist);
    }

    return status;
}
#endif
