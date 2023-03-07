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

#ifndef _TESTRUNTIME_H_
#define _TESTRUNTIME_H_

#if VIBRANTE_PDK_DECIMAL < 6000400
#include <memory>
#include <string>

#include "dla.hpp"
#include "tensor.hpp"

//! Class to test runtime mode

class TestRuntime final
{
    friend class DlaWorker;

public:
    TestRuntime(
        uint32_t dlaId,
        uint32_t numTasks);

    ~TestRuntime();

    NvMediaStatus SetUp();

    NvMediaStatus RunTest();

    void ExpectFinished();
    void ExpectUnfinished();

    NvMediaDla* GetDlaPtr() { return m_upDla->GetPtr(); }

protected:
    NvMediaStatus InitNvSciBuf(void);

    void DeinitNvSciBuf(void);

    NvMediaStatus ReconcileAndAllocSciBufObj(
        NvMediaTensorAttr tensorAttrs[],
        uint32_t numAttrs,
        NvSciBufObj* sciBuf);

private:
    uint32_t m_dlaId;

    uint32_t m_numTasks;

    uint32_t m_loadableIndex;

    NvMediaDevice* m_device = nullptr;

    std::unique_ptr<Dla> m_upDla;

    std::vector<NvSciBufObj> m_pInputTensorScibuf;

    std::vector<std::unique_ptr<Tensor>> m_vupInputTensor;

    std::vector<NvSciBufObj> m_pOutputTensorScibuf;

    std::vector<std::unique_ptr<Tensor>> m_vupOutputTensor;

    NvSciBufModule m_NvscibufModule = nullptr;
};

#endif
#endif // end of _TESTRUNTIME_H_
