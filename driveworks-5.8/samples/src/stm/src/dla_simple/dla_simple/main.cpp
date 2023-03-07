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
#include <stm.h>
#include "dla/testRuntime.hpp"
#include <unistd.h>
#include <stdio.h>

#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif

void confirmUnfinished(void* params)
{
    (void)params;
    auto test = (TestRuntime*)params;

    test->ExpectUnfinished();
}

void confirmFinished(void* params)
{
    auto test = (TestRuntime*)params;

    test->ExpectFinished();
}

void dlaSubmit(void* params, NvMediaDla* dla)
{
    (void)dla; // unused

    auto test = (TestRuntime*)params;
    test->RunTest();
}
#endif

int main(int argc, const char** argv)
{
#if VIBRANTE_PDK_DECIMAL < 6000400
    (void)argc;
    (void)argv;

    // Runtime test
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    TestRuntime test(0, 1);
    status = test.SetUp();
    if (status != NVMEDIA_STATUS_OK)
        throw std::runtime_error("Unable to set up test.");

    stmClientInit("clientDla"); // Needs to be called before registration

    stmRegisterCpuRunnable(confirmUnfinished, "confirmUnfinished", &test);
    stmRegisterCpuRunnable(confirmFinished, "confirmFinished", &test);
    stmRegisterDlaSubmitter(dlaSubmit, "dlaSubmit", &test);

    // Register all resources in the workload
    stmRegisterDlaResource("DLA_HANDLE0", test.GetDlaPtr());

    stmEnterScheduler();

    stmClientExit(); // Removes all STM data structures. Can't use STM calls anymore after this.
#endif
}
