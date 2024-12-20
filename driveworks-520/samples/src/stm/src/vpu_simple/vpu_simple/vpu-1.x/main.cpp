/*
 * Copyright (c) 2020-2024 NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stm.h>
#include <stdlib.h>
#include <stdexcept>

#if VIBRANTE_PDK_DECIMAL >= 6000000
#include <VpuTestContextWrapper.hpp>

void confirmUnfinished(void* params)
{
    auto test = (VpuTestContextWrapper*)params;
    test->testConfirmUnfinished();
}

void confirmFinished(void* params)
{
    auto test = (VpuTestContextWrapper*)params;
    test->testConfirmFinished();
}

void vpuSubmit(void* params, cupvaStream_t vpu)
{
    (void)vpu; //unused
    auto test = (VpuTestContextWrapper*)params;
    test->testVpuSubmit();
}
#endif

// ------------------------------------- //
// testbench                             //
// ------------------------------------- //
int main(int argc, const char** argv)
{
#if VIBRANTE_PDK_DECIMAL >= 6000000
    (void)argc;
    (void)argv;
    VpuTestContextWrapper test;
    test.createStreamAndSetUp();
    test.createCmdProg(0);
    stmErrorCode_t ret;
    ret = stmClientInit(("client0"));
    assert(ret == STM_SUCCESS);
    stmRegisterCpuRunnable(confirmUnfinished, "confirmUnfinished", &test);
    stmRegisterCpuRunnable(confirmFinished, "confirmFinished", &test);

    stmRegisterVpuSubmitter(vpuSubmit, "vpuSubmit0", &test);
    stmRegisterVpuSubmitter(vpuSubmit, "vpuSubmit1", &test);

    // Register all resources in the workload
    stmRegisterVpuResource("PVA_STREAM0", (cupvaStream_t)*test.getStream());

    stmEnterScheduler();

    stmClientExit(); // Removes all STM data structures. Can't use STM calls anymore after this.
#endif
}