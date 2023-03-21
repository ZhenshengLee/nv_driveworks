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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <ssm/commServer.hpp>
#include <ssm/commClient.hpp>
#include <ssm/StateMachine.hpp>
#include <SSMClass.hpp>
#include <sm4/SSMClone.hpp>
#include <SM1Class.hpp>
#include <SM2Class.hpp>
#include <SM22Class.hpp>

using std::cout;
using std::endl;

#define MAX_CYCLES 30
typedef std::atomic<int> AtomicInt;
AtomicInt count{0};

SystemStateManager::SM4::SSMClass* ssm{};
extern void startSM(SystemStateManager::SMBaseClass* sm);

void PRE_INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void POST_INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void PRE_READY_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void READY_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void standby_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void normalOperation_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void resetHandler_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void degrade_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void urgentOperation_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void SHUTDOWN_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}
void PRE_SHUTDOWN_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
}

void* startSWC(void* args)
{
    SystemStateManager::SM4::SSMClone* sm{};
    std::string swcName = *(std::string*)args;
    sm                  = new SystemStateManager::SM4::SSMClone(swcName);
    sm->registerFunctionHandler(PRE_INIT_STATE, SystemStateManager::SSMFunctionHandler(PRE_INIT_STATE_fun));
    sm->registerFunctionHandler(INIT_STATE, SystemStateManager::SSMFunctionHandler(INIT_STATE_fun));
    sm->registerFunctionHandler(POST_INIT_STATE, SystemStateManager::SSMFunctionHandler(POST_INIT_STATE_fun));
    sm->registerFunctionHandler(PRE_READY_STATE, SystemStateManager::SSMFunctionHandler(PRE_READY_STATE_fun));
    sm->registerFunctionHandler(READY_STATE, SystemStateManager::SSMFunctionHandler(READY_STATE_fun));
    sm->registerStdFunctionHandler(EXECUTE_RESET_HNDL_STR, SystemStateManager::SSMFunctionHandler(resetHandler_fun));
    sm->registerFunctionHandler("Standby", SystemStateManager::SSMFunctionHandler(standby_fun));
    sm->registerFunctionHandler("NormalOperation", SystemStateManager::SSMFunctionHandler(normalOperation_fun));
    sm->registerFunctionHandler("Degrade", SystemStateManager::SSMFunctionHandler(degrade_fun));
    sm->registerFunctionHandler("UrgentOperation", SystemStateManager::SSMFunctionHandler(urgentOperation_fun));
    sm->registerFunctionHandler(SHUTDOWN_STATE, SystemStateManager::SSMFunctionHandler(SHUTDOWN_STATE_fun));
    sm->registerFunctionHandler(PRE_SHUTDOWN_STATE, SystemStateManager::SSMFunctionHandler(PRE_SHUTDOWN_STATE_fun));
    sm->runInitPhase(200000);

    startSM(sm);
    delete sm;
    return (EXIT_SUCCESS);
}

#define SWCFILE "/tmp/swc_list.txt"
#define TOTAL_SWCS 5

SystemStateManager::SM4::SM1Class* sm1{};
SystemStateManager::SM4::SM2Class* sm2{};
SystemStateManager::SM4::SM22Class* sm22{};

void* startSM1(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES * 2)
    {
        sm1->runStateMachine();
        usleep(1100000);
        counter++;
        if (sm1->isReadyForShutdown())
        {
            break;
        }
    }
    cout << "Terminating SM1" << endl;
    return NULL;
}

void* startSM2(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES * 2)
    {
        sm2->runStateMachine();
        usleep(1100000);
        counter++;
        if (sm2->isReadyForShutdown())
        {
            break;
        }
    }
    cout << "Terminating SM2" << endl;
    return NULL;
}

void* startSM22(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES * 2)
    {
        sm22->runStateMachine();
        usleep(1100000);
        counter++;
        if (sm22->isReadyForShutdown())
        {
            break;
        }
    }
    return NULL;
}

int main()
{
    std::string swcNameList[TOTAL_SWCS];
    SystemStateManager::setupBufferCommunication();
    SystemStateManager::setupLogErrorLevel();
    std::remove(SWCFILE);
    std::ofstream file;
    file.open(SWCFILE);
    for (int i = 0; i < TOTAL_SWCS; i++)
    {
        swcNameList[i] = std::to_string(i + 1);
        file << swcNameList[i] << ",127.0.0.1" << endl;
    }
    file.close();

    pthread_t psm[TOTAL_SWCS];
    pthread_t psm1, psm2, psm22;

    for (int i = 0; i < TOTAL_SWCS; i++)
    {
        pthread_create(&psm[i], NULL, &startSWC, &swcNameList[i]);
    }

    ssm = new SystemStateManager::SM4::SSMClass();
    ssm->initHierarchy();
    ssm->initialize();

    sm1 = new SystemStateManager::SM4::SM1Class();
    sm1->initHierarchy();

    sm2 = new SystemStateManager::SM4::SM2Class();
    sm2->initHierarchy();

    sm22 = new SystemStateManager::SM4::SM22Class();
    sm22->initHierarchy();

    SystemStateManager::StringVector vec;
    SystemStateManager::StringVector vec1;
    SystemStateManager::StringVector vec2;
    ssm->getPrimaryInitList(vec);
    logStringVector(vec, "Contents of primaryInitList");
    cout << "============" << endl;

    SystemStateManager::StringVector sx;
    sx.push_back("3");
    sx.push_back("4");
    sx.push_back("SM22");
    ssm->removeFromPrimaryInitList(sx);
    ssm->getPrimaryInitList(vec1);
    logStringVector(vec1, "Contents of primaryInitList after removal");
    cout << "============" << endl;

    // SSM waits until all the state machines initialize
    while (1)
    {
        if (ssm->areClientsReady())
        {
            break;
        }
        sleep(1);
    }

    ssm->initClientComm();
    sm1->initClientComm();
    sm2->initClientComm();
    sm22->initClientComm();

    pthread_create(&psm1, NULL, &startSM1, NULL);
    pthread_create(&psm2, NULL, &startSM2, NULL);
    pthread_create(&psm22, NULL, &startSM22, NULL);

    ssm->startClients();

    // ======== Now run init phase ========== //
    if (!ssm->executeInitPhase(PRE_INIT_PHASE))
    {
        cout << "Unable to initialize state machines" << endl;
        std::exit(1);
    }

    ssm->executeInitPhase(INIT_PHASE);
    ssm->executeInitPhase(POST_INIT_PHASE);
    ssm->executeInitPhase(PRE_READY_PHASE);
    ssm->executeInitPhase(READY_PHASE);
    ssm->changeInitPhase(ENTER_STATE_MACHINE);
    // ======================================== //

    ssm->runStateMachine();
    ssm->runStateMachine();
    ssm->runStateMachine();

    logStringVector(sx, "Contents of final primary list before initializing deferred list");
    ssm->initializeDeferredList(sx);

    SystemStateManager::StringVector sx2;
    ssm->getPrimaryInitList(sx2);
    logStringVector(sx2, "Contents of primaryInitList after deferred Init");

    int counter = 0;
    while (counter < MAX_CYCLES)
    {
        ssm->runStateMachine();
        usleep(1100000);
        counter++;
    }

    std::cout << "[SSM] Pre-Shut down SSM : INIT" << std::endl;
    ssm->preShutdown();
    while (!ssm->isPreShutdownPhaseComplete())
    {
        usleep(1100000);
    }

    std::cout << "[SSM] Pre-Shut down SSM : DONE" << std::endl;
    ssm->generatePerfHistogram();
    ssm->shutdown();

    for (int i = 0; i < TOTAL_SWCS; i++)
    {
        pthread_join(psm[i], NULL);
    }
    pthread_join(psm1, NULL);
    pthread_join(psm2, NULL);
    pthread_join(psm22, NULL);

    delete ssm;

    return (EXIT_SUCCESS);
}
