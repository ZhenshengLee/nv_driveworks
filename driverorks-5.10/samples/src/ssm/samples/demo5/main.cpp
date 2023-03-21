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

#include <ssm/commServer.hpp>
#include <ssm/commClient.hpp>
#include <ssm/StateMachine.hpp>
#include <ssm/SSMHistogram.hpp>
#include <SSMClass.hpp>
#include <SM1Class.hpp>
#include <SM2Class.hpp>

#define MAX_CYCLES 50

typedef std::atomic<int> AtomicInt;
AtomicInt count{0};
SystemStateManager::SM5::SSMClass* ssm{};
SystemStateManager::SM5::SM1Class* sm1{};
SystemStateManager::SM5::SM2Class* sm2{};

typedef struct _tt
{
    int a;
    int b;
    int x;
    char xx[10];
} TT;

void printLogMessage(std::string logStr)
{
    std::cout << logStr << std::endl;
}

void* startSSM(void* args)
{
    ssm->startClients();
    SystemStateManager::SM5::UserDataPkt pkt{};

    // ======== Now run init phase ========== //
    ssm->executeInitPhase(PRE_INIT_PHASE);
    ssm->executeInitPhase(INIT_PHASE);
    ssm->executeInitPhase(POST_INIT_PHASE);
    ssm->executeInitPhase(PRE_READY_PHASE);
    ssm->executeInitPhase(READY_PHASE);
    ssm->changeInitPhase(ENTER_STATE_MACHINE);
    // ======================================== //

    int counter = 0;
    while (counter < MAX_CYCLES)
    {

        /// Query for data //////
        if (ssm->getUserSpecificData(pkt))
        {
            TT a = *((TT*)pkt.data);
            std::cout << "#### Received Data SSM #####" << std::endl;
            std::cout << " TS:  " << pkt.timestamp << std::endl;
            std::cout << " DT:  " << pkt.dataType << std::endl;
            std::cout << " Sz:  " << pkt.size << std::endl;
            std::cout << " a :  " << a.a << std::endl;
            std::cout << " b :  " << a.b << std::endl;
            std::cout << " c :  " << a.x << std::endl;
        }

        ssm->runStateMachine();
        usleep(110000);
        counter++;
    }
    ssm->generatePerfHistogram();
    return NULL;
}

void* startSM(void* args)
{
    int counter = 0;
    SystemStateManager::SM5::UserDataPkt pkt{};
    while (counter < MAX_CYCLES)
    {

        /// Query for data //////
        if (sm2->getUserSpecificData(pkt))
        {
            TT a = *((TT*)pkt.data);
            std::cout << "#### Received Data SM2 #####" << std::endl;
            std::cout << " TS:  " << pkt.timestamp << std::endl;
            std::cout << " DT:  " << pkt.dataType << std::endl;
            std::cout << " Sz:  " << pkt.size << std::endl;
            std::cout << " a :  " << a.a << std::endl;
            std::cout << " b :  " << a.b << std::endl;
            std::cout << " c :  " << a.x << std::endl;
        }

        /// Query for data //////
        if (sm1->getUserSpecificData(pkt))
        {
            TT a = *((TT*)pkt.data);
            std::cout << "#### Received Data SM1 #####" << std::endl;
            std::cout << " TS:  " << pkt.timestamp << std::endl;
            std::cout << " DT:  " << pkt.dataType << std::endl;
            std::cout << " Sz:  " << pkt.size << std::endl;
            std::cout << " a :  " << a.a << std::endl;
            std::cout << " b :  " << a.b << std::endl;
            std::cout << " c :  " << a.x << std::endl;

            TT t{};
            t.a = 9999;
            t.b = 9999;
            t.x = 9999;

            // SM1 sends data to SSM using enums
            if (!sm1->sendDataByID(SystemStateManager::SM5::StateMachines::SSM, (void*)&t, sizeof(TT), "1"))
            {
                std::cout << "error" << std::endl;
                std::exit(0);
            }
        }
        sm1->runStateMachine();
        sm2->runStateMachine();
        usleep(120000);
        counter++;
    }
    return NULL;
}

int main()
{
    pthread_t pssm{};
    pthread_t psm{};
    SystemStateManager::setupBufferCommunication();
    SystemStateManager::setupLogErrorLevel();
    ssm = new SystemStateManager::SM5::SSMClass();
    ssm->initHierarchy();
    ssm->initialize();

    sm1 = new SystemStateManager::SM5::SM1Class();
    sm1->initHierarchy();

    sm2 = new SystemStateManager::SM5::SM2Class();
    sm2->initHierarchy();

    // SSM waits until all the state machines initialize
    while (1)
    {
        if (ssm->areClientsReady())
        {
            break;
        }
        sleep(1);
    }

    //==================================
    ssm->initClientComm();
    sm1->initClientComm();
    sm2->initClientComm();
    //==================================

    TT t{};
    t.a = 989;
    t.b = 777;
    t.x = 343;

    // SM1 sends data to SSM
    if (!sm1->sendData("SSM", (void*)&t, sizeof(TT), "1"))
    {
        std::cout << "error" << std::endl;
        std::exit(0);
    }

    t.a = 111;
    t.b = 222;
    t.x = 333;

    // SSM sends data to SM1
    if (!ssm->sendData("SM1", (void*)&t, sizeof(TT), "1"))
    {
        std::cout << "error" << std::endl;
        std::exit(0);
    }

    pthread_create(&psm, NULL, &startSM, NULL);
    usleep(600000);
    pthread_create(&pssm, NULL, &startSSM, NULL);

    pthread_join(pssm, NULL);
    pthread_join(psm, NULL);

    std::cout << "=======================" << std::endl;

    delete ssm;
    delete sm1;
    delete sm2;
    return (EXIT_SUCCESS);
}
