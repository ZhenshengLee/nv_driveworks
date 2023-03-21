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

#define MAX_CYCLES 30

typedef std::atomic<int> AtomicInt;
AtomicInt count{0};
SystemStateManager::SM2::SSMClass* ssm{};
SystemStateManager::SM2::SM1Class* sm1{};
SystemStateManager::SM2::SM2Class* sm2{};

void printLog(SystemStateManager::SSM_LOG_TYPE slt, std::string timestr, std::string logMsg)
{
    std::cout << timestr << " " << logMsg << std::endl;
}

void* startSSM(void* args)
{
    ssm->startClients();

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
        ssm->runStateMachine();
        usleep(500000);
        counter++;
    }
    ssm->generatePerfHistogram();
    return NULL;
}

void* startSM(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES)
    {
        sm1->runStateMachine();
        sm2->runStateMachine();
        usleep(550000);
        counter++;
    }
    return NULL;
}

int main()
{
    pthread_t pssm, psm;
    SystemStateManager::setupLogFunction(printLog);
    SystemStateManager::setupBufferCommunication();
    ssm = new SystemStateManager::SM2::SSMClass();
    ssm->initHierarchy();
    ssm->initialize();

    sm1 = new SystemStateManager::SM2::SM1Class();
    sm1->initHierarchy();

    sm2 = new SystemStateManager::SM2::SM2Class();
    sm2->initHierarchy();

    // Look at the JSON file and you will see that SM2 is disabled by default
    // Lets verify that using the following API
    if (ssm->isSMPresentInPrimaryList(SSM_SM2_str)) {
        // Control comes here if the state machine remains enabled during initialization
        // Check the JSON file (SM2), ensure that "disableByDefault" is set to "true"
        std::cout << "ERROR: SM2 is enabled by default and shouldn't be; please chedk the JSON file" << std::endl;
        std::exit(1);
    }

    // Now lets enable SM2; by adding it to the primary list
    // SSM framework ensures that all the state machines added to the primary list
    // are enabled all the time.
    SystemStateManager::StringVector vec;
    vec.push_back(SSM_SM2_str);
    if (!ssm->addToPrimaryInitList(vec)) {
        std::cout << "Unable to add state machine: " << SSM_SM2_str << std::endl;
        std::exit(1);
    }

    // Now, lets check if SM2 is in the primary list
    if (!ssm->isSMPresentInPrimaryList(SSM_SM2_str)) {
        std::cout << "ERROR: SM2 is still disabled" << std::endl;
        std::exit(1);
    }


    // SSM waits until all the state machines initialize
    while (1) {
        std::cout << "Waiting for clients to be ready..." << std::endl;
        if (ssm->areClientsReady()) {
            break;
        }
        sleep(1);
    }

    //==================================
    ssm->initClientComm();
    sm1->initClientComm();
    sm2->initClientComm();
    //==================================

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
