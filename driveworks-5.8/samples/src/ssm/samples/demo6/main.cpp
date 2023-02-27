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
#include <ssm/GenericClone.hpp>
#include <SSMClass.hpp>

using std::cout;
using std::endl;

typedef std::atomic<int> AtomicInt;
AtomicInt count{0};
SystemStateManager::SM6::SSMClass* ssm{};
SystemStateManager::GenericClone *agent{};
#define MAX_CYCLES 30


void PRE_INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  PRE_INIT_STATE_fun" << endl;
}
void INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  INIT_STATE_fun" << endl;
}
void POST_INIT_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  POST_INIT_STATE_fun" << endl;
}
void PRE_READY_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  PRE_READY_STATE_fun" << endl;
}
void READY_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  READY_STATE_fun" << endl;
}
void standby_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  standby_fun" << endl;
}
void normalOperation_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  normalOperation_fun" << endl;
}
void resetHandler_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:  resetHandler_fun" << endl;
}
void degrade_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:     degrade_fun" << endl;
}
void urgentOperation_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:     urgentOperation_fun" << endl;
}
void SHUTDOWN_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:     SHUTDOWN_STATE_fun" << endl;
}
void PRE_SHUTDOWN_STATE_fun(SystemStateManager::SMBaseClass* ptr)
{
    cout << "GClone:     PRE_SHUTDOWN_STATE_fun" << endl;
}

void printLogMessage(std::string logStr)
{
    std::cout << logStr << std::endl;
}

bool safetyModule(SystemStateManager::GenericClone *ptr,
                const char *fromState,
                const char *toState,
                const uint64_t transitionID,
                uint64_t timeout)
{
    cout << "######################## GlobalStateChange: " << fromState << "  " << toState << " " << transitionID << endl;

    sleep(1);

    // You can call this function either within this function handler or outside
    ptr->globalStateChangeComplete(transitionID);
    return true;
}

void* startSSM(void* args)
{
    int counter = 0;

    ssm->executeInitPhase(PRE_INIT_PHASE);
    ssm->executeInitPhase(INIT_PHASE);
    ssm->executeInitPhase(POST_INIT_PHASE);
    ssm->executeInitPhase(PRE_READY_PHASE);
    ssm->executeInitPhase(READY_PHASE);
    ssm->changeInitPhase(ENTER_STATE_MACHINE);

    while (counter < MAX_CYCLES)
    {
        ssm->runStateMachine();
        if (ssm->getCurrentState() == "UrgentOperation")
        {
            break;
        }
        usleep(1000000);
        counter++;
    }
    return NULL;
}

void printLog(SystemStateManager::SSM_LOG_TYPE slt, std::string timestr, std::string logMsg)
{
    cout << timestr << " " << logMsg << endl;
}

void* startSM1(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES)
    {
        agent->runStateMachine();
        usleep(1000000);
        counter++;
        if (agent->isReadyForShutdown())
        {
            break;
        }
    }
    cout << "Terminating Generic Clone" << endl;
    return NULL;
}

void* initSM1(void* args)
{
    agent = new SystemStateManager::GenericClone("genericClone");
    agent->initHierarchy();
    agent->initialize();

    // Register safety
    agent->registerSafetyModuleHandler(safetyModule);

    agent->registerFunctionHandler(PRE_INIT_STATE, SystemStateManager::SSMFunctionHandler(PRE_INIT_STATE_fun));
    agent->registerFunctionHandler(INIT_STATE, SystemStateManager::SSMFunctionHandler(INIT_STATE_fun));
    agent->registerFunctionHandler(POST_INIT_STATE, SystemStateManager::SSMFunctionHandler(POST_INIT_STATE_fun));
    agent->registerFunctionHandler(PRE_READY_STATE, SystemStateManager::SSMFunctionHandler(PRE_READY_STATE_fun));
    agent->registerFunctionHandler(READY_STATE, SystemStateManager::SSMFunctionHandler(READY_STATE_fun));
    agent->registerStdFunctionHandler(EXECUTE_RESET_HNDL_STR, SystemStateManager::SSMFunctionHandler(resetHandler_fun));
    agent->registerFunctionHandler("Standby", SystemStateManager::SSMFunctionHandler(standby_fun));
    agent->registerFunctionHandler("NormalOperation", SystemStateManager::SSMFunctionHandler(normalOperation_fun));
    agent->registerFunctionHandler("Degrade", SystemStateManager::SSMFunctionHandler(degrade_fun));
    agent->registerFunctionHandler("UrgentOperation", SystemStateManager::SSMFunctionHandler(urgentOperation_fun));
    agent->registerFunctionHandler("UrgentOperation2", SystemStateManager::SSMFunctionHandler(urgentOperation_fun));
    agent->registerFunctionHandler(SHUTDOWN_STATE, SystemStateManager::SSMFunctionHandler(SHUTDOWN_STATE_fun));
    agent->registerFunctionHandler(PRE_SHUTDOWN_STATE, SystemStateManager::SSMFunctionHandler(PRE_SHUTDOWN_STATE_fun));
    return NULL;
}

int main()
{
    pthread_t psm, psm1, psm2;
    //SystemStateManager::setupLogDebugLevel();
    SystemStateManager::setupLogFunction(printLog);
    //SystemStateManager::setupBufferCommunication();

    // Generic clone initHierarchy needs to be done in a 
    // seperate thread. It connects with SSM thread and
    // attempts to get hold of the port it needs to connect
    pthread_create(&psm2, NULL, &initSM1, NULL);

    // Wait for a few seconds and ensure that generic clone
    // connects with SSM
    sleep(3);

    ssm = new SystemStateManager::SM6::SSMClass();
    ssm->initHierarchy();
    ssm->initialize();

    pthread_join(psm2, NULL);

    while (1) {
        std::cout << "Waiting for clients to be ready..." << std::endl;
        if (ssm->areClientsReady()) {
            break;
        }
        sleep(1);
    }

    ssm->initClientComm();
    agent->initClientComm();

    pthread_create(&psm, NULL, &startSSM, NULL);
    pthread_create(&psm1, NULL, &startSM1, NULL);

    pthread_join(psm, NULL);
    pthread_join(psm1, NULL);
    delete ssm;
    delete agent;
    return (EXIT_SUCCESS);
}

