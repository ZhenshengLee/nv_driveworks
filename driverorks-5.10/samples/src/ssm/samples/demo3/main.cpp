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
#include <SSMClass.hpp>
#include <SM1Class.hpp>
#include <SM2Class.hpp>

#define MARKER "======"
#define MAX_CYCLES 20
typedef std::atomic<int> AtomicInt;
AtomicInt count{0};
SystemStateManager::SM3::SSMClass* ssm{};
SystemStateManager::SM3::SM1Class* sm1{};
SystemStateManager::SM3::SM2Class* sm2{};
using std::cout;
using std::endl;
std::vector<std::string> logVec;

std::mutex cmdLock;
SystemStateManager::StringVector ssmVec;
SystemStateManager::StringVector sm1Vec;
SystemStateManager::StringVector sm2Vec;

void addL(std::string cmd, std::string name)
{
    ////////// Critical Section ////////////
    SystemStateManager::SSMLock lg(cmdLock);
    if (name == "SSM") {
        ssmVec.push_back(cmd + "_" + name);
    } else if (name == "SM1") {
        sm1Vec.push_back(cmd + "_" + name);
    } else if (name == "SM2") {
        sm2Vec.push_back(cmd + "_" + name);
    }
    ////////////////////////////////////////
}

void fn_CMD1(SystemStateManager::SMBaseClass *ptr) { addL("CMD1", ptr->getName());}
void fn_CMD2(SystemStateManager::SMBaseClass *ptr) { addL("CMD2", ptr->getName());}
void fn_CMD3(SystemStateManager::SMBaseClass *ptr) { addL("CMD3", ptr->getName());}
void fn_CMD4(SystemStateManager::SMBaseClass *ptr) { addL("CMD4", ptr->getName());}

void insertMarker()
{
    ssmVec.push_back(MARKER);
    sm1Vec.push_back(MARKER);
    sm2Vec.push_back(MARKER);
}

void printVector(SystemStateManager::StringVector &vec)
{
    for (auto &s : vec.getVector()) {
        cout << s << endl;
    }
}

void* startSSM(void* args)
{
    ssm->startClients();

    // Lock Stepped commands can be executed before starting clients
    while(!ssm->executeLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd2)) {
        ssm->runStateMachine();
        sleep(1);
    }
    insertMarker();


    // ======== Now run init phase ========== //
    ssm->executeInitPhase(PRE_INIT_PHASE);
    ssm->executeInitPhase(INIT_PHASE);

    // Lock Stepped commands can be executed during initialization
    while(!ssm->executeLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd1)) {
        ssm->runStateMachine();
        sleep(1);
    }
    insertMarker();
    ssm->executeInitPhase(POST_INIT_PHASE);
    ssm->executeInitPhase(PRE_READY_PHASE);
    ssm->executeInitPhase(READY_PHASE);
    ssm->changeInitPhase(ENTER_STATE_MACHINE);
    // ======================================== //

    // Lock Stepped commands can be executed before entering the first state
    while(!ssm->executeLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd3)) {
        ssm->runStateMachine();
        sleep(1);
    }
    insertMarker();

    int counter = 0;
    while (counter < MAX_CYCLES)
    {
        if (counter == 10) {
            // Lock Stepped commands can be executed during normal operation
            while(!ssm->executeLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd4)) {
               ssm->runStateMachine();
                sleep(1);
            }
            insertMarker();
        }
        ssm->runStateMachine();
        usleep(1100000);
        counter++;
    }

    while(!ssm->executeLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd2)) {
        ssm->runStateMachine();
        sleep(1);
    }

    return NULL;
}

void* startSM(void* args)
{
    int counter = 0;
    while (counter < MAX_CYCLES)
    {
        sm1->runStateMachine();
        sm2->runStateMachine();
        usleep(1200000);
        counter++;
    }
    return NULL;
}

void printLog(SystemStateManager::RCID logType, std::string logTag, void *buffer, unsigned int bufferSize)
{
    if (logType == SystemStateManager::RCID::SSM_INTERNAL) {
        std::string logStr = std::string((char *)buffer, bufferSize);
        logVec.push_back(logStr.substr(0, logStr.find("[")));
    }
}

int main()
{
    pthread_t pssm, psm;

    // Setup RoadCast upfront
    SystemStateManager::setupCentralizedLogger(printLog);
    SystemStateManager::setupLogPerfLevel();
    SystemStateManager::setupBufferCommunication();
    ssm = new SystemStateManager::SM3::SSMClass();
    ssm->initHierarchy();
    ssm->initialize();

    sm1 = new SystemStateManager::SM3::SM1Class();
    sm1->initHierarchy();

    sm2 = new SystemStateManager::SM3::SM2Class();
    sm2->initHierarchy();

    ssm->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd1, fn_CMD1);

    ssm->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd2, fn_CMD2);
    sm2->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd2, fn_CMD2);

    ssm->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd3, fn_CMD3);
    sm1->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd3, fn_CMD3);

    sm1->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd4, fn_CMD4);
    sm2->registerLockedCommand(SystemStateManager::SM3::LockSteppedCommands::cmd4, fn_CMD4);

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

    pthread_create(&psm, NULL, &startSM, NULL);
    usleep(600000);
    pthread_create(&pssm, NULL, &startSSM, NULL);

    pthread_join(pssm, NULL);
    pthread_join(psm, NULL);
    delete ssm;
    delete sm1;
    delete sm2;

    // Check the order in which the lock stepped commands have executedc
    printVector(ssmVec);
    cout << "###############" << endl;
    printVector(sm1Vec);
    cout << "###############" << endl;
    printVector(sm2Vec);
    cout << "###############" << endl;

    // Print log vector
    for (auto &a : logVec) {
        cout << a << endl;
    }

    return (EXIT_SUCCESS);
}
