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

using std::cout;
using std::endl;

typedef std::atomic<int> AtomicInt;
AtomicInt count{0};
SystemStateManager::SM1::SSMClass* ssm{};

void printLogMessage(std::string logStr)
{
    std::cout << logStr << std::endl;
}

void* startSSM(void* args)
{
    while (1)
    {
        ssm->runStateMachine();
        if (count == 6) {
            int error=system("echo \"process external cmd 0x2222\" | nc localhost 6069");
            if (error < 0) {
                printLogMessage("Error while sending cmd");
            }
        }

        if (count == 10) {
            ssm->executeExternalQACommands("process internal cmd 0x11111");
        }

        if (ssm->getCurrentState() == "UrgentOperation") {
            break;
        }
        usleep(1100000);
        count ++;
    }
    return NULL;
}

void printLog(SystemStateManager::SSM_LOG_TYPE slt, std::string timestr, std::string logMsg)
{
    cout << timestr << " " << logMsg << endl;
}
void qaFunctionHandler(SystemStateManager::SMBaseClass *ssmPtr, SystemStateManager::CharVector &argVector)
{
    cout << "Processing command: " << endl;
    for (auto &a : argVector.getVector()) {
        cout << a.name << endl;
    }
}

int main()
{
    pthread_t psm;
    SystemStateManager::setupLogFunction(printLog);
    ssm = new SystemStateManager::SM1::SSMClass();
    ssm->initHierarchy();
    ssm->registerQAFunctionHandler(qaFunctionHandler);
    ssm->initialize();
    ssm->initClientComm();
    ssm->changeInitPhase(PRE_INIT_PHASE);
    ssm->changeInitPhase(INIT_PHASE);
    ssm->changeInitPhase(POST_INIT_PHASE);
    ssm->changeInitPhase(PRE_READY_PHASE);
    ssm->changeInitPhase(READY_PHASE);
    ssm->changeInitPhase(ENTER_STATE_MACHINE);
    pthread_create(&psm, NULL, &startSSM, NULL);
    pthread_join(psm, NULL);
    delete ssm;
    return (EXIT_SUCCESS);
}
