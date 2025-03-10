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
// Parser Version: 0.7.5
// SSM Version:    0.9.1
//
/////////////////////////////////////////////////////////////////////////////////////////

/**
* The following file has been generated by SSM's parser.
* Please do not manually modify the files
*/

#include <ssm/StateMachine.hpp>

namespace SystemStateManager
{

namespace SM6
{

static std::atomic<bool> showHeadSpecs {true};

static std::mutex showHeadSpecsLock;

bool setupHierarchy(StateMachineVector &stateMachineVector, StateMachinePtr &headPtr)
{
    int smStartPort = 0;
    int cloneStartPort = 0;
    SWCVector swcVector;
    swcVector.push_back({"genericClone", "127.0.0.1"});

    std::string startPort = getKeyFromMasterQAFile("startport");
    if (startPort != "")
    {
        overrideBasePort = std::stoi(startPort);
        setPorts();
    }
    std::string remoteSSMIP = getKeyFromMasterQAFile("remoteSSMIP");
    if (remoteSSMIP != "")
    {
        ssmMasterIPAddr = remoteSSMIP;
    }

    smStartPort = startSocketPort;
    cloneStartPort = smStartPort + MAX_SMs_ALLOWED;
    StateMachinePtr SSM_sm = StateMachinePtr(new StateMachine("SSM", ssmMasterIPAddr.c_str(), smStartPort++, true));
    SSM_sm->addStates({"Degrade", "NormalOperation", "Standby", "UrgentOperation"});
    SSM_sm->finalizeStates();

    if (!SSM_sm->addTransition("Degrade", "UrgentOperation")) return false;
    if (!SSM_sm->addTransition("NormalOperation", "Degrade")) return false;
    if (!SSM_sm->addTransition("Standby", "NormalOperation")) return false;
    if (!SSM_sm->addTransition("UrgentOperation", "Standby")) return false;
    if (!SSM_sm->setStartState("Standby")) return false;

    stateMachineVector.push_back(SSM_sm);
    SSM_sm->addClones(swcVector, stateMachineVector, cloneStartPort);

    headPtr = SSM_sm;

    ////////// CRITICAL SECTION //////////////
    SSMLock lg(showHeadSpecsLock);
    if (showHeadSpecs) {
        showHeadSpecs = false;
        SSM_LOG("Setup Hierarchy for SM6");
        headPtr->logStateMachineSpecs();
    }
    /////////////////////////////////////////
    finalMaxPort = smStartPort > finalMaxPort ? smStartPort : finalMaxPort;
    finalMaxPort = cloneStartPort > finalMaxPort ? cloneStartPort : finalMaxPort;
    return true;
}}
}
