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
// SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// Parser Version: 0.7.1
// SSM Version:    0.8.2
//
/////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ssm/SMBaseClass.hpp>
#include <ssm/SSMHistogram.hpp>

namespace SystemStateManager
{

class GenericClone : public SMBaseClass
{
public:
    ~GenericClone(){};
    GenericClone(std::string myname)
    {
        name = myname;
        runnableHist.setName(name);
    }
    virtual void initStateMachine() {};
    void executeStateMachine() override;

    bool registerFunctionHandler(std::string name, SSMFunctionHandler ptr);
    void runInitPhase(int micros);
    bool sendCloneCommand(int cmd);
    void printClientHistogram() override;
    void setRunnableId(const char* runnableId) {
        m_runnableId = runnableId;
    }

    void registerSafetyModuleHandler(SSMSafetyModuleFunctionPtr ptr) {
        safetyModuleHandler = ptr;
    }

    void handleSetGobalStatesMessage(StateUpdate &su) override;

    void externalStateChangeEventHandler(StateMachinePtr ptr,
                                         std::string fromState) override;
    void executeSpecialCommands(StateUpdate &su) override;
    void globalStateChangeComplete(uint64_t transitionID, int error=0) {
        static_cast<void>(transitionID);
        static_cast<void>(error);
        sendCloneCommand(SAFE_STATE_CHANGE_COMPLETE);
    }

    void initClientComm() {
        SMBaseClass::initClientComm();
        sendCloneCommand(GET_GLOBAL_STATES);
        mySMPtr->clearStates();
        if (safetyModuleHandler != NULL) {
            sendCloneCommand(REGISTER_SAFETY_MODULE);
        }
    }

    const std::string& getRunnableId()
    {
        return m_runnableId;
    }

    // API to register LockedStepCommands
    bool registerLockedCommand(int command, SSMFunctionHandler ptr);
    bool setupHierarchy(StateMachineVector &stateMachineVector, StateMachinePtr &headPtr);

private:
    SSMHistogram runnableHist{};
    int totalFuncs{0};
    SSMSafetyModuleFunctionPtr safetyModuleHandler {};
    std::string m_runnableId;
    SSMFunctionHandlerVector funcVector;
    SSMFunctionHandler enterHandlerPtr{};
    SSMFunctionHandler exitHandlerPtr{};
    SSMFunctionHandler pre_initHandlerPtr{};
    SSMFunctionHandler initHandlerPtr{};
    SSMFunctionHandler postHandlerPtr{};
    SSMFunctionHandler preReadyHandlerPtr{};
    SSMFunctionHandler readyHandlerPtr{};
    SSMFunctionHandler preSwitchHandlerPtr{};
    SSMFunctionHandler postSwitchHandlerPtr{};
    bool setupTreeHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr) override;
    void generateLockedSteppedFunctions();
};

}
