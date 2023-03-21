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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <ssm/SSMIncludes.hpp>
#include <ssm/commServer.hpp>
#include <ssm/commClient.hpp>
#include <ssm/SSMBuffer.hpp>
namespace SystemStateManager
{

class StateMachine;
typedef std::shared_ptr<StateMachine> StateMachinePtr;

typedef FixedVector<StateMachinePtr> StateMachineVector;
extern StateMachineVector stateMachineVector;

typedef struct _stateChildRelation {
    std::string state;
    StateMachinePtr stateMachinePtr;
} StateChildRelation;

typedef std::shared_ptr<StateChildRelation> StateChildRelationPtr;
typedef FixedVector<StateChildRelationPtr> StateChildRelationVector;

typedef struct _stateTransition {
    std::string from;
    std::string to;
} StateTransition;

typedef std::shared_ptr<StateTransition> StateTransitionPtr;
typedef FixedVector<StateTransitionPtr> StateTransitionVector;

class StateMachine {
public:
    StateMachine(std::string name_, std::string ip, int port, bool createHeadSMChannels = true) {
        this->name = name_;
        ipAddress = ip;
        listenPort = port;
        m_createHeadSMChannels = createHeadSMChannels;
    };

    ~StateMachine() {
        delete ssmServerThread;
        delete ssmServerCommThread;
        delete ssmLogCommThread;
        delete ssmClientPtr;
        delete ssmServerPtr;
        delete ssmCommandServerPtr;
        delete ssmLogServerPtr;
    }

    // Add states to the State machine framework
    void addStates(std::initializer_list<const char *> states);
    void addState(std::string state);
    void clearStates() {
        statesVector.clearVector();
    }
    void startServer();
    void stopServer();
    void setParentName(std::string pname) {
        parentName = pname;
    }

    void printHistogram();

    // API to add transition between two valid states
    bool addTransition(std::string from, std::string to);

    // Add a child state machine to a parent state machine
    bool addChild(std::string str, StateMachinePtr smPtr);

    // Define start state
    bool setStartState(std::string sstate);

    // Returns the first start state
    std::string getStartState() { return startState; };
    void overrideStateChangeLegalityCheck() {
        overrideStatechangeLegalityCheckFlag = true;
    }
    bool setupServer();
    void initClient();
    std::string getName() {return name;}
    std::string getParentName() {return parentName;}
    std::string getCurrentState();
    bool changeStateByString(std::string newState);
    void changeStateOverride(std::string newState);
    bool isChangeStateValid(std::string newState);
    void addClones(SWCVector &swcVector,
                   StateMachineVector &stateMachineVector,
                   int &startPort);

    void setIgnoreClonesFile(std::string path) {swcIgnorePath = path;};
    void setSWCClonesListFile(std::string path) {swcListFilePath = path;};
    inline SSMServer *getSSMServer() { return ssmServerPtr; } 
    inline bool readMessage(StateUpdate &su) {
        // TODO: Remove SSMServerPtr Jira: NVSTM-1298
        if (!isMessageBufferConfigured()) {
            if (ssmServerPtr) {
                return ssmServerPtr->readMessages(su);
            }
        } else {
            return serverBuffer->readMessages(this->name, su);
        }
        return false;
    }

    inline void setSWCCountToInit(int swcInitCount) {
        // TODO: Remove SSMServerPtr Jira: NVSTM-1298
        if (!isMessageBufferConfigured()) {
            if (ssmServerPtr) {
                return ssmServerPtr->setSWCCountToInit(swcInitCount);
            }
        }
    }

    inline bool getUserSpecificDataInternal(UserDataPktInternal &pkt) {
        // TODO: Remove SSMServerPtr Jira: NVSTM-1298
        if (!isMessageBufferConfigured()) {
            if (ssmServerPtr) {
                return ssmServerPtr->readUserData(pkt);
            }
        } else {
            return serverBuffer->readUserData(this->name + SSM_DATA_CHANNEL, pkt);
        }
        return false;
    }

    inline bool getRouteSpecificData(UserDataPktInternal &pkt) {
        if (ssmServerPtr) {
            return ssmServerPtr->readHeadRouteData(pkt);
        }
        return false;
    }

    inline bool readCommandServer(StateUpdate &su) {
        // TODO: Remove SSMServerPtr Jira: NVSTM-1298
        if (!isMessageBufferConfigured()) {
            if (ssmCommandServerPtr) {
                return ssmCommandServerPtr->readMessages(su);
            }
        } else {
            return serverBuffer->readMessages(SSM_CMD_CHANNEL, su);
        }
        return false;
    }

    void waitForInitialization() {
        if (ssmServerPtr) {
            ssmServerPtr->waitForInitialization();
        }
    }

    void deactivateChildren(std::string state);
    void activateChildren(std::string state);
    void startHeadSMServers();
    void overRideInitPhase();
    void changeInitPhase(std::string from, int initPhase);

    void activateClient();

    inline bool checkIfClientIsReady() {
        if (!isMessageBufferConfigured()) {
            if (ssmClientPtr) {
                return ssmClientPtr->checkIfClientIsReady();
            }
        } else {
            return clientBuffer->isSMReady(this->name);
        }
        return false;
    }

    /// Broadcast APIs
    void broadcastLockSteppedCommand(std::string sender, int cmd, int lockCmd);
    void broadcastCommand(std::string sender, int command, std::string extraInfo="");
    void broadCastStateChange(std::string stateMachineName,
                                         std::string newState,
                                         bool isPayloadPresent = false,
                                         int value = 0);
    void broadCastActivateStateMachine(std::string sender, bool activate);
    void notifyStateMachine(std::string name, long notID, int arg);
    void executedInitPhase(std::string name, std::string initPhase);
    void executedPreShutdownPhase(std::string name);
    void resetState();
    void updateInitFlag(int initPhase);
    bool getPreShutdownPhaseFlag() { return preShutdownComplete; }
    void setPreShutdownFlag() {preShutdownComplete = true;}
    bool getInitPhaseFlag(int initPhase);
    bool isSMReadyForScheduler () { return startSchedular; }
    void setStartSchedular() { startSchedular = true;}
    void finalizeStates();
    std::string getIPAddress() {return ipAddress;}
    int getPort() {return listenPort;}
    void printChildren();
    StateMachineVector *getChildrenVector() { return &childSMVector; }
    bool sendData(UserDataPktInternal &pkt);
    bool sendData(void *ptr,
                  int size,
                  const void *extraPayload,
                  int sizeOfExtraPayload);

    void setGlobalAwareness() { isGlobalStateAware = true; }
    bool isSMGloballyStateAware() {return isGlobalStateAware; }

    void setStateGloballyRelevant() {
        isStateGloballyRelevant = true;
    }
    bool isSMStateGloballyRelevant() { return isStateGloballyRelevant; }

    void getCloneNames(StringVector &vector) {
        for (auto &swc : swcCloneVector.getVector()) {
            vector.push_back(swc.name);
        }
    }

    void getStatesList(CharVector &vector) {
        for (auto &nm : statesVector.getVector()) {
            CharStruct cst;
            memset(cst.name, '\0', MAX_SIZE_GENERAL_STRINGS);
            strncpy(cst.name, nm.c_str(), MAX_SIZE_GENERAL_STRINGS-1);
            vector.push_back(cst);
        }
    }

    void getStatesList(StringVector &vector) {
        for (auto &nm : statesVector.getVector()) {
            vector.push_back(nm);
        }
    }

    void setDisabledByDefaultFlag() {
        this->isDisabledByDefault = true;
    }

    bool getDisabledByDefaultFlag() {
        return this->isDisabledByDefault;
    }

    void logStateMachineSpecs() {
        StringVector cloneVec;
        StringVector statesVec;
        getCloneNames(cloneVec);
        getStatesList(statesVec);
        //logStringVector(cloneVec, getName() + "'s Clones");
        //logStringVector(statesVec, getName() + "'s Global States");
    }

    StringVector getIgnoreStateMachineList() { return ignoreStateMachineList; }
    bool getCreateHeadSMChannels() { return m_createHeadSMChannels; }
    void doNotDeactivate() {
        isSMDeactivatableFlag = false;
    }
    bool isSMDeactivatable() {
        return isSMDeactivatableFlag;
    }

    // Set the id BEFORE starting the servers
    void setSSMBufferInstanceID(int id) {
        ssmBufferInstanceID = id;
        if (isMessageBufferConfigured()) {
            clientBuffer = SSMBuffer::getInstance(ssmBufferInstanceID);
        }
    }

private:
    std::string name;
    std::string ipAddress {"localhost"};
    int listenPort {0};
    std::string swcIgnorePath;
    std::string swcListFilePath;
    std::string parentName;
    bool isServerPresent {false};
    StringVector statesVector;
    StateMachineVector childSMVector;
    SWCVector swcCloneVector;
    StateTransitionVector stateTransitionVec;
    StateChildRelationVector stateChildRelationVector;
    std::string startState;
    StringVector ignoreStateMachineList;


    // This is used for logging the specs of each state machine once during startup
    static StringSet stateMachineLogSet;
    static std::mutex stateMachineLogSetLock;

    bool m_createHeadSMChannels {true};
    SSMClient *ssmClientPtr{};
    SSMServer *ssmServerPtr{};
    SSMServer *ssmCommandServerPtr{};
    SSMServer *ssmLogServerPtr{};
    SSMServerThread *ssmServerThread{};
    SSMServerThread *ssmServerCommThread{};
    SSMServerThread *ssmLogCommThread{};
    int ssmBufferInstanceID {0};

    // This pointer points to an object ONLY if it supports a server
    // else it is NULL
    SSMBuffer *serverBuffer {};

    // Points to the buffer through which messages can be sent to the server
    SSMBuffer *clientBuffer {};

    bool overrideStatechangeLegalityCheckFlag {false};
    bool isSMDeactivatableFlag {true};

    bool preShutdownComplete{false};
    bool preInitComplete{false};
    bool initComplete{false};
    bool postInitComplete{false};
    bool preReadyComplete{false};
    bool readyComplete{false};
    bool startSchedular{false};
    bool isGlobalStateAware {false};
    bool isStateGloballyRelevant {false};
    bool isDisabledByDefault {false};

    std::string currentState{PRE_INIT_STATE};
    std::mutex stateChangeMutex;
};

extern StateMachinePtr getStateMachinePointer(StateMachineVector &stateMachineVector, std::string smName);

}
