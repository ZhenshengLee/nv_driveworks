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

#include <ssm/QAServer.hpp>
#include <ssm/StateMachine.hpp>

namespace SystemStateManager
{

extern SSMCentralizedLogFunctionPtr centralizedLogFunPtr;
extern void shutdownCentralizedLogger();
extern bool startCentralizedLogger();
extern void startExternalLoggerClient(std::string name);

class SMBaseClass;
typedef std::vector<SMBaseClass *> FixedSMVector;

typedef FixedMap<std::string, StateMachinePtr> StateMachineMap;

typedef struct _childPayload {
    std::string childName;
    int arg;
    bool sendPayload;
} ChildPayload;

typedef struct _notification {
    long notID;
    int arg;
    std::string receivedFrom;
    bool isReady;
} Notification;

typedef struct _safetyModuleSpecs {
    std::string stateMachineName {};
    StateMachinePtr safetySMPtr {};
    bool isStateChangeInitiated {};
    uint64_t stateChangeTimeoutUS {};
    uint64_t stateChangeInitiatedAt {};
    std::string fromState {};
    std::string toState {};
} SafetyModuleSpecs;
typedef std::shared_ptr<SafetyModuleSpecs> SafetyModulePtr;

typedef struct deferredStruct {
    SSM_DEFERRED_INIT status {SSM_DEFERRED_INIT::FREE_BUCKET};
    StateMachineMap deferredMap {};
    SSMFunctionHandler pre_init_handler {};
    SSMFunctionHandler init_handler {};
    SSMFunctionHandler post_init_handler {};
    SSMFunctionHandler pre_ready_handler {};
    SSMFunctionHandler ready_handler {};
} DeferredStruct;

typedef FixedVector<DeferredStruct> DeferredVector;

typedef FixedVector<Notification> NotificationVector;

class SMBaseClass {

public:
    virtual ~SMBaseClass() {
        stopServer();
        shutdownCentralizedLogger();
        delete ssmChannelClientPtr;
        delete ssmLogClientPtr;
        pthread_cancel(qassm);
    };

    SMBaseClass();

    // todo: ptumati to move this to generic clone once ssmclone is deprecated
    void overrideStateChangeLegalityCheck()
    {
        if (headPtr && mySMPtr)
        {
            headPtr->overrideStateChangeLegalityCheck();
            mySMPtr->overrideStateChangeLegalityCheck();
        }
    }

    bool changeStateByString(std::string stateName);
    bool resetSSM();
    bool changeStateWithPayload(std::string stateName, std::string childName, int value);
    virtual void initialize() {};
    bool initHierarchy();
    void runStateMachine();
    int sendData(std::string stateName, const void *data, int datalen, std::string dataStructID);
    virtual void ENTER() {}
    virtual void EXIT() {}
    virtual void PRE_INIT() {}
    virtual void INIT() {}
    virtual void POST_INIT() {}
    virtual void preSwitchHandler() {}
    virtual void postSwitchHandler() {}
    virtual void printClientHistogram() {}
    virtual bool setupTreeHierarchy(StateMachineVector &stateMachineVector_, StateMachinePtr &headPtr_) = 0;
    virtual void handleSetGobalStatesMessage(__attribute__((unused)) StateUpdate &su) {};
    virtual void preStateChangeHandler( __attribute__((unused))std::string currentState,
                                        __attribute__((unused))std::string toState) {};
    virtual void postStateChangeHandler(__attribute__((unused))std::string prevState,
                                        __attribute__((unused))std::string currentState) {};
    virtual void externalStateChangeEventHandler(__attribute__((unused))StateMachinePtr ptr,
                                                 __attribute__((unused))std::string fromState) {};
    virtual void executeSpecialCommands(__attribute__((unused))StateUpdate &su) {};

    // Helper function that can document state changes to an external logger
    virtual void documentStateChangeExternally() {};

    // Helper function that initializes all the state machines
    // along with the head state machine
    bool initializeEntireHierarchy();

    // This API initializes the entire hierarchy along with
    // the state machines that are not specified in the arguments
    template <typename... Rest>
    bool initializeEntireHierarchy(SMBaseClass &first, Rest&&... args) {
        FixedSMVector smVector;
        this->initHierarchy();
        if (this->mySMPtr != this->headPtr) {
            return false;
        }
        this->initialize();
        smVector.push_back(&first);
        return initializeSubHierarchyInternal(smVector, args...);
    }

    // This API isolates head state machine from all its children
    // and its just the head state machine
    bool initializeSubHierarchy();

    // This API initializes only a portion of the hierarchy
    // input arguments specify the list of state machines it should isolate
    template <typename... Rest>
    bool initializeSubHierarchy(SMBaseClass &first, Rest&&... args) {
        FixedSMVector smVector;
        this->initHierarchy();
        if (this->mySMPtr != this->headPtr) {
            return false;
        }
        this->initialize();
        this->setPrimaryInitList();
        StringVector st;
        st.push_back(first.getName());
        addToPrimaryInitList(st);
        smVector.push_back(&first);
        return initializeSubHierarchyInternal(smVector, args...);
    }

    bool generatePerfHistogram();
    void initClientComm() {
        for (auto ptr : this->stateMachineVector.getVector()) {
            initClientComm(ptr);
        }
        if (broadCastCentralizedLoggerPresent) {
            headPtr->broadcastCommand(name, EXTERNAL_LOGGER_PRESENT, SSM_LOCAL_IP);
        }
    };
    bool areClientsReady();
    void sendMessageInSSMChannel(int cmd);
    void logStateChange(std::string toState);
    void logInitComplete(std::string toState);
    void startClients() {
        SSM_PERF("SSMStart: ["+std::to_string(getMilliseconds()) + "]");
        for (auto ptr : this->stateMachineVector.getVector()) {
            startClients(ptr);
        }
        broadcastCommandToAllSMs(CLIENTS_ARE_CONNECTED);
    }
    bool isPreShutdownPhaseComplete();

    bool overRideInitPhase(std::string swname) {
        auto sptr = masterStateMachineMap.find(swname);
        if (sptr == masterStateMachineMap.end()) {
            SSM_ERROR("Invalid State machine: " + swname);
            return false;
        }

        masterStateMachineMap[swname]->overRideInitPhase();
        return false;
    }

    bool overRideInitPhase() {
        if (mySMPtr) {
            mySMPtr->overRideInitPhase();
            return true;
        }
        return false;
    }

    //This function is used to connect a client state machine to SSM initially.
    //The function tries to recieve a message from the head state machine which confirms its connectivity
    void waitForConnectionToSsm(int sleepTIme);

    bool executeInitPhase(int initPhase);
    void changeInitPhase(int initPhase);

    bool isStartStateMachineMsgReceived();
    bool isSMReadyForScheduler();
    void shutdown();
    void preShutdown();
    bool isReadyForShutdown() { return shutdownStateMachine;}
    bool isReadyForPreShutdown() { return preShutdownStateMachine;}
    bool getCommandFromSSMChannel(StateUpdate &su);
    void startServer() { mySMPtr->startServer(); }
    void stopServer();
    bool isSMActive() { return isStateMachineActive; }
    std::string getName() { return this->name; }
    std::string getCurrentState() { return mySMPtr->getCurrentState(); }
    std::string getCurrentState(std::string targetStateMachine);
    uint64_t getStateTransitionID() { return transitionID; }
    bool clearNotificationVector();
    std::string getParentName() {return mySMPtr->getParentName();}

    // QA apis
    void setQAResetFlag() { qaResetFlag = true; }

    /**
     * Make head SM execute adhoc QA commands
     * Calling this function triggers a function call
     * to the function handler that is registed through
     * 'registerQAFunctionHandler(...)'
     *
     * @param QA instruction with relevant details
     * @return true if accepted; false if not.
     */
    bool executeExternalQACommands(std::string message);

    // Allows users to register a function handler that gets called
    // whenever the state machine receives an external QA command
    bool registerQAFunctionHandler(SSMQAFunctionHandler ptr);

    bool registerStdFunctionHandler(std::string name,
                                    SSMFunctionHandler ptr);
    bool getUserSpecificDataInternal(UserDataPktInternal &pkt) {
        return mySMPtr->getUserSpecificDataInternal(pkt);
    }

    // List of APIs that assist developers in designing inititalized priority lists
    // Returns the names of state machines present in the master vector
    void getPrimaryInitList(StringVector &sv);      // Returns the primary list (except head state machine)
    void setPrimaryInitList();                      // Clear the primary list
    void setPrimaryInitList(StringVector &sv);      // Fill sthe primary list (except head state machine)
    void setPrimaryInitListToDefaults();            // Resets the primary list to the default post initialization
    bool addToPrimaryInitList(StringVector &sv);    // Adds SMs to the primary list
    void removeFromPrimaryInitList(StringVector &sv);
    bool initializeDeferredList(StringVector &sv,
                                SSMFunctionHandler pre_init_handler = nullptr,
                                SSMFunctionHandler init_handler = nullptr,
                                SSMFunctionHandler post_init_handler = nullptr);

    // Helper function that returns the list of all the clones
    void getCloneNames(StringVector &vector) {
        mySMPtr->getCloneNames(vector);
    }

    // Helper function that returns the list of all the states within the SM
    void getStateNames(CharVector &vector) {
        mySMPtr->getStatesList(vector);
    }

    // Helper function that returns the list of global states
    void getGlobalStates(CharVector &vector) {
        headPtr->getStatesList(vector);
    }

    // Helper function that returns the list of global states
    void getGlobalStates(StringVector &vector) {
        headPtr->getStatesList(vector);
    }

    bool isSMPresentInPrimaryList(std::string swname) {
        for (auto &xptr : stateMachineVector.getVector()) {
            if (xptr->getName() == swname) {
                return true;
            }
        }
        return false;
    }

    // This needs to be accessed from gtests where parallelism
    // is not acceptable
    bool isInitPhaseComplete(int initPhase);

    // Returns the port that the statemachine listens to
    int getServerPort(std::string stateMachineName);
    void doNotDeactivate() {
        mySMPtr->doNotDeactivate();
    };

    // Use to debug multi hierarchy scenarios
    int getSSMBufferInstanceID() {
        return ssmBufferInstanceID;
    }

protected:
    void initClientComm(StateMachinePtr ptr);
    void startClients(StateMachinePtr ptr);
    virtual void executeStateMachine() {};

    void broadCastPrintHistogram();
    void broadCastStateChange(std::string stateMachineName,
                              std::string newState);

    void executeStateChange(std::string &toState);

    void processMessageQueue();
    void deactivateChildren(StateMachinePtr smptr);
    void activateChildren(StateMachinePtr smptr);
    void sendNotification(std::string target, int arg);
    void updateChildInitPhase(std::string name, int initPhase);
    void updateChildPreShutdownPhaseAsComplete(std::string name_);
    void notifyParent(int arg) {
        sendNotification(mySMPtr->getParentName(), arg);
    }
    bool getNextNotification(Notification &notification);
    void clearLogMsgQueue();
    bool clearLogMessages();  // Only used in SSMBuffer case
    bool clearPerfMessages(); // Only used in SSMBuffer case
    void printHistogram();
    void reconnectClients();

    // This function initialized the infrastructure required to execute
    // the lock stepped commands
    void addLockSteppedCommand(int command);
    bool executeLockSteppedCommand(int cmdIndex);
    bool registerLockSteppedFunction(int cmdIndex, SSMFunctionHandler ptr);

    void broadcastCommandToNearestNeighbors(int cmd);
    void broadcastCommandToChildren(int command, std::string extraInfo="");
    void broadcastCommandToAllSMs(int command);
    void processDeferredInializations();

    ///// The following function MUST runs inside critical section [deferredLock] ///
    bool getDeferredInitList(std::mutex &lock, int index, StringVector &sv);
    bool areDeferredClientsReady(std::mutex &lock, int index);
    bool isDeferredInitPhaseComplete(std::mutex &lock, int index, int initPhase);
    bool changeInitPhaseForDeferred(std::mutex &lock, int index, int initPhase);
    bool processDeferredInializationList(std::mutex &lock, int index);
    ////////////////////////////////////////////////////////////////////////////////

private:
    // Private Helper function that initializes all the state machines
    // included in the vector
    bool initializeSubHierarchyInternal(FixedSMVector &smVector);

    // Private Helper function that initializes all the state machines
    // included in the vector and the variadic arguments
    template <typename... Rest>
    bool initializeSubHierarchyInternal(FixedSMVector &smVector, SMBaseClass &first, Rest&&... args) {
        StringVector st;
        st.push_back(first.getName());
        smVector.push_back(&first);
        addToPrimaryInitList(st);
        return initializeSubHierarchyInternal(smVector, args...);
    }

    bool executeLockSteppedFunctions();

    int ssmBufferInstanceID {0};

protected:
    std::string name {};
    StateMachinePtr headPtr {};
    StateMachinePtr mySMPtr {};
    pthread_t qassm {};

    bool broadCastCentralizedLoggerPresent {false};
    long notificationID {0};
    bool qaResetFlag {false};
    bool startQAServerFlag {true};
    bool resetSSMFlag {false};
    bool executeResetHandlerFlag {false};
    bool shutdownStateMachine {false};
    bool preShutdownStateMachine {false};
    bool isStateMachineActive {false};
    bool startStateMachineMsgReceived {false};
    bool isFunctionHandlerUpdated {false};
    bool preInitExecuted {false};
    bool initExecuted {false};
    bool postInitExecuted{false};
    bool preReadyExecuted{false};
    bool readyExecuted{false};
    bool enterStateMachineFlag {false};
    bool preShutdownExecuted {false};
    AtomicUInt transitionID {0};

    SSMFunctionHandler shutdownHandlerPtr {};
    SSMFunctionHandler preShutdownHandlerPtr {};
    SSMFunctionHandler resetFunctionHandlerPtr {};
    LockSteppedCmdRecordVector lockSteppedCmdRecordVector;
    SSMFunctionHandlerVector lockedSteppedCmdFunVector {};
    SSMClient *ssmChannelClientPtr {};
    SSMClient *ssmLogClientPtr {};
    ChildPayload childPayload {};
    SSMHistogram runStateMachineOverhead {};
    std::string newState {""};
    NotificationVector notificationVector {};
    std::queue<StateUpdate> logMsgQueue {};
    StateMachineVector globalStateAwareSMVector {};

    // This pointer points to an object ONLY if it supports a server
    // else it is NULL
    SSMBuffer *serverBuffer {};

    // Points to the buffer through which messages can be sent to the server
    SSMBuffer *clientBuffer {};

    // Points to the safetyModule if present
    SafetyModulePtr safetyModulePtr {};

    // List of important state machine lists
    StateMachineVector *childSMVector {};
    StateMachineVector stateMachineVector {};
    StateMachineMap masterStateMachineMap {};

    AtomicFlag qaCommandsPresent {false};
    std::mutex qaCommandThreadLock {};
    CharVector qaCMDLineCurlTokens {};
    SSMQAFunctionHandler qaFunctionHandler {};

    std::mutex deferredLock;
    // TODO: Revisit this issue when https://jirasw.nvidia.com/browse/NVSTM-1057
    //
    // Replacing the Array with DeferredVector is create memory related issues
    // std::map uses the standard memory allocator where as FixedVector uses MyAllocator
    // Both seem to be conflicting. Valgrind complains of memory corruption issues.
    // DeferredVector deferredVector;
    DeferredStruct deferredVector[TOTAL_ALLOWED_DEFERRED_LISTS];
};

}
