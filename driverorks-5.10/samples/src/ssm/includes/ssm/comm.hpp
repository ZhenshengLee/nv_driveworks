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

#include <cstddef>
#include <iostream>
#include <fcntl.h>
#include <mqueue.h>
#include <queue>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include <functional>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fstream>
#include <sstream>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <poll.h>
#include <map>
#include <set>
#include <sstream>
#include <arpa/inet.h>

namespace SystemStateManager
{

using std::cout;
using std::endl;

#define SSM_LATENCY_THRESHOLD           5
#define MAX_INIT_RESENT_COUNT           50
#define MAX_MESSAGES_ALLOWED            10
#define MAX_MESSAGE_SIZE                1024
#define MAX_DATA_PKG_SIZE               2048
#define MAX_ALLOWED_MSG_QUEUE_SIZE      64
#define MAX_SSM_BUFFERS_ALLOWED         16

#define FRAGMENTION_LIMIT               10
#define MAX_CONNECTIONS_ALLOWED         FD_SETSIZE
#define MAX_QA_CONNECTIONS_ALLOWED      1
#define INVALID_SOCKET_FD               -1
#define MAX_HISTOGRAM_SIZE              MAX_MESSAGE_SIZE
#define MAX_ANALYSIS_SIZE               64
#define MAX_RECONNECT_ATTEMPTS          3
#define BASE_TCP_PORT                   6069
#define MAX_STATE_MACHINE_SIZE          64
#define MAX_STATE_NAME                  64
#define MILLISECONDS_IN_NANOS           1000000
#define MAX_WAIT_TIME_MS                50
#define ONE_MILL_IN_MICROS              1000
#define TEN_MILLS_IN_MICROS             10000
#define SECOND_IN_MICROS                1000000

// TODO: Put checks in the code to disallow child state machines
//       beyond this number
#define MAX_SMs_ALLOWED                 50
#define BILLION                         1000000000

#define MSG_STOP                        "exit"
#define SSM_CHANNEL                     "SSM"
#define SSM_CMD_CHANNEL                 "SSM_CMD"
#define SSM_PERF_CHANNEL                "SSM_PERF"
#define SSM_LOG_CHANNEL                 "SSM_LOG"
#define SSM_DATA_CHANNEL                "_DATA"
#define SSM_CENTRAL_LOG_CHANNEL         "EXTERNAL_LOG"
#define SSM_LOCAL_IP                    "127.0.0.1"
#define SSM_EXTERNAL_LOGGER_IP          SSM_LOCAL_IP

//// Commands that SSM sends to SSMClients
#define STATE_CHANGE_CMD                 1
#define ACTIVATE_CMD                     2
#define DEACTIVATE_CMD                   3
#define NOTIFICATION_CMD                 4
#define START_STATE_MACHINE              5
#define PRE_INIT_PHASE                   6
#define INIT_PHASE                       7
#define POST_INIT_PHASE                  8
#define PRE_READY_PHASE                  9
#define READY_PHASE                     10
#define ENTER_STATE_MACHINE             11
#define INIT_COMPLETE                   12
#define INIT_CHANGE_STATE               13
#define PRE_SHUTDOWN_SYSTEM             14
#define PRE_SHUTDOWN_COMPLETE           15
#define SHUTDOWN_SYSTEM                 16
#define PING_CLIENT                     17
#define PRINT_HISTOGRAM                 18
#define EXECUTE_RESET_HNDL              19
#define ENTER_DEACTIVATED_STATE_MACHINE 20
#define CLIENT_DATA_PKT                 21
#define CLIENT_DATA_PKT_REDIRECT        22
#define EXECUTE_LOCK_COMMAND            23
#define EXECUTE_LOCK_COMMAND_COMPLETE   24
#define EXTERNAL_LOG                    25
#define EXTERNAL_LOGGER_PRESENT         26
#define REGISTER_EXTERNAL_LOGGER        27
#define CLIENTS_ARE_CONNECTED           28
#define CLIENT_DATA_SEH                 29

//// Commands that SSM accepts from external sources
//// Make sure the numbers are unique
#define INIT_SHUTDOWN                   30
#define SAFE_STATE                      31
#define DRIVER_OVERRIDE                 32
#define LOG_PERF_STATE_CHANGE           33
#define LOG_PERF_INIT_COMPLETE          34
#define INIT_RESET                      35
#define INIT_SHUTDOWN_SKIP_PRESHUTDOWN  36

/// GenericClone Commands
#define GET_GLOBAL_STATES               70
#define SET_GLOBAL_STATE                71
#define SET_GLOBAL_START_STATE          72
#define GET_LOCK_CMDS_LIST              73
#define SET_LOCK_CMDS_LIST              74
#define REGISTER_SAFETY_MODULE          75
#define EXECUTE_SAFE_STATE_CHANGE       76
#define SAFE_STATE_CHANGE_COMPLETE      77


#define PRE_INIT_PHASE_STR              "PRE_INIT"
#define INIT_PHASE_STR                  "INIT"
#define POST_INIT_PHASE_STR             "POST_INIT"
#define PRE_READY_PHASE_STR             "PRE_READY"
#define READY_PHASE_STR                 "READY"
#define ENTER_STATE_MACHINE_STR         "ENTER_STATE_MACHINE"
#define EXECUTE_RESET_HNDL_STR          "EXECUTE_RESET_HNDL"
#define SWC_LIST                        "SWC_LIST"
#define RC_LOG_STRING                   "_DEFAULT_SSM_LOG_"
#define RC_STATE_CHANGE                 "_STATE_CHANGE_"

// TODO (NDF-1015): Move these LCM-specific codes to autogenerated LCM Clone
// ------------------------------------------------------------------------
/// LCM Strings
#define LCM_COMMAND         "LCM_COMMAND"
#define LCM_SM_NAME         "LCM"
#define LCM_CLIENT_MESSAGE  "LCM_CLIENT_MESSAGE"

/// LCM Requests to SSM
enum class LcmRequestToSSM : uint32_t
{
    /// Invalid request
    INVALID = 0,
    /// Request SSM to perform state change as part of Start_P0 LockedSteppedCommand
    STATE_CHANGE_START_P0,
    /// Request SSM to perform state change as part of Start_P1 LockedSteppedCommand
    STATE_CHANGE_START_P1,
    /// Request SSM to change state to End of Mission
    STATE_CHANGE_END_OF_MISSION,
};

/// Commands sent from LCM to LCM Clients
enum class LcmCommand : uint32_t
{
    /// Invalid command
    INVALID = 0,
    /// Command to LCM Safety Module to Call DriveOS API for system shutdown
    CALL_DRIVEOS_SHUTDOWN,
    /// Command to LCM Safety Module to Call DriveOS API for system suspend
    CALL_DRIVEOS_SUSPEND,
    /// Command to LCM VAL Bridge to update shutdown/suspend/reboot details
    END_OF_CYCLE,
};

/// Requests from LCM Clients to LCM
enum class LcmClientRequest : uint32_t
{
    /// Invalid request
    INVALID = 0,
    /// Start shutdown sequence, terminate apps and call DriveOS API for shutdown
    SHUTDOWN,
    /// Reboot system
    REBOOT,
    /// Suspend system
    SUSPEND,
    /// No-op Enum to be used for setting start value of LCMClientResponse
    LCM_CLIENT_REQUEST_MAX,
};

// TODO (NDF-1016): Both LcmClientRequest and LcmClientResponse are from Client to LCM.
// Both are typecasted into uint32_t for use in LcmMessage.message. Need to do better.
// Maybe messageType should be used for this and macros LCM_COMMAND and LCM_CLIENT_MESSAGE
// should be deprecated.

/// Response from LCM Clients
enum class LcmClientResponse : uint32_t
{
    /// LCM Client accepted the request
    ACK = static_cast<uint32_t>(LcmClientRequest::LCM_CLIENT_REQUEST_MAX) + 1,
    /// LCM Client rejected the request
    NACK,
};

/// Bootmode to be used in next boot cycle. Set before shutdown/reboot
enum class NextBootMode : uint32_t
{
    /// Invalid Boot Mode
    INVALID = 0,
    /// Same Boot Mode as current Boot Mode
    UNCHANGED,
    /// Normal Boot Mode
    NORMAL,
    /// Factory Boot Mode
    FACTORY,
};

typedef struct ShutdownParams
{
    bool runIST;
    NextBootMode nextBootMode;
} ShutdownParams;

typedef struct RebootParams
{
    bool runIST;
    NextBootMode nextBootMode;
} RebootParams;

typedef struct SuspendParams
{
    uint32_t reserved;
} SuspendParams;

enum class EndOfCycleInfo : uint32_t
{
    ///Invalid info
    INVALID = 0,
    ///Shutdown without IST
    SHUTDOWN_WITHOUT_IST,
    ///Shutdown with IST
    SHUTDOWN_WITH_IST,
    ///Reboot without IST
    REBOOT_WITHOUT_IST,
    ///Reboot with IST
    REBOOT_WITH_IST,
    ///Suspend
    SUSPEND,
};

typedef struct EndOfCycleParams
{
    EndOfCycleInfo info;
} EndOfCycleParams;

/// LCM Message
typedef struct LcmMessage
{
    uint32_t message{};
    ShutdownParams shutdownParams;
    RebootParams rebootParams;
    SuspendParams suspendParams;
    EndOfCycleParams endOfCycleParams;
} LcmMessage;
// ------------------------------------------------------------------------

#define MAX_ACTIVATE_RETRY_TIMES        10
#define FSI_MSGLENGTH_INDEX             2
#define FSI_MSGLENGTH_VALUE_LENGTH      4
#define MAX_SYSTEM_FAILURE_REPORT_COUNT 200

enum class SSM_DEFERRED_INIT : uint8_t {
    START_INIT = 0,
    PRE_INIT_PHASE_UNDERWAY,
    INIT_PHASE_UNDERWAY,
    POST_INIT_PHASE_UNDERWAY,
    PRE_READY_PHASE_UNDERWAY,
    READY_PHASE_UNDERWAY,
    FREE_BUCKET,
};

enum class SSM_LOG_TYPE : uint8_t {
    ERROR = 0,
    PERF,
    HIST,
    INFO,
    DEBUG,
    RC,
};

#define spin_unlock(x)  while (!__sync_bool_compare_and_swap(&x, 1, 0)) {}
#define spin_lock(x)    while (!__sync_bool_compare_and_swap(&x, 0, 1)) {}

////// Generic spin lock class for lock_guard
class SpinLock {
public:
    SpinLock(){};
    inline void lock() {spin_lock(lk);}
    inline void unlock() {spin_unlock(lk);}

private:
    bool lk{false};
};

class GenericClone;

// Use this for unit tests ONLY
#define CHECK_FILE_MACRO(A) if (!std::getline(infile, line) || line != A) {return false;}

typedef std::function<void(SSM_LOG_TYPE, std::string, std::string)> SSMLogFunctionPtr;

typedef std::function<bool( GenericClone *ptr,
                            const char *fromState,
                            const char *toState,
                            const uint64_t transitionID,
                            uint64_t timeout)> SSMSafetyModuleFunctionPtr;

typedef std::atomic<uint32_t> AtomicUInt;
typedef std::atomic<bool> AtomicFlag;
typedef std::lock_guard<SpinLock> SSMSpinLock;
typedef std::lock_guard<std::mutex> SSMLock;
typedef std::map<std::string, std::string> StringMap;

// Turn this flag to true incase you want to see debug info
extern SSM_LOG_TYPE ssmLogLevel;

extern SSMLogFunctionPtr logFunPtr;
extern int stateMachineInitTimeout;
extern int overrideBasePort;
extern int qaSSMCmdChannelPort;
extern int finalMaxPort;
extern int ssmCommandChannelPort;
extern int ssmLogPort;
extern int startSocketPort;
extern bool isInitHierarchyComplete;
extern std::string ssmMasterIPAddr;

//////////////////////////////////////////////////////////////////
// TODO: Replace std::map & std::ostringstream
//       with fixed size containers
//       [https://jirasw.nvidia.com/browse/NVSTM-1057]
extern StringMap SSMMasterQACommandMap;
typedef std::set<std::string> StringSet;
typedef std::ostringstream FixedStringStream;
//////////////////////////////////////////////////////////////////

extern bool isSSMMasterFileProcessed;
extern bool isMessageBufferConfigured();

/************************************************************************************************
* The following APIs should be used to redirect logs from all applications to a centralized
* application. Roadcast accepts logs from a binary that has roadcast node and so
* all SSM related applications must be redirected to that application that has the node.
* Use the following instructions to implement the centralized logging facility:
*    1. Select the binary that has the roadcast node and implement a global function hander
*       that has a roadcast producer using the SSMCentralizedLogFunctionPtr definition
*    2. Call the setupCentralizedLogger(...) function UPFRONT before all the SSM children are
*       initialized
*    3. Check the log redirections to the roadcast producer function handler
*
* NOTE: Add RCIDs as needed & update the getRCIDString() function
*
*/
enum class RCID : int {
    SSM_INTERNAL = 0,         // Meant for SSM internal usage
    SSM_STATE_CHANGE,         // Meant for documenting state changes
    SSM_APPLICATION,
};

typedef struct stateChangeMsgStruct_ {
    uint64_t timestamp;
    int stateMachineID;
    int stateID;
} SSMStateChangeMsg;

typedef std::function<void(RCID logType,
                           std::string eventTag,
                           void *dataBuffer,
                           unsigned int bufferSize)> SSMCentralizedLogFunctionPtr;

extern int centralizedLoggerServerPort;

/// Primary API for registering logger function handler
extern void setupCentralizedLogger(SSMCentralizedLogFunctionPtr rcptr);
extern std::string getRCIDString(SystemStateManager::RCID rcid);
extern bool documentRCLog(SystemStateManager::RCID logType,
                          std::string eventTag,
                          void *dataBuffer,
                          unsigned int dataBufferSize);
void documentStateChange(int stateMachineID, int stateID);
inline bool documentEvent(std::string eventTag,
                         void *dataBuffer,
                         unsigned int dataBufferSize) {
    return documentRCLog(SystemStateManager::RCID::SSM_APPLICATION,
                         eventTag,
                         dataBuffer,
                         dataBufferSize);
}
///////////////////////////////////////////////////////////////////////////////////////////////


static inline std::string getTimestamp()
{
    struct timespec time
    {
    };
    std::ostringstream ostr;
    clock_gettime(CLOCK_REALTIME, &time);
    struct tm* calendar = localtime(&time.tv_sec);
    ostr << (calendar->tm_mday < 10 ? "[0" : "[") << calendar->tm_mday;
    ostr << (calendar->tm_mon < 9 ? "-0" : "-") << (calendar->tm_mon + 1) << "-" << (1900 + calendar->tm_year);
    ostr << (calendar->tm_hour < 10 ? " 0" : " ") << calendar->tm_hour;
    ostr << (calendar->tm_min < 10 ? ":0" : ":") << calendar->tm_min;
    ostr << (calendar->tm_sec < 10 ? ":0" : ":") << calendar->tm_sec << "]";
    return ostr.str();
}

void ssmLoggerAPI(SSM_LOG_TYPE logType,
                  std::string str,
                  SystemStateManager::RCID rcid);

#define DEBUG_LOG_TYPE SystemStateManager::SSM_LOG_TYPE::DEBUG
#define INFO_LOG_TYPE SystemStateManager::SSM_LOG_TYPE::INFO
#define ERROR_LOG_TYPE SystemStateManager::SSM_LOG_TYPE::ERROR
#define PERF_LOG_TYPE SystemStateManager::SSM_LOG_TYPE::PERF
#define RC_LOG_TYPE SystemStateManager::SSM_LOG_TYPE::RC

#define DEFAULT_RC_TYPE SystemStateManager::RCID::SSM_INTERNAL

#define SSM_DEBUG(A)                                                                              \
    if (SystemStateManager::ssmLogLevel >= DEBUG_LOG_TYPE)                                        \
    {                                                                                             \
        SystemStateManager::ssmLoggerAPI(DEBUG_LOG_TYPE, std::string(A), DEFAULT_RC_TYPE);        \
    }

#define SSM_LOG(A)                                                                                \
    if (SystemStateManager::ssmLogLevel >= INFO_LOG_TYPE)                                         \
    {                                                                                             \
        SystemStateManager::ssmLoggerAPI(INFO_LOG_TYPE, std::string(A), DEFAULT_RC_TYPE);         \
    }

#define SSM_ERROR(A)                                                                              \
    if (SystemStateManager::ssmLogLevel >= ERROR_LOG_TYPE)                                        \
    {                                                                                             \
        SystemStateManager::ssmLoggerAPI(ERROR_LOG_TYPE, std::string(A), DEFAULT_RC_TYPE);        \
    }

#define SSM_PERF(A)                                                                               \
    if (SystemStateManager::ssmLogLevel >= PERF_LOG_TYPE)                                         \
    {                                                                                             \
        SystemStateManager::ssmLoggerAPI(PERF_LOG_TYPE, std::string(A), DEFAULT_RC_TYPE);         \
    }

#define SSM_ERROR_WITH_ERRNO(A)                                                                   \
    SystemStateManager::ssmLoggerAPI(ERROR_LOG_TYPE, std::string(A), DEFAULT_RC_TYPE);

class SSMPerfStream : public std::stringstream
{
public:
    ~SSMPerfStream() 
    {
        SSM_PERF(this->str());
    }
};
#define SSM_PERF_STREAM SystemStateManager::SSMPerfStream()

#define SSM_HISTOGRAM(A) cout << A << endl;

typedef enum ActuationSupervisorPayload
{
    PAYLOAD_AS_NONE = 0,         // Both lat and long not toggled
    PAYLOAD_AS_LAT_TOGGLED,      // Only lat toggled
    PAYLOAD_AS_LONG_TOGGLED,     // Only long toggled
    PAYLOAD_AS_LAT_LONG_TOGGLED, // Both lat and long toggled
} ActuationSupervisorPayload;

// Try to have this struct as small as possible
typedef struct _StateUpdate {
    uint64_t timestamp;
    int command;
    char stateMachineName[MAX_STATE_MACHINE_SIZE];
    char toState[MAX_STATE_NAME];
    char extraInfo[MAX_STATE_NAME];
    int arg;
    int metaData;
    bool isPayloadPresent;
} StateUpdate;


typedef struct _userDataPktInternal {
    // Arbitrary type that the receiver can use to typecast
    // the data buffer. For example, SSM can receive
    // different types of data buffers from different SMs.
    // This field helps SSM in choosing the appropriate type
    std::string dataType;
    uint64_t timestamp;
    std::string targetStateMachine;
    std::string sourceStateMachine;
    int size;
    char data[MAX_DATA_PKG_SIZE];
} UserDataPktInternal;


// Use this to debug messages
void setPorts();
void printMessage(StateUpdate *buffer, std::string strName="SSM Msg: ");
std::string getCommandStr(int cmd);
uint64_t getMilliseconds();
uint64_t getMicroseconds();
int getStateMachineInitTimeout();
void setupLogPerfLevel();
void setupLogErrorLevel();
void setupLogDebugLevel();
bool setSSMBasePort(int portNumber);

// Returns true if successful; false if ssmBasePort is not set yet
bool getSSMPortRange(int &startPort, int &endPort);

void setupLogFunction(SSMLogFunctionPtr lfptr);

// This function sets up the buffer infrastructure for in process communication
void setupBufferCommunication();

// Use this to create a parallel hierarchy when using buffer communication
int createNewParallelHierarchy();

void printPerfInitComplete(std::string smName, std::string stateName, uint64_t timestamp);
void printPerfStateChange(std::string smName, std::string stateName, uint64_t timestamp);

// Process ssm_master file and store key value pairs in map
void processSSMMasterQACommand();
std::string getKeyFromMasterQAFile(std::string key);

// Used for getting non-zero fd, as fd 0 is risky to be closed by other code accidentally
int getNonZeroFd(int fd);

};
