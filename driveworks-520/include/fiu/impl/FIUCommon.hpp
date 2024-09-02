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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// FI Tool Version:    0.2.0
//
/////////////////////////////////////////////////////////////////////////////////////////
#ifndef FIU_COMMON_H
#define FIU_COMMON_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <string.h>
#include <cxxabi.h>
#include <mutex>
#include <map>
#include <set>
#include <cstdlib>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <limits.h>
#include <atomic>

namespace fiu2
{

extern bool FI_var_ARE_FIUS_ENABLED;
constexpr uint8_t FIU_MAX_STRING_SIZE{128};

constexpr uint8_t FIU_MAX_FAULT_PARAM_TYPE_DESC{16};
constexpr uint8_t FIU_MAX_FAULT_NAME_SIZE{128};
constexpr uint32_t FIU_MAX_INSTANCE_NAME_SIZE{1024};
constexpr uint8_t FIU_MAX_PARAMETER_NAME_SIZE{128};

constexpr uint32_t TOTAL_INSTANCES_SUPPORTED_PER_FAULT{2048};
constexpr uint32_t TOTAL_UNIQUE_INSTANCES_SUPPORTED{2048};

constexpr uint32_t MAX_FAULTS_PER_INSTANCE{2048};
constexpr uint8_t MAX_INSTANCES_PER_FAULT_ENABLED_AT_ANY_POINT_OF_TIME{255};

constexpr uint8_t TOTAL_PARAMETERS_ALLOWED_PER_FAULT{24};
constexpr uint8_t TOTAL_PARAMETERS_ALLOWED_PER_FAULT_COMMAND{4};
constexpr int32_t MAGIC_NUMBER{0x16e4};

/// This version supports only these number of faults. This can change
/// in subsequent versions or as need arises
constexpr uint8_t TOTAL_FAULT_COMMAND_SUPPORTED{16};

/// Maximum buffer size that needs to be allocated for JSON
constexpr int32_t MAX_ALLOWED_JSON_ARRAY_SIZE{1024};

/// Total types supported by the tool
constexpr uint8_t TOTAL_SUPPORTED_TYPES{24};

/// Macro renames all the variables that need to be used for special sections
#define FI_FIU_VAR_INTERNAL(A) FI_var_##A

/** @enum fiu2::FaultFrequency
 *
 *  A a strongly typed enum class representing the user specified frequency for a fault
 */
enum class FaultFrequency : uint8_t
{
    NEVER    = 0,
    ONCE     = 1,
    ALWAYS   = 2,
    SEQUENCE = 3
};

/** @enum fiu2::FaultLogLevel
 *  A a strongly typed enum class representing the log level
 */
enum class FaultLogLevel : uint8_t
{
    ERROR = 0,
    INFO  = 1,
    PERF  = 2,
    DEBUG = 4,
    NONE  = 0 // FIXME: None should be different from ERROR value. Fix this when
              //        the logger functionality is revamped
};

/** @enum fiu2::InstanceStructType
 *  A a strongly typed enum class representing the type of instance handle object
 *  the user has defined.
 */
enum class InstanceStructType : uint8_t
{
    uniqueGlobalInstance = 0,
    faultInstancePair    = 1,
    notAnInstance        = 2,
};

/** @struct Holds a single vector Parameter
 *
 *  @var paramName  Name of the vector parameter
 *  @var demangledTypeName  Name of the type in demangled form
 *  @var blobPtr Pointer to the buffer that holds the vector data
 *  @var blobSize Size of the vector data in bytes
 *  @var blobReference Reference that ObjectManager keep track of for buffer allocation/deallocation.
 *      Used in FIUCommand for reference to the memory pool allocation.
 */
struct FaultVectorParameter
{
    char paramName[FIU_MAX_PARAMETER_NAME_SIZE]{'\0'};
    char demangledTypeName[FIU_MAX_FAULT_PARAM_TYPE_DESC]{'\0'};
    void* blobPtr{};
    size_t blobSize{0};
    uint32_t blobReference{};
};

/** @struct Holds a single scalar Parameter
 *
 *  @var paramName  Name of the scalar parameter
 *  @var demangledTypeName  Name of the type in demangled form
 *  @var value Value of the scalar variable
 */
struct FaultScalarParameter
{
    char paramName[FIU_MAX_PARAMETER_NAME_SIZE]{'\0'};
    char demangledTypeName[FIU_MAX_FAULT_PARAM_TYPE_DESC]{'\0'};
    double value;
};

/** @struct Holds a sequence frequncy Counters
 *
 *  @var freqStart  Count at which the first fault trigger should happen after an enable
 *  @var freqCount  Number of times the fault has to trigger after freqStart
 *  @var freqStepIncrement Step increment between sequential freqCount
 */
struct SequenceFreqCounters
{
    uint64_t freqStart{};
    uint64_t freqCount{};
    uint32_t freqStepIncrement{1};
};

/** @struct Information about fault the user's fault instructions
 *
 *  @var magicNumber  Static magicNumber to verify the validty of the FI instruciton
 *  @var faultName  Name of the fault
 *  @var instanceName Name of the instance
 *  @var freq User specified frequency
 *  @var logLevel Fault specific log level as specified by the user
 *  @var sequenceFreq SequenceFreqCounters structure if frequency is a sequence
 *  @var totalScalarParamsAdded Total number of user provided scalar params for the fault
 *  @var totalVectorParamsAdded Total number of user provided vector params for the fault
 *  @var scalarArray Array of FaultScalarParameter of size totalScalarParamsAdded for
 *      holding scalar data
 *  @var totalVectorParamsAdded Array of FaultVectorParameter of size totalVectorParamsAdded
 *      for holding vector data
 */
struct FaultDescription
{
    int magicNumber{MAGIC_NUMBER};
    char faultName[FIU_MAX_FAULT_NAME_SIZE]{'\0'};
    char instanceName[FIU_MAX_INSTANCE_NAME_SIZE]{'\0'};
    FaultFrequency freq{FaultFrequency::NEVER};
    FaultLogLevel logLevel{FaultLogLevel::ERROR};
    SequenceFreqCounters sequenceFreq{};
    uint8_t totalScalarParamsAdded{0};
    uint8_t totalVectorParamsAdded{0};
    FaultScalarParameter* scalarArray[TOTAL_PARAMETERS_ALLOWED_PER_FAULT]{};
    FaultVectorParameter* vectorArray[TOTAL_PARAMETERS_ALLOWED_PER_FAULT]{};
};

/** @struct Tuple for the mangledName and its corresponding demangledName
 *
 *  @var mangledName  Mangled type name as per the RTTI
 *  @var demangledName  Type name as per the standard conventions
 */
struct TypeStructure
{
    char mangledName[3]{'\0'};
    char demangledName[FIU_MAX_FAULT_PARAM_TYPE_DESC]{'\0'};
};

/** @struct Handle that holds the instance information
 *
 *  @var type Type of instance; it is a uniqueGlobalInstance(set) or a tuple
 *  @var isInstanceEnabled Pointer to the instance's enable/disable flag
 *  @var isFaultInstanceEnabled Pointer to the fault's enable/disable flag
 *  @var instanceName Pointer to the name of the instance stored in FIUInstance object
 *  @var instanceIndex Index of the instance within the FIUInstance object
 *  @var faultInstanceIndex Index of the fault in the FIUInstance Object
 *  @var instanceIndexInsideFault Index of the fault inside FIUClass object
 *  @var isValid Is Instance valid or not
 *  @var isFaultSet Specifies if this instance is a set or not
 */
struct InstanceStruct
{
    InstanceStructType type{InstanceStructType::uniqueGlobalInstance};
    unsigned char* isInstanceEnabled{};
    unsigned char* isFaultInstanceEnabled{};
    const char* instanceName{};
    uint32_t instanceIndex{};
    uint32_t faultInstanceIndex{};
    uint32_t instanceIndexInsideFault{};
    bool isValid{true};
    bool isFaultSet{true};
    InstanceStruct(){};
    InstanceStruct(bool faultSet) { isFaultSet = faultSet; };
};

/**
 * Prints the contents of the instance handler
 *
 * @param[in] inst Reference to the instance handler
 */
void printInstanceStruct(InstanceStruct& inst);

/** Userspce spin lock class
 *
 * Userspace spinlocks should be used as an absolute last option. Mutexes are general recommended!
 * Time sensitive applications can't afford to use mutexes within their FI tool chain. Mutexes
 * can put contending threads to sleep and so the FI tool can introduce spurious perf issues that
 * impact realtime schedules.
 *
 * Userspace spinlock topic is very sensitive and involved; the following class uses a proven design
 * pattern that is based on compiler intrinsics. Compiler's compare and swap instructions are directly
 * converted into machine instructions that the underlying processor supports for synchronization.
 *
 * Note: DO NOT change this spinlock mechanism with another mechanism without performing an extensive
 *       perf analysis.
 */
class SpinLock
{
public:
    SpinLock(){};

    /// Acquires a lock (SpinLock)
    inline void lock()
    {
        while (!__sync_bool_compare_and_swap(&lk, 0, 1))
        {
        }
    }

    /// Releases a lock (SpinLock)
    inline void unlock()
    {
        while (!__sync_bool_compare_and_swap(&lk, 1, 0))
        {
        }
    }

private:
    bool lk{false};
};

/// Helper type definition for lock_guard
using FILock = std::lock_guard<SpinLock>;

/// String comparision operator for map search
struct CharBufferComparator
{
    bool operator()(const char* key1, const char* key2) const
    {
        return strcmp(key1, key2) < 0;
    }
};

/** Copies strings from source to destination in a safe manner
 *
 * @param[in] dest char * pointer where a string needs to be copied to
 * @param[in] source char * pointer from where a string needs to be copied to dest
 * @param[in] maxCopySize Maximum copy size allowed into dest
 */
inline void safeStringCopy(char* dest, const char* source, size_t maxCopySize)
{
    memset(static_cast<void*>(dest), '\0', maxCopySize);
    strncpy(dest, source, maxCopySize);
    if (maxCopySize > 0)
    {
        dest[maxCopySize - 1] = '\0';
    }
}

///////// TODO: Replace them with Fixed containers if required //////////
class FIUClass;
using FIClassMap    = std::map<const char*, FIUClass*, CharBufferComparator>;
using FIClassMapPtr = std::shared_ptr<FIClassMap>;

using InstanceMap    = std::map<std::string, uint32_t>;
using InstanceMapPtr = std::unique_ptr<InstanceMap>;

using IndexMap = std::map<uint32_t, uint32_t>;

using InstanceSet    = std::set<std::string>;
using InstanceSetPtr = std::shared_ptr<InstanceSet>;
///////////////////////////////////////////////////////////////////////////

/** @struct Handle that holds the instance information
 *
 *  @var isBool Is param bool or not
 *  @var isChar Is param char or not
 *  @var isArray Is param an array or not
 *  @var demangedNamed Pointer to the demangled name of the type
 *  @var minValue Minimum value allowed for the type
 *  @var maxValue Maximum value allowed for the type
 *  @var value Value of the variable
 *  @var arrayPtr Pointer to the array buffer if its an array
 *  @var arraySize Size of the array
 *  @var typeSize Size of the type in bytes
 */
struct ParamType
{
    bool isBool{false};
    bool isChar{false};
    bool isArray{false};
    const char* demangledName{};
    double minValue{};
    double maxValue{};
    double value{};
    void* arrayPtr{};
    size_t arraySize{};
    size_t typeSize{};
};

/**
 * This function is used to create a mapping between mangled names with their
 * respective demangled names. It calls the APIs that query the RTTI,
 * allocates memory and returns the name of the string (See abi function
 * abi::__cxa_demangle(...)). Once the API returns the demangled name it
 * copies the name into the TypeStructure object and deallocates the memory
 * that the abi function allocates. The mapping data structure is later used
 * to retreive the demangled name for every mangled name provided and vice versa.
 *
 * @param[in]  T Templated value.
 * @param[out] *dest pointer to which the unmangled string can be copied to
 * @return true if type is allowed
 */
template <typename T>
inline bool fillTypeDetails(T, TypeStructure* ts)
{
    int status{};

    if (!ts)
    {
        std::cerr << "Destination is empty!" << std::endl;
        return false;
    }

    // This code contains regions where memory is allocated and freed dynamically.
    // This code MUST not be present in FI Library runtime
    // A call to this function can be placed when FI library is initializing
    // This code is ok in every context of FI Client
    // https://stackoverflow.com/questions/1055452/c-get-name-of-type-in-template
    const char* mangledName = typeid(T).name();
    if (strlen(mangledName) > 2)
    {
        std::cerr << "Invalid manged name: " << mangledName << std::endl;
        return false;
    }

    ts->mangledName[0] = mangledName[0];
    ts->mangledName[1] = mangledName[1];

    bool returnType      = false;
    char* demangled_name = abi::__cxa_demangle(mangledName, NULL, NULL, &status);
    if (status == 0)
    {
        if (strlen(demangled_name) < FIU_MAX_FAULT_PARAM_TYPE_DESC)
        {
            strncpy(ts->demangledName, demangled_name, FIU_MAX_FAULT_PARAM_TYPE_DESC);
            returnType = true;
        }
        else
        {
            std::cerr << "Invalid Type: " << demangled_name << std::endl;
            returnType = false;
        }
    }
    std::free(demangled_name);
    return returnType;
}

/**
 * Init function that uses calls fillTypeDetails with all the allowed datatypes
 * and fills a data structure that maps the mangled names with their respective
 * demangled names.
 *
 * @return true if the data structures are created successfully
 */
bool createTypeStructures();

/**
 * Demangles the given mangled name
 *
 * @param[in]  mangedName Pointer to the RTTI's manged name
 * @return pointer to the demangled type name
 */
const char* getDemangledName(const char* mangledName);

/**
 * Get the demanged type name of a given template variable
 *
 * @param[in] a value of a template
 * @return pointer to the demangled type name
 */
template <typename T>
const char* getDemangledTypeName(T a)
{
    const char* mangledName = typeid(T).name();
    return getDemangledName(mangledName);
}

/**
 * Fills the type specs for a scalar parameter
 *
 * @param[in] a value for a template variable
 * @param[out] ptype Reference to the ParamType object that needs
 *                   to be filled with the type specs
 * @return true if the ParamType object was filled with type specs
 */
template <typename T>
bool fillPTypeStruct(T a, ParamType& ptype)
{
    ptype.typeSize      = sizeof(T);
    ptype.demangledName = getDemangledName(typeid(a).name());
    ptype.minValue      = std::numeric_limits<T>::min();
    ptype.maxValue      = std::numeric_limits<T>::max();
    return true;
}

/**
 * @brief Get the current time in microseconds since epoch.
 */
uint64_t getMicroseconds();

/**
 * Fills the type specs for a vector parameter
 *
 * @param[in] a value for a template variable
 * @param[in] b Size of the template variable
 * @param[out] ptype Reference to the ParamType object that needs
 *                   to be filled with the type specs
 * @return true if the ParamType object was filled with type specs
 */
template <typename T, typename S>
bool fillPTypeStructVector(T a, S b, ParamType& ptype)
{
    ptype.demangledName = getDemangledName(typeid(a).name());
    ptype.typeSize      = sizeof(S);
    ptype.minValue      = std::numeric_limits<S>::min();
    ptype.maxValue      = std::numeric_limits<S>::max();
    return true;
}

/// Base case for recursion
inline void printError()
{
    std::cerr << std::endl;
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void printError(T&& arg, Args&&... args)
{
    std::cerr << arg << ' ';
    printError(args...);
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void FI_ERROR(T&& arg, Args&&... args)
{
    std::cerr << "[FI ERROR] " << arg << ' ';
    printError(args...);
}

/// Base case for recursion
inline void printInfo()
{
    std::cout << std::endl;
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void printInfo(T&& arg, Args&&... args)
{
    std::cout << arg << ' ';
    printInfo(args...);
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void FI_INFO(T&& arg, Args&&... args)
{
    std::cout << "[FI INFO] " << arg << ' ';
    printInfo(args...);
}

/// Base case for recursion
inline void printDebug()
{
    std::cout << std::endl;
}

// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void printDebug(T&& arg, Args&&... args)
{
    std::cout << arg << ' ';
    printDebug(args...);
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void FI_DEBUG(T&& arg, Args&&... args)
{
    std::cout << "[FI DEBUG] " << arg << ' ';
    printDebug(args...);
}

/// Base case for recursion
inline void printPerf()
{
    std::cout << "[" << getMicroseconds() << "]" << std::endl;
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void printPerf(T&& arg, Args&&... args)
{
    std::cout << arg << ' ';
    printPerf(args...);
}

/// Recursive function to print variadic arguments
template <typename T, typename... Args>
inline void FI_PERF(T&& arg, Args&&... args)
{
    std::cout << "[FI PERF] " << arg << ' ';
    printPerf(args...);
}

/// Increments the total enabled fault counter
void incrementTotalEnabledFIUs();

/// Decrements the total enabled fault counter
void decrementTotalEnabledFIUs();

/**
 * Examines a specific bit in a buffer and returns if its set or not
 *
 * @param[in] buffer Pointer to the start of the buffer
 * @param[in] index Index of the specific bit from the start
 * @return true if the bit is set; else false
 */
inline bool getBit(const unsigned char* buffer, uint32_t index)
{
    if (!buffer)
    {
        FI_ERROR("Buffer NULL: ", __LINE__, __PRETTY_FUNCTION__, __FILE__);
        FI_ERROR("Looks like Instance is wrongly being used");
        FI_ERROR("Check the Fault/Instance pairs you are using; ensure instance is correctly paired");
        return false;
    }
    uint32_t byteIndex = index / CHAR_BIT;
    uint32_t bitOffset = index % CHAR_BIT;
    return static_cast<bool>(buffer[byteIndex] & (1 << bitOffset));
}

/// Connect to the FI Server
#define FI_CONNECT_FISERVER "FI_CONNECT_FISERVER"

/// Do not init FI Lib if this environment variable is set
#define FI_ENABLE_FLAG "FI_ENABLE_FLAG"

/// Flag that represents if the FI library is enabled or not
extern bool FI_var_IS_FI_LIB_ENABLED;

/** Add Unique instance to the InstanceMap
 *
 *  @param[in] inst Instance handle
 *  @param[in] instanceName Name of the instance
 *  @return true if the fault instances has been added to the global map
 */
bool globalAddUniqueInstance(InstanceStruct& inst, const char* instanceName);

/** API that FIClass uses to pass a used FIUCommand to the FIUManager
 *  so that it can be garbage collected.
 *
 * @param[in] object FIUCommand object that needs to be garbage collected
 * @return true if the object has been successfully garbage collected or not
 */
bool globalReuseFIUCommandObject(void* object);

/** FIUClass uses this API to add its "this" pointer to the FIUManager
 *
 *  @param[in] object Pointer to the FIUClass object
 *  @return true if the fault has been added to FIUManager's internal cache
 */
bool globalEnableFault(void* fiuCommandObj);

/** FIUClass uses this API to add its "this" pointer to the FIUManager
 *
 *  @param[in] object Pointer to the FIUClass object
 *  @return true if the fault has been added to FIUManager's internal cache
 */
bool globalAddFIUObject(void* object);

/** Indicates to the FIUManager that an instance has been added
 *
 *  @param[in] faultName Name of the fault
 *  @param[in] instanceName Name of the instance
 *  @return true if the fault instances was enabled right away or not
 */
bool globalAddInstanceEvent(const char* faultName, const char* instanceName);

/** Instructs the FIUManager to parse the file that contains the FI instructions
 *
 *  @param[in] filename Name of the file including its file path
 *  @return true if the file has been successfully read or not
 */
bool globalParseFIUFile(const char* filename);

/** Checks if the specific type is supported or not
 *
 *  @param[in] typeString Demangled name of the type
 *  @param[out] ptype Scalar type properties
 *  @return true if the type is supported
 */
bool isTypeSupported(const char* typeString, ParamType& ptype);

/** Check if the instance is present in the FIUInstance object or not
 *
 *  @param[in] instName Name of the instance
 *  @return true if the instance is present; false if otherwise
 */
bool globalIsUniqueInstancePresent(const char* instName);

/** Helper API that returns the total number of unique instance names defined
 *
 * @return Count of the total instances present
 */
uint32_t globalGetUniqueInstanceCount();
};

#endif
