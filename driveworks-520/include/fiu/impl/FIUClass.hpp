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

#ifndef FIU_CLASS_H
#define FIU_CLASS_H
#include <map>
#include <memory>
#include <string.h>
#include <fiu/impl/FIUCommand.hpp>
#include <fiu/impl/FIUInstance.hpp>

namespace fiu2
{

/** @struct Holds the fault specific parameter data
 *
 * @var name Pointer to the name of the Fault parameter
 * @var isRequired Flag specifies if the param is absolutely required or not
 * @var type Pointer to the demangled type name
 */
struct FaultParam
{
    const char* name;
    bool isRequired;
    const char* type;
};

/// Initializer list of all the defined FaultParam objects
using ParamList = std::initializer_list<FaultParam>;

/**  @struct Holds the fault specific parameter data within a node
 *
 * @var name Buffer to hold the name of the Fault parameter
 * @var isRequired Flag specifies if the param is absolutely required or not
 * @var type Buffer to hold the demangled type name
 */
struct FaultParamNode
{
    char name[FIU_MAX_PARAMETER_NAME_SIZE];
    bool isRequired{false};
    char type[FIU_MAX_FAULT_PARAM_TYPE_DESC];
};

/** @struct Holds the Key Value pairs
 *
 * @var Key Pointer to the key
 * @var value Pointer to the value
 */
struct KeyValue
{
    const char* key;
    const char* value;
};

/** @struct Contains the information of the added instance
 *
 * @var nane Pointer to the name of the instance; present in FIUInstance object
 * @var flock lock that needs to be acquired before the contents of this object are modified
 * @var fiuCommand FIUCommand object that contains the user provided fault information
 * @var instance Copy of the instance information
 * @var fiuLineNumber line number where the FIU specific to this instance has been declared
 * @var fiuFile File where the FIU specific to this instance has been declared
 * @var validFIU Flag that specifies if the instance is valid or not
 * @var triggerCount Counter that holds the number of times this fault has been triggered
 * @var hitCount Counter that holds the number of times the fault has been hit
 * @var enableCount Counter that holds the number of times a fault has been enabled
 */
struct InstanceInfo
{
    const char* name{};
    SpinLock flock;
    FIUCommand* fiuCommand{};

    InstanceStruct instance{};

    // The following variables should be used to store the FIU location information
    int fiuLineNumber{0};
    const char* fiuFunction{};
    const char* fiuFile{};
    bool validFIU{true};

    // Fault Stats
    uint64_t triggerCount{};
    uint64_t hitCount{};
    uint32_t enableCount{};
};

/// Shard pointer for the InstanceInfoPtr;
using InstanceInfoPtr = std::shared_ptr<InstanceInfo>;

/// Initializer list type for the key value pair
using KeyValueList = std::initializer_list<FaultParam>;

class FIUClass
{
public:
    /**
     * Create a FIUClass object that represents a fault
     *
     * @param [in] _isBaseFaultEnabled Ptr to an external byte allocated in the data section that is used
     *             to verify if the fault is enabled or not. FI_DEFINE_FAULT() creates
     *             this flag when it is invoked. See fiu.hpp
     * @param [in] enabledInstancesCount Keeps track of the total number of instances enabled for this fault
     * @param [in] name Name of the fault
     * @param [in] module Name of the module this fault belongs to
     * @param [in] synopsys Synopsys for the fault (brief description)
     * @param [in] paramList Initializer list for all the parameters
     * @param [in] keyValueList key value pair list (for future use)
     *
     */
    FIUClass(uint8_t& _isBaseFaultEnabled,
             uint8_t& enabledInstancesCount,
             const char* name,
             const char* module,
             const char* synopsys,
             ParamList paramList,
             KeyValueList keyValueList);

    /**
     * Get fault Name
     *
     * @return name of the fault
     */
    const char* getName() { return m_fault->name; }

    /**
     * Get param count
     *
     * @return number of parameters
     */
    uint32_t getParamCount() { return m_totalParamCount; }

    /**
     * Get module name
     *
     * @return name of the module
     */
    const char* getModule() { return m_module; }

    /**
     * Get synopsys name
     *
     * @return name of the synopsys
     */
    const char* getSynopsys() { return m_synopsys; }

    /**
     * Add a new instance
     *
     * @param[in] instanceName Name of the new instance
     * @return true if the instance is added successfully
     */
    bool addInstance(const char* instName);

    /**
     * Add a new instance
     *
     * @param[out] inst InstanceStruct Object that will be filled with info
     * @param[in] name of the instance
     *
     * @return true if instance got added successfully
     */
    bool addInstance(InstanceStruct& inst, const char* instanceName);

    /**
     * Check if instance is present
     *
     * @param[in] instName Name of the instance
     * @return true if instance present
     */
    bool isInstancePresent(const char* instName);

    /**
     * Get total instances present in the fault
     *
     * @return total number of instances
     */
    uint32_t getInstanceCount();

    /**
     * Get instance status
     *
     * @param[in] index The index of instance
     * @return true if instance is enabled
     *
     * Note: This should be called from FIU_CHECK_xxx macros ONLY
     * Note: The index can range between 0 and getInstanceCount().
     */
    bool isEnabled(uint32_t index);

    /**
    * Get the instance status
     *
     * @return true if instance is enabled
     */
    bool isEnabled(const char* instName);

    /**
     * Get fault status
     *
     * @return true if fault is enabled
     */
    bool isEnabled();

    /// Debug function that prints the list all the available instances
    void printInstances();

    /**
     * API that inputs a fualt before it executes it
     *
     * @param[in] fiuCMD Inputs the FIUCommand object ptr
     * @return true if fault is valid and accepted
     */
    bool enableFault(FIUCommand* fiuCMD);

    /**
     * Lock the Fault
     */
    inline void lockFault()
    {
        m_fault->flock.lock();
    }

    /**
     * Unlock the Fault
     */
    inline void unLockFault()
    {
        m_fault->flock.unlock();
    }

    /**
     * Is the fault valid and usable
     *
     * @return true if fault is valid
     */
    bool isFIUValid()
    {
        return m_fault->validFIU;
    }

    /**
     * Lock the instance
     *
     * param[in] index of the instance
     */
    void lockFault(uint32_t index)
    {
        m_fiuInstanceRef.lock(index);
    }

    /**
     * Unlock the instance
     *
     * @param[in] index Index of the instance
     */
    void unLockFault(uint32_t index)
    {
        m_fiuInstanceRef.unlock(index);
    }

    /**
     * Is fault valid and usable
     *
     * @param[in] index Index of the instance
     * @return true if fault is valid
     */
    bool isFIUValid(uint32_t index)
    {
        return instanceArray[index]->validFIU;
    }

    /**
     * Is the param present
     *
     * @param[in] paramName Name of the parameter
     * @param[in] ptype Type of the parameter in char
     *
     * @return true if the instance is added successfully
     */
    bool isParamPresent(const char* fname, const char* ptype);

    /**
     * API to disable the fault
     */
    void disableFault() { disableFault(m_fault); }

    /**
     * Disable a specific instance of the fault
     *
     * @param[in] instanceName Name of the instance that should be disabled
     * @return true if the instance is added successfully
     */
    void disableFault(const char* instanceName);

    /**
     * Inform that the fault's trigger logic executes next
     */
    void executeTriggerLogic();

    /**
     * Inform that the specific instance's trigger logic executes next
     *
     * @param[in] instanceIndex Index of the instance
     */
    void executeTriggerLogic(uint32_t instanceIndex);

    /**
     * Update fault's hit statistics
     */
    void executeHitLogic();

    /**
     * Update the specific instance's hit statistics
     *
     * @param[in] instanceIndex Index of the instance
     */
    void executeHitLogic(uint32_t instanceIndex);

    /**
     * Disable the fault if conditions are met
     */
    void checkFaultDisableConditions();

    /**
     * Disable the specific instance if the conditions are met
     *
     * @param[in] instanceIndex Index of the instance
     */
    void checkFaultDisableConditions(uint32_t instanceIndex);

    /**
     * Check if the fault's frequency permits a trigger
     *
     * @return true if the fault can trigger
     */
    bool checkFrequency();

    /**
     * Check if the frequency permits an instance trigger
     *
     * @param[in] instanceIndex Index of the instance
     * @return true if the instance can trigger
     */
    bool checkFrequency(uint32_t instanceIndex);

    /**
     * Validate the FIU's location and get FaultCommand object
     *
     * @param[in] lineNumber Location of the FIU
     * @param[in] funcName Name of the function where the FIU resides
     * @param[in] fileName Name of the file where the FIU resides
     * @return FaultCommand object
     */
    FaultCommand* getFC(int lineNumber, const char* function, const char* file);

    /**
     * Validate the FIU's location and get FaultCommand object for the instance
     *
     * @param[in] inst InstanceStruct object for the instance used in the FIU
     * @param[in] instanceIndex Index of the instance
     * @param[in] lineNumber Location of the FIU
     * @param[in] funcName Name of the function where the FIU resides
     * @param[in] fileName Name of the file where the FIU resides
     * @return FaultCommand object
     */
    FaultCommand* getFC(InstanceStruct& inst,
                        uint32_t& index,
                        int lineNumber,
                        const char* function,
                        const char* file);

    /**
     * Set the total allowed parameters
     *
     * @param[in] newParamCount New value for the number of parameters allowed
     *
     * NOTE: Do NOT use in production. This is meant for testing
     */
    void setTotalParamsAllowed(uint32_t newParamCount) { m_totalParamCount = newParamCount; }

    /// Pointer to the fault instance
    FIUInstance& m_fiuInstanceRef;

private:
    /// Pointer to the fault synopsys
    const char* m_synopsys{};

    /// Pointer to the fault module name
    const char* m_module{};

    /// Object stores the data relevant to the fault
    InstanceInfoPtr m_fault{};

    /// Checks if the fault is enabled or not
    uint8_t& m_isBaseFaultEnabledRef;

    /// Maintains the count of enabled instances
    uint8_t& m_enabledInstancesCountRef;

    /// Objects store the data of the instances
    InstanceInfoPtr instanceArray[TOTAL_INSTANCES_SUPPORTED_PER_FAULT]{};

    /// Lock this whenever totalIntances needs to be used in a meaningful way
    SpinLock m_totalInstancesLock{};

    /// Counter that holds the total number of intances added to the fault
    uint32_t m_totalInstances{0};

    /// Counter that holds the total parameters added to the specific fault
    uint32_t m_totalParamCount{0};

    /// Array of the defined fault Parameter nodes
    FaultParamNode m_faultParamNodes[TOTAL_PARAMETERS_ALLOWED_PER_FAULT];

    /// This holds instanceMap vs index in the InstanceInfoPtr;
    // TODO: Optimize this... use a bitmap instead of a map
    IndexMap m_indexMap;

    //////////////// Member Functions ///////////////////
    /// API that returns the index of an instance if present

    /** Checks if the specified instance is present or not
     *
     *  @param[in] instName Name of the instance
     *  @param[out] index Index of the specified instance if presnet
     *  @return true if the instance is present; false otherwise
     */
    bool isInstancePresent(const char* instName, uint32_t& index);

    /** Returns the FIUCommand object that belongs to the enabled FIU object
     *
     *  @param[in] ptr Pointer to the InstanceInfo object
     *  @param[in] fiuLine Line number where the FIU is located at
     *  @param[in] function Function name within which the FIU is present
     *  @param[in] file File name where the FIU is present
     *  @return FaultCommand object if the fault is enabled
     */
    FaultCommand* getFC(InstanceInfoPtr ptr, int fiuLine, const char* function, const char* file);

    /** Check if the provided parameters are present in the defined fault
     *
     *  @param[in] fiuCMD Pointer to the fiu command
     */
    bool doParamsMatch(FIUCommand* fiuCMD);

    /** Check if the instance is defined and enable the fault as appropriate
     *
     *  @param[in] fiuCMD Pointer to the fiu command
     */
    bool checkForInstanceEnableFault(FIUCommand* fiuCMD);

    /** Disable the fault instance
     *
     *  @param[in] instanceInfoPtr Pointer to the instanceInfo object
     */
    void disableFault(InstanceInfoPtr instanceInfoPtr);

    /** Updates the stats that tracks the hit count
     *
     *  @param[in] instanceInfoPtr Pointer to the instanceInfo object
     */
    void executeHitLogic(InstanceInfoPtr instanceInfoPtr);

    /** Updates the stats that tracks the execute count
     *
     *  @param[in] instanceInfoPtr Pointer to the instanceInfo object
     */
    void executeTriggerLogic(InstanceInfoPtr instanceInfoPtr);

    /** Checks if the fault can be disabled after execution
     *
     *  @param[in] instanceInfoPtr Pointer to the instanceInfo object
     */
    void checkFaultDisableConditions(InstanceInfoPtr instanceInfoPtr);

    /** Checks if the frequency allows for the execution of the specific hit
     *
     *  @param[in] instanceInfoPtr Pointer to the instanceInfo object
     */
    bool checkFrequency(InstanceInfoPtr instanceInfoPtr);
};

/** Helper function to add an instance to multiple faults
 *
 *  @param instName Name of the instance
 *  @param faultList List faults linked to the instance set
 */
inline void mapInstanceToFault(const char* instName, std::initializer_list<fiu2::FIUClass*> faultList)
{
    for (auto& name : faultList)
    {
        name->addInstance(instName);
    }
}
};

#endif
