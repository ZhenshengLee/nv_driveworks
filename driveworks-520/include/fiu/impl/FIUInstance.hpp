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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef FIU_INSTANCE_H
#define FIU_INSTANCE_H
#include <fiu/impl/FIUCommon.hpp>

namespace fiu2
{

constexpr int INSTANCES_SUPPORTED_BIT_BUFFER{1 + (TOTAL_UNIQUE_INSTANCES_SUPPORTED / CHAR_BIT)};
constexpr int FAULTS_BIT_BUFFER{1 + (MAX_FAULTS_PER_INSTANCE / CHAR_BIT)};

// This structure has to support low latency to the maximum extant. Consider cache
// implications when you modify this structure.
//
// The reasoning behind logic is described in Section 5 of the FI Tool's SADD
//  - https://docs.google.com/document/d/1In4jMnI5CdNOPMKI5y_1_kfqAuCc81Qp/edit#heading=h.dfu2666ifhhm
//  - Search for "cache prefetch algorithms to load cache lines"
struct UniqueInstanceStruct
{

    // This buffer contains the bits for all the instances [1->enabled, 0-> disabled]
    unsigned char isInstanceEnabledBitBuffer[INSTANCES_SUPPORTED_BIT_BUFFER];

    // This matrix contains the bits for all the faults for each instance
    unsigned char isFaultInstanceEnabledBitBuffer[TOTAL_UNIQUE_INSTANCES_SUPPORTED][FAULTS_BIT_BUFFER];

    // Containers the counters that indicate how many faults have enabled the same instance
    uint16_t instanceUsageCounter[TOTAL_UNIQUE_INSTANCES_SUPPORTED];

    // Contains the number of faults assigned to each instance
    uint32_t uniqueFaultsPerInstanceLength[MAX_FAULTS_PER_INSTANCE];
};

class FIUInstance
{
public:
    static FIUInstance& getInstance()
    {
        static FIUInstance instance;
        return instance;
    }

    /**
     * Converts a string into an object that can be referenced by fault/instance pairs to identify
     * the instance
     *
     * @param [in-out] Instance Handle
     * @param [in]     Instance Name string
     *
     * NOTE: API is Thread safe
     */
    bool addUniqueInstance(InstanceStruct& inst, const char* instanceName);

    /**
     * Dedicates a bit to the fault/instance combo that the FI tool can look up to 
     *
     * @param [in-out] Instance Handle
     * @param [in]     Instance Name string
     *
     * NOTE: API is Thread safe
     */
    bool addFaultInstance(InstanceStruct& inst, const char* instanceName);

    /// Note: DO NOT put this code within a critical section of instanceLock[inst.instanceIndex]
    ///       instanceLock should be acquired in FIUClass before the individual InstanceInfo object
    ///       acquires the lock... else the critical sections in this api and the one present in
    ///       markInstanceAsDisabled become out of order and can lead to deadlocks.
    ///
    ///       ORDER OF LOCKING IS VERY IMP... else DEADLOCK
    bool markInstanceAsEnabled(InstanceStruct& inst);

    /// Note: DO NOT put this code within a critical section because it already runs within the critical
    ///       section created within FI_CHECK_INSTANCE(...) macro
    bool markInstanceAsDisabled(InstanceStruct& inst);

    /**
     * Acquires the lock for the specific index. This ensures that another thread can't work
     * on the same instance at any point of time
     *
     * @param [in] Instance index
     *
     * NOTE: API makes everything Thread safe
     */
    void lock(uint32_t instanceIndex)
    {
        instanceLock[instanceIndex].lock();
    }

    /**
     * Releases the lock for the specific index.
     *
     * @param [in] Instance index
     *
     * NOTE: API makes everything Thread safe
     */
    void unlock(uint32_t instanceIndex)
    {
        instanceLock[instanceIndex].unlock();
    }

    /**
     * Checks if the fault is enabled
     *
     * @param [in] Instance handle
     * @return true if the fault instance is enabled
     *
     * NOTE: API makes everything Thread safe
     */
    bool isFaultInstanceEnabled(InstanceStruct& inst)
    {
        return getBit(m_uniqueInstanceStruct->isFaultInstanceEnabledBitBuffer[inst.instanceIndex], inst.faultInstanceIndex);
    }

    uint32_t getUniqueInstanceCount();

    /**
     * Checks if the unique instance is present or not
     *
     * @param [in] Instance handle
     * @return true if the instance is present
     * 
     */
    bool isUniqueInstancePresent(const char* instanceName);

    ~FIUInstance()
    {
        delete m_uniqueInstanceStruct;
    }

    /**
     * The following APIs permanently alter the FIUManager functionality
     * These APIs should be unsed for test purposes only.
     *
     * Note: These APIs is not exposed to the production code during the build process
     */
    InstanceSetPtr gtests_getEnabledInstances();
    InstanceSetPtr gtests_getAllInstances();

private:
    FIUInstance();

    /// Use raw pointer... shared_ptr's impact on low latency cache pre-fetch cycle
    /// friendly data structure is unknown at this point. R&D is required to confirm
    /// the impact
    UniqueInstanceStruct* m_uniqueInstanceStruct{};

    /// Lock the associated instance when you enable/disable instance
    SpinLock instanceLock[TOTAL_UNIQUE_INSTANCES_SUPPORTED]{};

    /// Comment this to detect race conditions using helgrind; see the mutex
    /// in FIUInstance.cpp
    SpinLock m_uniqueInstanceStructLock{};

    /// Pointer to the InstanceMap object
    InstanceMapPtr m_instanceMapPtr{};

    /// Disables the copy Constructors
    FIUInstance(const FIUInstance&) = delete;

    /// Disables copy assignment operator
    FIUInstance& operator=(const FIUInstance&) = delete;

    /// Specifies if the FIUInstance object is valid or not
    bool m_isValid{true};

    /// Holds the count of the total instances availabe in the system
    uint32_t m_totalUniqueInstancesAvailable{};

    /**
     * Sets the bit within the buffer based on the specified index
     *
     * @param buffer Pointer to the buffer of bits
     * @param index Index of the bit that needs to be set
     */
    void setBit(unsigned char* buffer, uint32_t index);

    /**
     * Resets the bit within the buffer based on the specified index
     *
     * @param buffer Pointer to the buffer of bits
     * @param index Index of the bit that needs to be reset
     */
    void resetBit(unsigned char* buffer, uint32_t index);
};
};

#endif
