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

#ifndef FIU_COMMAND_H
#define FIU_COMMAND_H
#include <fiu/FaultCommand.hpp>
#include <fiu/impl/ObjectManager.hpp>

namespace fiu2
{

class FIUCommand : public FaultCommand
{
public:
    ~FIUCommand(){};

    /**
     * Constructor
     *
     * @param[in] omgr Pointer to the ObjectManager pointer
     *
     * Note: omgr must be fully instantiated before the constructor is called
     */
    FIUCommand(ObjectManager* omgr)
        : FaultCommand(), m_objectManager(*omgr){};

    /**
     * Setter for the fault name
     *
     * @param[in] faultName Name of the fault
     */
    bool setFault(const char* faultName);

    /**
     * Setter for the fault name and instance name
     *
     * @param[in] faultName Name of the fault
     * @param[in] instanceName Name of the instance
     */
    bool setFault(const char* faultName, const char* instanceName);

    /**
     * Setter for the instance name
     *
     * @param[in] instanceName Name of the instance
     */
    bool setFaultInstance(const char* instanceName);

    /**
     * Getter for the fault description
     *
     * @return Description of the fault
     */
    FaultDescription* getFaultDescription()
    {
        return &m_fd;
    }

    /**
     * Getter for the fault description
     *
     * @param[out] fcID Fault Command ID
     * @return Description of the fault
     */
    FaultDescription* getFaultDescription(uint32_t& fcID)
    {
        fcID = m_faultCommandID;
        return &m_fd;
    }

    /**
     * Getter for the fault command ID
     *
     * @return fault command id
     */
    uint32_t getFaultID()
    {
        return m_faultCommandID;
    }

    /**
     * Setter for declaring that the FIUCommand object is ready
     * can be garbage collected and reused
     */
    void setReuseFlag()
    {
        m_reuseFlag = true;
    }

    /**
     * Getter for the fault command ID
     *
     * @return fault command id
     */
    bool getReuseFlag()
    {
        return m_reuseFlag;
    }

    /**
     * Getter for fault frequency
     *
     * @return Fault Frequency
     */
    FaultFrequency getFrequency();

    /**
     * Setter for setting the frequency to ONCE
     */
    void setFrequencyOnce();

    /**
     * Setter for setting the frequency to ALWAYS
     */
    void setFrequencyAlways();

    /**
     * Setter for setting the frequency to sequence
     *
     * @param[in] start N'th time the instruction counter (IC) executes the fault
     * @param[in] count Total number of fault triggers after start (1st param)
     * @param[in] stepIncrement General step increments between two successive counts
     */
    bool setFrequencySequence(uint64_t start, uint64_t count, uint32_t stepIncrement);

    /**
     * Setter for setting the frequency to Never; there by the fault disables itself
     */
    void setFrequencyNever();

    /**
     * Function that adds scalar parameters
     *
     * @param[in] name Name of the Scalar parameter
     * @param[in] value Value of the scalar parameter
     * @param[in] type Type of the scalar parameter
     * @return true if scalar parameter has been added successfully; false otherwise
     */
    bool addScalarParam(const char* name, double value, const char* type)
    {
        uint32_t totalParamsSpecified = getTotalParams();

        if (totalParamsSpecified >= m_totalParamsAllowed)
        {
            FI_ERROR("Too many parameters specified in FIUCommand: ", totalParamsSpecified);
            return false;
        }

        if (!name)
        {
            FI_ERROR("Parameter name can't be NULL");
            return false;
        }

        if (strlen(name) >= FIU_MAX_PARAMETER_NAME_SIZE)
        {
            FI_ERROR("Parameter length is too long: ", name);
            return false;
        }

        if (isParamNameFound(name))
        {
            FI_ERROR("Parameter with the specified type not found: ", name);
            return false;
        }

        FaultScalarParameter* fsp = m_objectManager.getScalarParameter(m_faultCommandID);

        if (!fsp)
        {
            FI_ERROR("Unable to allocate memory for scalar param: ", name);
            return false;
        }

        strncpy(fsp->paramName, name, FIU_MAX_PARAMETER_NAME_SIZE);
        strncpy(fsp->demangledTypeName, type, FIU_MAX_FAULT_PARAM_TYPE_DESC);
        fsp->value                                      = value;
        m_fd.scalarArray[m_fd.totalScalarParamsAdded++] = fsp;
        return true;
    }

    /**
     * Function that adds vector parameters
     *
     * @param[in] name Name of the vector parameter
     * @param[in] value Pointer to the value buffer for the vector parameter
     * @param[in] size Size of the value buffer
     * @param[in] type Type of the vector parameter
     * @return true if vector parameter has been added successfully; false otherwise
     */
    bool addVectorParam(const char* name, void* value, size_t size, const char* type)
    {
        uint32_t totalParamsSpecified = getTotalParams();
        if (totalParamsSpecified >= m_totalParamsAllowed)
        {
            FI_ERROR("Too many parameters specified in FIUCommand: ", totalParamsSpecified);
            return false;
        }

        if (!name)
        {
            FI_ERROR("Parameter name can't be NULL");
            return false;
        }

        if (strlen(name) >= FIU_MAX_PARAMETER_NAME_SIZE)
        {
            FI_ERROR("Parameter length is too long: ", name);
            return false;
        }

        if (isParamNameFound(name))
        {
            FI_ERROR("Parameter with the specified type not found: ", name);
            return false;
        }

        FaultVectorParameter* fvp = m_objectManager.getVectorParameter(m_faultCommandID);
        if (!fvp)
        {
            FI_ERROR("Unable to allocate memory for vector param: ", name);
            return false;
        }

        if (!m_objectManager.getMemory(reinterpret_cast<unsigned char**>(&fvp->blobPtr), size, m_faultCommandID, fvp->blobReference))
        {
            FI_ERROR("Unable to get memory for vector blobs: ", size);
            return false;
        }

        strncpy(fvp->paramName, name, FIU_MAX_PARAMETER_NAME_SIZE);
        strncpy(fvp->demangledTypeName, type, FIU_MAX_FAULT_PARAM_TYPE_DESC);
        memcpy(static_cast<void*>(fvp->blobPtr), static_cast<void*>(value), size);

        fvp->blobSize                                   = size;
        m_fd.vectorArray[m_fd.totalVectorParamsAdded++] = fvp;
        return true;
    }

    /**
     * Getter to check if fault can trigger or not
     *
     * @return true if the fault can trigger; else false
     */
    bool doesFrequencyPermitFIUToExecute();

    /**
     * Getter to check if sequence (if defined) is complete
     *
     * @return true if the fault can trigger more times; else false
     */
    bool checkIfSequenceIsComplete();

    /**
     * Function to reset the FIUCommand object
     *
     * @return true if the FIUCommand object has been reset successfully or not
     */
    bool resetObject();

protected:
    // Pointer to the ObjectManager object
    ObjectManager& m_objectManager;

    // Used to keep track of the frequency counters when the user decided to execute
    // this fault using sequence logic. see doesFrequencyPermitFIUToExecute() for details
    SequenceFreqCounters m_sequenceFreq{};

    // Holds the total parameters allowed
    uint8_t m_totalParamsAllowed{TOTAL_PARAMETERS_ALLOWED_PER_FAULT_COMMAND};

    /**
     * Function to check if a given parameter is present in the FIUCommand or not
     *
     * @param[in] name Name of the fault parameter
     * @return true if the parameter is present; else false
     */
    bool isParamNameFound(const char* name)
    {
        // Check for duplicates in scalars
        for (uint8_t index = 0; index < m_fd.totalScalarParamsAdded; index++)
        {
            if (!strcmp(m_fd.scalarArray[index]->paramName, name))
            {
                FI_ERROR("Duplicate Param name found: ", name);
                return true;
            }
        }

        // Check for duplicates in vectors
        for (uint8_t index = 0; index < m_fd.totalVectorParamsAdded; index++)
        {
            if (!strcmp(m_fd.vectorArray[index]->paramName, name))
            {
                FI_ERROR("Duplicate Param name found: ", name);
                return true;
            }
        }
        return false;
    }
};
};

#endif
