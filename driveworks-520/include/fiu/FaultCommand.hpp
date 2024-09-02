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

#ifndef FAULT_COMMAND_H
#define FAULT_COMMAND_H
#include <fiu/impl/FIUCommon.hpp>

namespace fiu2
{

class FaultCommand
{
public:
    virtual ~FaultCommand(){};

    /**
     * Constructor
     *
     */
    FaultCommand()
    {
        m_faultCommandID = FaultCommand::m_globalFaultCommandID++;
    }

    /**
     * Setter for the log level
     *
     * @param[in] &lgLevel inputs log level
     */
    void setLog(FaultLogLevel lgLevel)
    {
        m_logLevel = lgLevel;
    }

    /**
     * Setter for the log level
     *
     * @return returns the log level
     */
    FaultLogLevel getLog()
    {
        return m_logLevel;
    }

    /**
     * Template Function that returns the value of the externally provided scalar parameter
     *
     * @param[out] &var Reference to which the value of the external scalar can be copied to
     * @param[in] *externalVarName Name of the external variable
     * @return true if scalar parameter is found; false otherwise
     */
    template <typename T>
    bool getScalar(T& var, const char* externalVarName)
    {
        const char* demangledTypeName = getDemangledName(typeid(T).name());
        for (int index = 0; index < m_fd.totalScalarParamsAdded; index++)
        {
            if (!strcmp(m_fd.scalarArray[index]->paramName, externalVarName))
            {
                if (strcmp(m_fd.scalarArray[index]->demangledTypeName, demangledTypeName))
                {
                    FI_ERROR("Invalid type query: ", demangledTypeName, "  ", m_fd.scalarArray[index]->demangledTypeName);
                    return false;
                }
                var = static_cast<T>(m_fd.scalarArray[index]->value);
                return true;
            }
        }
        return false;
    }

    /**
     * Template Function that returns the pointer to the externally provided data blob
     *
     * @param[out] *var Pointer reference to which the address of the buffer needs to be copied
     * @param[in] *externalVarName Name of the external variable
     * @param[in-out] &size Pass size appropriate to the buffer; API returns the size it copied to
     *                      if it has copied less than the allocated size. API does NOT copy more
     *                      data that the var object can hold.
     *                      API returns the total memory it has copied to.
     * @return true if vector parameter is found; false otherwise
     *
     * NOTE: Callers are expected to allocate sufficient memory
     */
    template <typename T>
    bool getVector(T var, const char* externalVarName, size_t& size)
    {
        if (size == 0)
        {
            FI_ERROR("Buffer to copy has size 0; provoide a proper buffer and size to copy the vector");
            return false;
        }
        const char* demangledTypeName = getDemangledName(typeid(T).name());

        if (!demangledTypeName)
        {
            FI_ERROR("Invalid Type: ", typeid(T).name());
            return false;
        }
        for (int index = 0; index < m_fd.totalVectorParamsAdded; index++)
        {
            if (!strcmp(m_fd.vectorArray[index]->paramName, externalVarName))
            {
                if (strncmp(m_fd.vectorArray[index]->demangledTypeName, demangledTypeName, 2))
                {
                    FI_ERROR("Invalid type query: ", demangledTypeName, "  ", m_fd.vectorArray[index]->demangledTypeName);
                    return false;
                }
                if (m_fd.vectorArray[index]->blobSize < size)
                {
                    size = m_fd.vectorArray[index]->blobSize;
                }

                memcpy(static_cast<void*>(var), m_fd.vectorArray[index]->blobPtr, size);
                //size = m_fd.vectorArray[index]->blobSize;
                return true;
            }
        }
        return false;
    }

    /**
     * Template Function that returns if the condition is true or not
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    bool isLessThan(T variable, const char* variableName)
    {
        T var;
        if (getScalar(var, variableName))
        {
            return (variable < var);
        }
        return false;
    }

    /**
     * Template Function that returns if the condition is true or not
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    bool isLessThanOrEqual(T variable, const char* variableName)
    {
        T var;
        if (getScalar(var, variableName))
        {
            return (variable <= var);
        }
        return false;
    }

    /**
     * Template Function that returns if the condition is true or not
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    bool isGreaterThan(T variable, const char* variableName)
    {
        T var;
        if (getScalar(var, variableName))
        {
            return (variable > var);
        }
        return false;
    }

    /**
     * Template Function that returns if the condition is true or not
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    bool isGreaterThanOrEqual(T variable, const char* variableName)
    {
        T var;
        if (getScalar(var, variableName))
        {
            return (variable >= var);
        }
        return false;
    }

    /**
     * Template Function that returns if the condition is true or not
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    bool isEqual(T variable, const char* variableName)
    {
        T var;
        if (getScalar(var, variableName))
        {
            return (variable == var);
        }
        return false;
    }

    /**
     * Template Function that logs the data
     *
     * @param[in] variable Variable that needs to compared against
     * @param[in] *variableName Name of the scalar variable that needs to be compared against
     * @return true if condition results in true
     */
    template <typename T>
    void log(const char* tag, T variable)
    {
        FI_INFO(tag, variable);
    }

    /**
     * Returns the total Parameters available
     *
     * @return total parameters (scalars + vectors)
     */
    uint32_t getTotalParams()
    {
        return static_cast<uint32_t>(m_fd.totalScalarParamsAdded) + static_cast<uint32_t>(m_fd.totalVectorParamsAdded);
    }

    /**
     * Returns the fault description pointer
     *
     * @return Pointer to the fault description
     */
    const FaultDescription* getFaultDescription()
    {
        return &m_fd;
    }

    /**
     * Returns the pointer to the fault name
     *
     * @return Pointer to the fault name
     */
    const char* getFaultName()
    {
        return m_fd.faultName;
    }

    /**
     * Returns the pointer to the instance name
     *
     * @return Pointer to the instance name
     */
    const char* getInstanceName()
    {
        return m_fd.instanceName;
    }

protected:
    // Log level for this fault
    FaultLogLevel m_logLevel{FaultLogLevel::INFO};

    // Fault Command structure
    FaultDescription m_fd{};

    // Static int to keep track of all the fault commands allocated in the system
    static uint32_t m_globalFaultCommandID;

    // ID of the current object; derived from m_globalFaultCommandID
    uint32_t m_faultCommandID{};

    // This flag will be set in production. This tells FIUClass that this object
    // can be reused after a fault is disabled. When this flag is not set to true
    // FIUClass does not attempt to pass this object to the FIUManager for reuse.
    //
    // Leaving this flag as false helps us in certain tests where we don't want to
    // have a FIUCommand object reused.
    bool m_reuseFlag{false};
};
};

#endif
