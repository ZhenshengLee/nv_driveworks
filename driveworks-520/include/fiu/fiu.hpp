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

#ifndef FIU2_H
#define FIU2_H

#include <fiu/impl/FIUClass.hpp>
#include <fiu/impl/FIUIncludes.hpp>
#include <fiu/impl/JSONParser.hpp>
#include <fiu/impl/FIUManager.hpp>

/**
 * @file
 * This header provides the public API of FI.
 *
 * The FI tool supports three different usage patterns:
 *   - Native Faults
 *   - Single Instance Faults
 *   - Multi Instance Faults
 *
 *
 * APIs for Native Faults
 * ======================
 *      Step 1: Define the fault      [Location: Outside Function Scope]
 *            FI_DEFINE_FAULT( <fault name>,
 *                             <module>,
 *                             <brief desc>
 *                           );
 *
 *            FI_DEFINE_FAULT_WITH_PARAMS( <fault name>,
 *                                         <module>,
 *                                         <brief desc>,
 *                                         <fault parameters>,
 *                                       );
 *
 *      Step 2: Check the faults      [Location: Inside Runtime Functions]
 *            FI_CHECK(<fault name>, <fault code>);
 *            FI_CHECK_COND(<fault name>, <fault condition>, <fault code>);
 *
 *
 * Single Instance Faults
 * ======================
 *      Step 1: Define the fault      [Location: Outside Function Scope]
 *            FI_DEFINE_FAULT( <fault name>,
 *                             <module>,
 *                             <brief desc>
 *                           );
 *
 *            FI_DEFINE_FAULT_WITH_PARAMS( <fault name>,
 *                                         <module>,
 *                                         <brief desc>,
 *                                         <fault parameters>,
 *                                       );
 *
 *      Step 2: Declare the Instance Handler     [Location: Outside Function Scope <OR> Inside Class Declaration]
 *            FI_DECLARE_INSTANCE_HANDLE(<instance handle name>);
 *
 *      Step 3: Define the instance by associating the instance handler with the instance name and the fault
 *              [Location: Class Construction/Initialization]
 *            FI_DEFINE_INSTANCE(<instance handle name>, <instance name as a string>, <name of the fault>);
 *
 *      Step 4: Check the faults      [Location: Inside Runtime Functions]
 *            FI_CHECK(<fault name>, <fault code>);
 *            FI_CHECK_COND(<fault name>, <fault condition>, <fault code>);
 *
 *
 * Multi Instance Faults
 * =====================
 *
 *      Step 1: Define the fault      [Location: Outside Function Scope]
 *            FI_DEFINE_FAULT( <fault name>,
 *                             <module>,
 *                             <brief desc>
 *                           );
 *
 *            FI_DEFINE_FAULT_WITH_PARAMS( <fault name>,
 *                                         <module>,
 *                                         <brief desc>,
 *                                         <fault parameters>,
 *                                       );
 *
 *      Step 2: Declare the Instance Handler   [Location: Outside Function Scope <OR> Inside Class Declaration]
 *            FI_DECLARE_INSTANCE_SET_HANDLE(<instance handle name>);
 *
 *      Step 3: Define the instance by associating the instance handler with the name and the fault
 *              [Location: Class Construction/Initialization]
 *            FI_DEFINE_INSTANCE_SET(<instance handle name>, <instance name as a string>, <name of the fault>);
 *
 *      Step 4: Check the faults      [Location: Inside Runtime Functions]
 *            FI_CHECK(<fault name>, <fault code>);
 *            FI_CHECK_COND(<fault name>, <fault condition>, <fault code>);
 *
 */

namespace fiu2
{

#define FI_PARSE_FIU_FILE(filename) fiu2::globalParseFIUFile(filename)

#define FI_PARAMS(args...) \
    {                      \
        args               \
    }
#define FI_KEYVAL(args...) \
    {                      \
        args               \
    }

/**
 * Declares that a fault is external to the current file. There are situations where
 * developers need to access a fault defined in another C++ file. In such cases, the
 * following API can be used to declare the fault, ensuring that the compiler recognizes
 * the fault type and compiles the fault-related APIs accordingly.
 *
 * @param[in] name Name of the fault
 */
#define FI_DECLARE_EXTERN_FAULT(name) _FI_DECLARE_EXTERN_FAULT(name)

/**
 * Defines a sample fault that doesn't input any parameters
 *
 * @param[in] name Name of the fault
 * @param[in] module Name of the module this fault belongs to
 * @param[in] synopsys Brief description of the fault
 */
#define FI_DEFINE_FAULT(name, module, synopsys)                      \
    namespace fiu2                                                   \
    {                                                                \
    uint8_t __attribute((section("fiumap"))) FI_var_##name{0U};      \
    uint8_t __attribute((section("fiuinstmap"))) FI_inst_##name{0U}; \
    FIUClass FI_obj_##name{FI_var_##name,                            \
                           FI_inst_##name,                           \
                           #name,                                    \
                           module,                                   \
                           synopsys,                                 \
                           FI_PARAMS(),                              \
                           FI_KEYVAL()};                             \
    };

/**
 * Defines a sample fault that doesn't input any parameters
 *
 * @param[in] name Name of the fault
 * @param[in] module Name of the module this fault belongs to
 * @param[in] synopsys Brief description of the fault
 * @param[in] faultParam Fault Parameters initializer list
 */
#define FI_DEFINE_FAULT_WITH_PARAMS(name, module, synopsys, faultParam) \
    namespace fiu2                                                      \
    {                                                                   \
    uint8_t __attribute((section("fiumap"))) FI_var_##name{0U};         \
    uint8_t __attribute((section("fiuinstmap"))) FI_inst_##name{0U};    \
    FIUClass FI_obj_##name{FI_var_##name,                               \
                           FI_inst_##name,                              \
                           #name,                                       \
                           module,                                      \
                           synopsys,                                    \
                           faultParam,                                  \
                           FI_KEYVAL()};                                \
    };

#define FI_PARAMETER_UINT8(name) FI_PARAMETER(name, true, "unsigned char")
#define FI_PARAMETER_INT8(name) FI_PARAMETER(name, true, "signed char")
#define FI_PARAMETER_UINT16(name) FI_PARAMETER(name, true, "unsigned short")
#define FI_PARAMETER_INT16(name) FI_PARAMETER(name, true, "short")
#define FI_PARAMETER_UINT32(name) FI_PARAMETER(name, true, "unsigned int")
#define FI_PARAMETER_INT32(name) FI_PARAMETER(name, true, "int")
#define FI_PARAMETER_UINT64(name) FI_PARAMETER(name, true, "unsigned long")
#define FI_PARAMETER_INT64(name) FI_PARAMETER(name, true, "long")
#define FI_PARAMETER_FLOAT(name) FI_PARAMETER(name, true, "float")
#define FI_PARAMETER_DOUBLE(name) FI_PARAMETER(name, true, "double")
#define FI_PARAMETER_CHAR(name) FI_PARAMETER(name, true, "char")
#define FI_PARAMETER_BOOL(name) FI_PARAMETER(name, true, "bool")
#define FI_PARAMETER_UINT8_VECTOR(name) FI_PARAMETER(name, true, "unsigned char*")
#define FI_PARAMETER_INT8_VECTOR(name) FI_PARAMETER(name, true, "signed char*")
#define FI_PARAMETER_UINT16_VECTOR(name) FI_PARAMETER(name, true, "unsigned short*")
#define FI_PARAMETER_INT16_VECTOR(name) FI_PARAMETER(name, true, "short*")
#define FI_PARAMETER_UINT32_VECTOR(name) FI_PARAMETER(name, true, "unsigned int*")
#define FI_PARAMETER_INT32_VECTOR(name) FI_PARAMETER(name, true, "int*")
#define FI_PARAMETER_UINT64_VECTOR(name) FI_PARAMETER(name, true, "unsigned long*")
#define FI_PARAMETER_INT64_VECTOR(name) FI_PARAMETER(name, true, "long*")
#define FI_PARAMETER_FLOAT_VECTOR(name) FI_PARAMETER(name, true, "float*")
#define FI_PARAMETER_DOUBLE_VECTOR(name) FI_PARAMETER(name, true, "double*")
#define FI_PARAMETER_CHAR_VECTOR(name) FI_PARAMETER(name, true, "char*")
#define FI_PARAMETER_BOOL_VECTOR(name) FI_PARAMETER(name, true, "bool*")

#define FI_PARAMETER(name, isRequired, type) \
    {                                        \
        #name, isRequired, type              \
    }
#define FI_KV(key, value) \
    {                     \
        #key, value       \
    }

/** \addtogroup InstanceFault Tuple APIs
 *  @{
 */

/**@{*/
/** Tuple Declaration APIs {Instance, Fault} */
/**
 * Define an instance handle to create a fault/instance tuple
 *
 * @param [in] variable name
 * e.g.,
 *       FI_DECLARE_INSTANCE_HANDLE(i1);
 * NOTE: API is Thread safe
 */
#define FI_DECLARE_INSTANCE_HANDLE(INSTANCE_HANDLE) \
    fiu2::InstanceStruct INSTANCE_HANDLE;

/**
 * Create multiple instance handles at ones; an extension to the FI_DECLARE_INSTANCE_HANDLE api
 *
 * @param [in] multiple variable names
 * e.g.,
 *       FI_DECLARE_INSTANCE_HANDLE_LIST(i1, i2, i3, i4, i5);
 * NOTE: API is Thread safe
 */
#define FI_DECLARE_INSTANCE_HANDLE_LIST(args...) \
    _FI_APPLY_TO_ALL(FI_DECLARE_INSTANCE_HANDLE, args);
/**@}*/

/**@{*/
/** Instance Definition APIs */

/**
 * Create fault/instance tuple and assign it to the data object.
 *
 * NOTE: This needs to be used for normal baseclass/derived class faults
 * NOTE: API is Thread safe
 */
#define FI_DEFINE_INSTANCE(INSTANCE_HANDLE, INSTANCE_NAME, FAULT) \
    FI_FIU_VAR(IS_FI_LIB_ENABLED) && FI_FIU_OBJ(FAULT).addInstance(INSTANCE_HANDLE, INSTANCE_NAME)
/**@}*/

/**@{*/
/** Handy APIs to check if fault/insance is enabled or not */
/**
 * Check if the fault or a fault instance is enabled
 *
 * @param [in] Fault Name
 * @param [in] Instance Handle
 *
 * Usage:
 *      FI_IS_FAULT_ENABLED(FIU1);
 *      FI_IS_FAULT_ENABLED(FIU1, instanceHandle);
 * NOTE: API is Thread safe
 */
#define FI_IS_FAULT_ENABLED(...) FI_FIU_VAR(IS_FI_LIB_ENABLED) && _FI_GET_MACRO(__VA_ARGS__, _IS_ENABLED2, _IS_ENABLED1)(__VA_ARGS__)
/**@}*/

/**@}*/

/** \addtogroup InstanceFaultSet APIs
 *  @{
 */

/**@{*/
/** Set Declaration APIs {Instance, {Fault list}} */

/**
 * Define an instance handle that is associated with a set of faults
 *
 * @param [in] variable name
 * e.g.,
 *       FI_DECLARE_INSTANCE_SET_HANDLE(i1);
 * NOTE: API is Thread safe
 */
#define FI_DECLARE_INSTANCE_SET_HANDLE(INSTANCE_SET_HANDLE) fiu2::InstanceStruct INSTANCE_SET_HANDLE;
/**@}*/

/**@{*/
/** APIs to create Instances */

/**
 * This API to create fault/instance pairs
 *
 * NOTE: This needs to be used in cases where a single instance can open paths to several faults
 *       implemented in handle functions
 *
 * NOTE: API is Thread safe
 */
#define FI_DEFINE_INSTANCE_SET(INSTANCE_SET_HANDLE, INSTANCE_NAME, FAULT_ARGS...)                                       \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                                                                                  \
    {                                                                                                                   \
        _FI_DEFINE_INSTANCE_SET(INSTANCE_SET_HANDLE, INSTANCE_NAME, _FI_APPLY_TO_ALL_LIST(FI_FIU_OBJ_PTR, FAULT_ARGS)); \
    }

/**@}*/
/**@}*/

/** \addtogroup Fault Verificaiton Execution APIs
 *  @{
 */

/**
 * Check if a global instance is enabled or not
 *
 * NOTE: API is lockless
 */
#define FI_IS_ANY_FAULT_ENABLED(INSTANCE_SET_HANDLE) \
    (FI_FIU_VAR(IS_FI_LIB_ENABLED) &&                \
     INSTANCE_SET_HANDLE.isInstanceEnabled &&        \
     fiu2::getBit(INSTANCE_SET_HANDLE.isInstanceEnabled, INSTANCE_SET_HANDLE.instanceIndex))

/**
 * Check if there are faults enabled in the system and then verify the specific instance is
 * enabled for any of those enabled faults. Execute the fault code if enabled
 *
 * @param [in] Instance Handle that uses the FI_SET_INSTANCE_NAME only
 * @param [in] Fault code that needs to execute if the conditions are met
 *
 * NOTE: API is Thread safe
 */
#define FI_EXECUTE_IF_ANY_FAULT_IS_ENABLED(INSTANCE_SET_HANDLE, FAULT_CODE)                         \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED) &&                                                            \
        FI_FIU_VAR(ARE_FIUS_ENABLED) &&                                                             \
        INSTANCE_SET_HANDLE.isInstanceEnabled)                                                      \
    {                                                                                               \
        if (fiu2::getBit(INSTANCE_SET_HANDLE.isInstanceEnabled, INSTANCE_SET_HANDLE.instanceIndex)) \
        {                                                                                           \
            FAULT_CODE;                                                                             \
        }                                                                                           \
    }

/**
 * Execute the fault code if the specific fault is enabled
 *
 * @param [in] Fault name that needs to be enabled
 * @param [in] Fault code that needs to execute if the fault is enabled
 *
 * NOTE: API is Thread safe
 */
#define FI_CHECK(FAULT_NAME, FAULT_CODE)              \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                \
    {                                                 \
        _FI_CHECK_COND(FAULT_NAME, true, FAULT_CODE); \
    }

/**
 * Execute the fault code if the specific fault is enabled and if the condition is met
 *
 * @param [in] Fault name that needs to be enabled
 * @param [in] Condition code that needs to return true
 * @param [in] Fault code that needs to execute if the fault is enabled
 *
 * e.g.,
 *      FI_CHECK_COND(FIU6, fault->isGreaterThanOrEqual(index, "index"), count++);
 *
 * NOTE: API is Thread safe
 */
#define FI_CHECK_COND(FAULT_NAME, CONDITION, FAULT_CODE)   \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                     \
    {                                                      \
        _FI_CHECK_COND(FAULT_NAME, CONDITION, FAULT_CODE); \
    }

/**
 * Execute the fault code if the specific fault's instance is enabled
 *
 * @param [in] Fault name that needs to verified against
 * @param [in] Instance handle for the specific fault that needs to verified against
 * @param [in] Fault code that needs to execute if the fault's instance is enabled
 *
 * NOTE: API is Thread safe
 */
#define FI_CHECK_INSTANCE(FAULT_NAME, INSTANCE, FAULT_CODE)              \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                                   \
    {                                                                    \
        _FI_CHECK_INSTANCE_COND(FAULT_NAME, INSTANCE, true, FAULT_CODE); \
    }

/**
 * Execute the fault code if the specific fault's instance is enabled and when the condition
 * is met
 *
 * @param [in] Fault name that needs to verified against
 * @param [in] Instance handle for the specific fault that needs to verified against
 * @param [in] Condition code that needs to return true
 * @param [in] Fault code that needs to execute if the fault's instance is enabled
 *
 * E.g.,
 *      FI_CHECK_INSTANCE_COND(FET_FIU6, inst1, fault->isGreaterThan(index, "index"), count++);
 *
 * NOTE: API is Thread safe
 */
#define FI_CHECK_INSTANCE_COND(FAULT_NAME, INSTANCE, CONDITION, FAULT_CODE)   \
    if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                                        \
    {                                                                         \
        _FI_CHECK_INSTANCE_COND(FAULT_NAME, INSTANCE, CONDITION, FAULT_CODE); \
    }

/**@}*/
};

#endif
