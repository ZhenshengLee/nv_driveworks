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

#ifndef FIU2_INCLUDES_H
#define FIU2_INCLUDES_H

#include <fiu/impl/FIUClass.hpp>

namespace fiu2
{

#define FI_FIU_INST(A) fiu2::FI_inst_##A
#define FI_FIU_VAR(A) fiu2::FI_var_##A

#define FI_FIU_OBJ(A) fiu2::FI_obj_##A
#define FI_FIU_OBJ_PTR(A) &FI_FIU_OBJ(A)

#define _FI_DECLARE_EXTERN_FAULT(A)   \
    namespace fiu2                    \
    {                                 \
    extern unsigned char FI_var_##A;  \
    extern unsigned char FI_inst_##A; \
    extern FIUClass FI_obj_##A;       \
    }

///////////////////////////////////////////////////////////////////////////////
///////////////////////// APPLY_TO_ALL Functionality //////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define _FI_EVAL_0_(...) __VA_ARGS__
#define _FI_EVAL_1_(...) _FI_EVAL_0_(_FI_EVAL_0_(_FI_EVAL_0_(__VA_ARGS__)))
#define _FI_EVAL_2_(...) _FI_EVAL_1_(_FI_EVAL_1_(_FI_EVAL_1_(__VA_ARGS__)))
#define _FI_EVAL_3_(...) _FI_EVAL_2_(_FI_EVAL_2_(_FI_EVAL_2_(__VA_ARGS__)))
#define _FI_EVAL_4_(...) _FI_EVAL_3_(_FI_EVAL_3_(_FI_EVAL_3_(__VA_ARGS__)))
#define _FI_EVAL(...) _FI_EVAL_4_(_FI_EVAL_4_(_FI_EVAL_4_(__VA_ARGS__)))

#define _FI_APPLY_TO_ALL_END(...)
#define _FI_APPLY_TO_ALL_OUT
#define _FI_APPLY_TO_ALL_COMMA ,

#define _FI_APPLY_TO_ALL_GET_END2() 0, _FI_APPLY_TO_ALL_END
#define _FI_APPLY_TO_ALL_GET_END1(...) _FI_APPLY_TO_ALL_GET_END2
#define _FI_APPLY_TO_ALL_GET_END(...) _FI_APPLY_TO_ALL_GET_END1
#define _FI_APPLY_TO_ALL_NEXT0(test, next, ...) next _FI_APPLY_TO_ALL_OUT
#define _FI_APPLY_TO_ALL_NEXT1(test, next) _FI_APPLY_TO_ALL_NEXT0(test, next, 0)
#define _FI_APPLY_TO_ALL_NEXT(test, next) _FI_APPLY_TO_ALL_NEXT1(_FI_APPLY_TO_ALL_GET_END test, next)

#define MAP0(f, x, peek, ...) f(x) _FI_APPLY_TO_ALL_NEXT(peek, MAP1)(f, peek, __VA_ARGS__)
#define MAP1(f, x, peek, ...) f(x) _FI_APPLY_TO_ALL_NEXT(peek, MAP0)(f, peek, __VA_ARGS__)

#define _FI_APPLY_TO_ALL_LIST_NEXT1(test, next) _FI_APPLY_TO_ALL_NEXT0(test, _FI_APPLY_TO_ALL_COMMA next, 0)
#define _FI_APPLY_TO_ALL_LIST_NEXT(test, next) _FI_APPLY_TO_ALL_LIST_NEXT1(_FI_APPLY_TO_ALL_GET_END test, next)

#define _FI_APPLY_TO_ALL_LIST0(f, x, peek, ...) f(x) _FI_APPLY_TO_ALL_LIST_NEXT(peek, _FI_APPLY_TO_ALL_LIST1)(f, peek, __VA_ARGS__)
#define _FI_APPLY_TO_ALL_LIST1(f, x, peek, ...) f(x) _FI_APPLY_TO_ALL_LIST_NEXT(peek, _FI_APPLY_TO_ALL_LIST0)(f, peek, __VA_ARGS__)

/**
 * Applies the function macro `f` to each of the remaining parameters.
 */
#define _FI_APPLY_TO_ALL(f, ...) _FI_EVAL(MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

/**
 * Applies the function macro `f` to each of the remaining parameters and
 * inserts commas between the results.
 */
#define _FI_APPLY_TO_ALL_LIST(f, ...) _FI_EVAL(_FI_APPLY_TO_ALL_LIST1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Use this environment variable to set the file path that FIUManager can use
// to input the list of faults that need to executed within the run
#define FI_FAULTS_LIST_FILE "FI_FAULTS_LIST_FILE"

/* How to use `APPLY_TO_ALL`*/
// 1. Apply a defined function over all the arguments
//    #define STRING(x) char const *x##_string = #x;
//    APPLY_TO_ALL(STRING, foo, bar, baz)
//
// 2. Apply `_FI_APPLY_TO_ALL_LIST` over function arguments
//    #define PARAM(x) int x
//    void function(_FI_APPLY_TO_ALL_LIST(PARAM, foo, bar, baz));
//////////////////////////////////////////////////////////////////////////////

/// Macro that checks if the fault is enabled or not
#define _IS_ENABLED1(FAULT_NAME) fiu2::FI_var_##FAULT_NAME

/// Macro that checks if the fault instance is enabled or not
#define _IS_ENABLED2(FAULT_NAME, INSTANCE_HANDLE) \
    (FI_FIU_INST(FAULT_NAME) > 0 && FI_IS_INSTANCE_ENABLED(INSTANCE))

/// Helper macro that calls either _IS_ENABLED1 or _IS_ENABLED2
#define _FI_GET_MACRO(_1, _2, NAME, ...) NAME

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

/**
 * Register the string and creates a InstanceStruct data object that can be used
 * to compare/verify/use instances that are created later on
 *
 * @param [in] Any string that can be used to reference an instance [const char *]
 * @param [out] An instance handle that was defined using FI_DECLARE_INSTANCE_SET_HANDLE
 *
 * NOTE: API is Thread safe
 */
#define _FI_SET_INSTANCE_NAME(INSTANCE_SET_HANDLE, INSTANCE_NAME) \
    static_cast<void>(globalAddUniqueInstance(INSTANCE_SET_HANDLE, INSTANCE_NAME));

/** Helper function to add an instance to multiple faults
 *
 *  @param [in] instObj InstanceStruct object
 *  @param [in] faultList List faults linked to the instance set
 */
inline void mapInstanceToFault(InstanceStruct& instObj, std::initializer_list<fiu2::FIUClass*> faultList)
{
    for (auto& name : faultList)
    {
        name->addInstance(instObj.instanceName);
    }
}

/** Helper macro that create instance/fault mapping
 *
 *  @param [in] INSTANCE_SET_HANDLE Instance handler
 *  @param [in] INSTANCE_NAME Name of the instance
 *  @param [in] args... Comma seperated list of fault names
 */
#define _FI_DEFINE_INSTANCE_SET(INSTANCE_SET_HANDLE, INSTANCE_NAME, args...) \
    _FI_SET_INSTANCE_NAME(INSTANCE_SET_HANDLE, INSTANCE_NAME);               \
    fiu2::mapInstanceToFault(INSTANCE_SET_HANDLE, {args});

/// The following three functions have been created to reduce cyclomatic complexity of _FI_CHECK_INSTANCE_COND()

/** Function verifies if the user has enabled the FIU & instance and executes the trigger code
 *
 *  @param [in] fault Pointer to the fault object
 *  @param [in] isInstanceEnabled Flag that specifies if the instance has been enabled or not
 *  @return True/False depending on the AND of the input parameters
 */
static inline bool _checkIsFaultEnabled(fiu2::FaultCommand* fault, bool isInstanceEnabled)
{
    return fault && isInstanceEnabled;
}

/** Function verifies if the given boolean conditions are all TRUE or not
 *
 *  @param [in] isAnyFaultEnabled Flag indicates if any fault is enabled
 *  @param [in] condition Flag indicates if the condition is true or not
 *  @param [in] checkFreq Checks if the frequency parameter lets the fault code to execute
 *  @return True/False depending on the AND of the input parameters
 */
static inline bool _checkIfInstanceConditionIsTrue(const bool& isAnyFaultEnabled, const bool& condition, const bool& checkFreq)
{
    return isAnyFaultEnabled && condition && checkFreq;
}

/** Function verifies if the given boolean conditions are all TRUE or not
 *
 *  @param [in] totalEnabledInstances Total number of instances enabled for the specific fault
 *  @param [in] isAnyFaultEnabled Flag indicates if the fault is enabled
 *  @return True/False depending on the AND of the input parameters
 */
static inline bool _checkInstanceCond(const uint32_t& totalEnabledInstances, const bool& isAnyFaultEnabled)
{
    return totalEnabledInstances > 0U && isAnyFaultEnabled;
}

/** Macro verifies if the user has enabled the FIU and executes the trigger code
 *
 *  @param [in] FAULT_NAME Name of the fault
 *  @param [in] COND Condition that needs to be met inorder for the FIU to trigger
 *  @param [in] TRIGGER_CODE Code that needs to execute if the enabled fault meets the COND
 */
#define _FI_CHECK_COND(FAULT_NAME, COND, TRIGGER_CODE)                                                        \
    {                                                                                                         \
        if (FI_FIU_VAR(ARE_FIUS_ENABLED) && FI_FIU_VAR(FAULT_NAME))                                           \
        {                                                                                                     \
            FI_FIU_OBJ(FAULT_NAME).lockFault();                                                               \
            fiu2::FaultCommand* fault{FI_FIU_OBJ(FAULT_NAME).getFC(__LINE__, __PRETTY_FUNCTION__, __FILE__)}; \
            const bool checkFreq{FI_FIU_OBJ(FAULT_NAME).checkFrequency()};                                    \
            if (FI_FIU_VAR(FAULT_NAME) && fault && COND && checkFreq)                                         \
            {                                                                                                 \
                FI_FIU_OBJ(FAULT_NAME).executeTriggerLogic();                                                 \
                TRIGGER_CODE;                                                                                 \
            }                                                                                                 \
            FI_FIU_OBJ(FAULT_NAME).executeHitLogic();                                                         \
            FI_FIU_OBJ(FAULT_NAME).checkFaultDisableConditions();                                             \
            FI_FIU_OBJ(FAULT_NAME).unLockFault();                                                             \
        }                                                                                                     \
    }

/** Macro verifies if the user has enabled the FIU & instance and executes the trigger code
 *
 *  @param [in] FAULT_NAME Name of the fault
 *  @param [in] INSTANCE Name of the instance
 *  @param [in] COND Condition that needs to be met inorder for the FIU to trigger
 *  @param [in] TRIGGER_CODE Code that needs to execute if the enabled fault meets the COND
 */
#define _FI_CHECK_INSTANCE_COND(FAULT_NAME, INSTANCE, COND, TRIGGER_CODE)                                                   \
    {                                                                                                                       \
        if (FI_FIU_VAR(ARE_FIUS_ENABLED) && fiu2::_checkInstanceCond(FI_FIU_INST(FAULT_NAME),                               \
                                                                     FI_IS_ANY_FAULT_ENABLED(INSTANCE)))                    \
        {                                                                                                                   \
            uint32_t instanceIndexInsideFault{0U};                                                                          \
            FI_FIU_OBJ(FAULT_NAME).lockFault(INSTANCE.instanceIndex);                                                       \
            fiu2::FaultCommand* fault{                                                                                      \
                FI_FIU_OBJ(FAULT_NAME).getFC(INSTANCE, instanceIndexInsideFault, __LINE__, __PRETTY_FUNCTION__, __FILE__)}; \
            if (fiu2::_checkIsFaultEnabled(fault, FI_FIU_OBJ(FAULT_NAME).isEnabled(instanceIndexInsideFault)))              \
            {                                                                                                               \
                const bool checkFreq{                                                                                       \
                    FI_FIU_OBJ(FAULT_NAME).checkFrequency(instanceIndexInsideFault)};                                       \
                if (fiu2::_checkIfInstanceConditionIsTrue(FI_IS_ANY_FAULT_ENABLED(INSTANCE), COND, checkFreq))              \
                {                                                                                                           \
                    FI_FIU_OBJ(FAULT_NAME).executeTriggerLogic(instanceIndexInsideFault);                                   \
                    TRIGGER_CODE;                                                                                           \
                }                                                                                                           \
                FI_FIU_OBJ(FAULT_NAME).executeHitLogic(instanceIndexInsideFault);                                           \
                FI_FIU_OBJ(FAULT_NAME).checkFaultDisableConditions(instanceIndexInsideFault);                               \
            }                                                                                                               \
            FI_FIU_OBJ(FAULT_NAME).unLockFault(INSTANCE.instanceIndex);                                                     \
        }                                                                                                                   \
    }
};

#endif
