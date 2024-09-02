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

#ifndef FIU2_FAULT_REGISTRY_HPP_
#define FIU2_FAULT_REGISTRY_HPP_

#include <functional>
#include <list>
#include <typeindex>
#include <fiu/fiu.hpp>

namespace fiu2
{

using RegistryFunction = std::function<void(InstanceStruct&, void*)>;

const std::list<RegistryFunction>& registryLookup(std::type_index typeIndex);

template <typename ArgT>
void registerFunction(void (*func)(InstanceStruct&, ArgT*))
{
    // Return if the FI library is disabled
    if (!FI_FIU_VAR_INTERNAL(IS_FI_LIB_ENABLED))
    {
        return;
    }
    auto generic_func = [func](fiu2::InstanceStruct& instance, void* data) {
        func(instance, static_cast<ArgT*>(data));
    };
    registerFunction(std::type_index(typeid(ArgT)), generic_func);
}

void registerFunction(std::type_index typeIndex, RegistryFunction func);

void registerHandleWithFaults(const char* name, InstanceStruct& instance, std::type_index typeIndex);

void addRegisteredFaultsToIndex(std::type_index typeIndex, std::list<fiu2::FIUClass*> faults);

} // namespace fiu2

#define FI_CHECK_REGISTRY(instance, typeIndex, pointer)              \
    {                                                                \
        if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                           \
        {                                                            \
            for (const auto& func : fiu2::registryLookup(typeIndex)) \
            {                                                        \
                func(instance, pointer);                             \
            }                                                        \
        }                                                            \
    }

#define FI_REGISTER_SET_INSTANCE_NAME_AND_TYPE(name, instance, typeIndex) \
    {                                                                     \
        if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                                \
        {                                                                 \
            _FI_SET_INSTANCE_NAME(instance, name);                        \
            fiu2::registerHandleWithFaults(name, instance, typeIndex);    \
        }                                                                 \
    }

#define FI_DEFINE_REGISTERED_FAULTS(argType, args...)                                                                                                                  \
    {                                                                                                                                                                  \
        if (FI_FIU_VAR(IS_FI_LIB_ENABLED))                                                                                                                             \
        {                                                                                                                                                              \
            fiu2::addRegisteredFaultsToIndex(std::type_index(typeid(argType)), std::initializer_list<fiu2::FIUClass*>({_FI_APPLY_TO_ALL_LIST(FI_FIU_OBJ_PTR, args)})); \
        }                                                                                                                                                              \
    }

#define FI_REGISTRY_PRESENT

#endif // FIU2_FAULT_REGISTRY_HPP_
