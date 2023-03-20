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
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_SAFETY_SAFETYEXCEPTION_HPP_
#define DW_CORE_SAFETY_SAFETYEXCEPTION_HPP_

#include <dw/core/base/ExceptionWithStackTrace.hpp>

namespace dw
{
namespace core
{

// Safety-specific result type indicating the reason / issue type for it's (potential) empty state
class BadSafetyResultAccess : public ExceptionWithStackTrace
{
public:
    // TODO(dwplc): FP - Declaring a forwarding constructor that is constrained (via SFINAE) to not match any other overloads
    // coverity[autosar_cpp14_a13_3_1_violation]
    using ExceptionWithStackTrace::ExceptionWithStackTrace;

    BadSafetyResultAccess(const BadSafetyResultAccess&) = default;
    BadSafetyResultAccess(BadSafetyResultAccess&&)      = default;

    BadSafetyResultAccess& operator=(const BadSafetyResultAccess&) = default;
    BadSafetyResultAccess& operator=(BadSafetyResultAccess&&) = default;

    ~BadSafetyResultAccess() override = default;
};

} // core
} // dw

#endif // DWSHARED_CORE_SAFETY_SAFETYEXCEPTION_HPP_
