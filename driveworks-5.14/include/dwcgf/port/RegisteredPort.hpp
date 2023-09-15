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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_REGISTEREDPORT_HPP_
#define DW_FRAMEWORK_REGISTEREDPORT_HPP_

#include "Port.hpp"

#include <utility>

namespace dw
{
namespace framework
{

struct RegisteredPort
{
    RegisteredPort(size_t portID)
        : m_portID(portID)
    {
    }
    size_t getPortID() const { return m_portID; }

protected:
    size_t m_portID;
};

/**
 * A specialization of PortInput that contains the port id.
 */
// coverity[autosar_cpp14_a14_1_1_violation]
template <typename T>
class RegisteredPortInput : public PortInput<T>, public RegisteredPort
{
public:
    using Base = PortInput<T>;

    template <typename... Args>
    explicit RegisteredPortInput(size_t portID, Args&&... args)
        : Base(std::forward<Args>(args)...)
        , RegisteredPort(portID)
    {
    }
};

/**
 * A specialization of PortOutput that contains the port id.
 */
// coverity[autosar_cpp14_a14_1_1_violation]
template <typename T>
class RegisteredPortOutput : public PortOutput<T>, public RegisteredPort
{
public:
    using Base = PortOutput<T>;

    template <typename... Args>
    RegisteredPortOutput(size_t portID, Args&&... args)
        : Base(std::forward<Args>(args)...)
        , RegisteredPort(portID)
    {
    }
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_REGISTEREDPORT_HPP_
