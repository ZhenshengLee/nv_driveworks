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
// SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_BASE_NODE_NODECONCEPT_HPP_
#define DW_FRAMEWORK_BASE_NODE_NODECONCEPT_HPP_

#ifndef NODE_CONCEPT_ONLY_ALLOW_FOR_COVERAGE
#error "This file shouldn't be included. It only exists for documentation purpose."
#endif

#include <dwcgf/parameter/ParameterProvider.hpp>

#include <memory>

namespace dw
{
namespace framework
{

// forward declaration for the return type of the factory function create()
class NodeT;

/**
 * @brief The class only exists for documentation.
 *
 * A node registered at the node factory with DW_REGISTER_NODE() must have the static functions declared in this class.
 */
// coverity[autosar_cpp14_a0_1_6_violation]
class NodeConcept
{
public:
    /**
     * @brief Create an instance of the node.
     *
     * Constructor arguments are being populated with parameters values as declared and mapped by declareParameters().
     *
     * @param provider to retrieve parameter values from
     * @return a unique pointer to the newly created node instance, NodeT should be substituted with the class name this function is defined in
     */
    static std::unique_ptr<NodeT> create(ParameterProvider& provider);

    /**
     * @brief Describe the input ports of the node.
     *
     * @return A tuple created by describePortCollection()
     */
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts();

    /**
     * @brief Describe the output ports of the node.
     *
     * @return A tuple created by describePortCollection()
     */
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts();

    /**
     * @brief Describe the parameters of the node and their mapping to the constructor arguments.
     *
     * @return A tuple created by describeConstructorArguments()
     */
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters();

    /**
     * @brief Describe the passes of the node.
     *
     * @return A tuple created by describePassCollection()
     */
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses();
};

} // namespace framework
} // namespace dw

#endif //DW_FRAMEWORK_BASE_NODE_NODECONCEPT_HPP_
