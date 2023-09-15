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
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_IVULKAN_NODE_HPP_
#define DW_FRAMEWORK_IVULKAN_NODE_HPP_

#if VIBRANTE_PDK_DECIMAL >= 6000500
#include <VulkanSC/vulkan/vulkan_sc.h>
#endif

namespace dw
{
namespace framework
{

/**
 * Nodes that want to use Vulkan hardware resource should implement this interface.
 * After the node has been instantiated the framework will call the getter functions to
 * get the Vulkan resource created in the node and register with the scheduler.
 *
 * Restrictions by scheduler to use Vulkan
 *  a. Only one node per process can use the Vulkan resource (i.e this Vulkan Node Interface).
 *  b. The node must be instantiated as part of the first schedule being run.
 */
// coverity[autosar_cpp14_a0_1_6_violation]
class IVulkanNode
{
public:
    virtual ~IVulkanNode() = default;

#if VIBRANTE_PDK_DECIMAL >= 6000500
    /**
     * @brief Gets the VkInstance
     * @param[out] instance reference to the created VkInstance
     * @return DW_SUCCESS if successfully, failure code otherwise
     */
    virtual dwStatus getVkInstance(VkInstance& instance) = 0;

    /**
     * @brief Gets the VkPhysicalDevice
     * @param[out] device reference to the created VkPhysicalDevice
     * @return DW_SUCCESS if successfully, failure code otherwise
     */
    virtual dwStatus getVkPhysicalDevice(VkPhysicalDevice& device) = 0;

    /**
     * @brief Gets the VkDevice
     * @param[out] device reference to the created VkDevice
     * @return DW_SUCCESS if successfully, failure code otherwise
     */
    virtual dwStatus getVkDevice(VkDevice& device) = 0;

    /**
     * @brief Gets the VkQueue
     * @param[out] queue reference to the created VkQueue
     * @return DW_SUCCESS if successfully, failure code otherwise
     */
    virtual dwStatus getVkQueue(VkQueue& queue) = 0;
#endif // VIBRANTE_PDK_DECIMAL >= 6000500
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_IVULKAN_NODE_HPP_
