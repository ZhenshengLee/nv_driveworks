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
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_IVIC_NODE_HPP_
#define DW_FRAMEWORK_IVIC_NODE_HPP_

#if VIBRANTE_PDK_DECIMAL >= 6000600
#include <nvmedia_6x/nvmedia_ldc.h>
#include <nvmedia_6x/nvmedia_2d.h>
#endif

#include <src/dwcgf/core/dwcgf/pass/Pass.hpp>

namespace dw
{
namespace framework
{
/**
 * Nodes that want to use VIC hardware resources (LDC & Nv2D) should implement these interface.
 * For each execution cycle,
 * a. STM will provide new pre-fences and the runtime framework will call the
 *    setter function to pass the pre-fences to the node before executing the pass.
 * b. And after the pass is executed, the runtime framework call the getter to get the
 *    post-fence from the node.
 *
 * Requirements by scheduler to use VIC
 *  a. scheduler supports 1 LDC handle and 1 Nv2D handle. The handle is created and registered by
 *     runtime framework, and it can be shared by multiple Nodes.
 *  b. One or more nodes can have any number or combination of VIC passes type LDC/Nv2D.
 */

// coverity[autosar_cpp14_a0_1_6_violation]
class IVicLdcNode
{
public:
    virtual ~IVicLdcNode() = default;

#if VIBRANTE_PDK_DECIMAL >= 6000600
    /**
     * @brief setter for LDC pre-fences. The scheduler provides the app nodes with the pre-fences to submit.
     *
     * @param[in] pass runnable for which this function is called
     * @param[in] ldc LDC handle created by framework
     * @param[in] ldcPrefenceList List of prefences the node needs to insert
     * @param[in] ldcPrefenceCount Count of prefences that need to be inserted
     * @param[in] ldcPostfencesyncObj Sync obj on which NvMediaLdcSetNvSciSyncObjforEOF needs to be called
     */
    virtual void setLdcPreFences(const Pass* pass, NvMediaLdc* ldc, NvSciSyncFence const ldcPrefenceList[], size_t ldcPrefenceCount, NvSciSyncObj ldcPostfencesyncObj) = 0;

    /**
     * @brief Get LDC  post-fence. After the workload has been submitted the node should provide scheduler with the post-fence.
     *
     * @param[in] pass Pointer to any data needed by the runnable
     * @param[in] ldc LDC handle created by framework
     * @param[out] ldcPostfence Pointer to post fence which the node needs to provide stm with
     */
    virtual void getLdcPostFence(const Pass* pass, NvMediaLdc* ldc, NvSciSyncFence* ldcPostfence) = 0;
#endif // VIBRANTE_PDK_DECIMAL >= 6000600
};     // class IVicLdcNode

// coverity[autosar_cpp14_a0_1_6_violation]
class IVicNv2DNode
{
public:
    virtual ~IVicNv2DNode() = default;

#if VIBRANTE_PDK_DECIMAL >= 6000600
    /**
     * @brief setter for Nv2D pre-fences. The scheduler provides the app nodes with the pre-fences to submit.
     *
     * @param[in] pass runnable for which this function is called
     * @param[in] nv2D NVMEDIA_2D handle created by framework
     * @param[in] nv2DPrefenceList List of prefences the node needs to insert
     * @param[in] nv2DPrefenceCount Count of prefences that need to be inserted
     * @param[in] nv2DPostfencesyncObj Sync obj on which NvMedia2DSetNvSciSyncObjforEOF needs to be called by the node
     */
    virtual void set2DPreFences(const Pass* pass, NvMedia2D* nv2D, NvSciSyncFence const nv2DPrefenceList[], size_t nv2DPrefenceCount, NvSciSyncObj nv2DPostfencesyncObj) = 0;

    /**
     * @brief getter for Nv2D post-fence. After the workload has been submitted the node should provide scheduler with the post-fence.
     *
     * @param[in] pass runnable for which this function is called
     * @param[in] nv2D NVMEDIA_2D handle created by framework
     * @param[out] nv2DPostfence Pointer to post fence which the node needs to provide stm with
    */
    virtual void get2DPostFence(const Pass* pass, NvMedia2D* nv2D, NvSciSyncFence* nv2DPostfence) = 0;
#endif // VIBRANTE_PDK_DECIMAL >= 6000600
};     // class IVicNv2DNode

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_IVIC_NODE_HPP_
