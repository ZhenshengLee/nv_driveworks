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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_GRAPH_HEALTH_SIGNAL_H_
#define DW_FRAMEWORK_GRAPH_HEALTH_SIGNAL_H_

#include <dw/core/base/Types.h>
#include <dw/core/health/HealthSignals.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The maximum length of the "source" string for a signal
#define DW_NODE_STATE_MAX_ERROR_STRING_LENGTH 256

/**
 * @brief Basic error signal that gets reported only when there is an error
 **/
typedef struct
{
    // the PID that the node is executing in
    uint32_t processId;
    // Source is a string with the processName.NodeName.PassName format
    // This is set at runtime by the process executing the node.
    char8_t source[DW_NODE_STATE_MAX_ERROR_STRING_LENGTH];

    // the DW error signal
    dwErrorSignal signal;
} dwGraphErrorSignal;

/**
 * @brief Basic health signal that describes the health status of the graph
 **/
typedef struct
{
    // the PID that the node is executing in
    uint32_t processId;
    // Source is a string with the processName.NodeName.PassName format
    // This is set at runtime by the process executing the node.
    char8_t source[DW_NODE_STATE_MAX_ERROR_STRING_LENGTH];

    // The DW health signal
    dwHealthSignal signal;
} dwGraphHealthSignal;

/**
 * @brief Represents an array of health signals
 **/
typedef struct
{
    dwGraphHealthSignal signal[DW_MAX_HEALTH_SIGNAL_ARRAY_SIZE];
    uint32_t count;
} dwGraphHealthSignalArray;

#ifdef __cplusplus
}
#endif

#endif // DW_FRAMEWORK_GRAPH_HEALTH_SIGNAL_H_
