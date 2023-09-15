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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Demuxer</b>
 *
 * @b Description: This file defines DW Demuxer.
 */

/**
 * @defgroup core_demuxers Core Demuxer
 * @brief Demuxer for dealing with containers.
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_SENSORS_CONTAINERS_DEMUXER_H_
#define DW_SENSORS_CONTAINERS_DEMUXER_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/containers/Container.h>
#include <dw/sensors/CodecHeader.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Holds demuxer frame info.
typedef struct dwDemuxerFrameInfo
{
    // The track id.
    uint64_t trackId;
    // The id of the logical bundling of different frames.
    // This is useful for presenting data together that
    // may have different timestamps.
    uint64_t bundleId;
    // The zero-based index representing the order of the frame in the track.
    uint64_t index;
    // The length of the frame payload.
    uint64_t size;
    // The offset of the frame payload in the stream.
    uint64_t offset;
    // Optional flags.
    uint64_t flags;
    // Generic timestamps that can be used
    dwTime_t timestamps[DW_CONTAINER_MAX_TIMESTAMPS];
} dwDemuxerFrameInfo;

/// Demuxer handle.
typedef struct dwDemuxerObject* dwDemuxerHandle_t;

// TODO(lukea) - move this to IOStream.hpp when the C interface moves to core.
/// Handle to IO stream object.
typedef struct dwIOStreamObject* dwIOStreamHandle_t;

/// Holds demuxer params.
typedef struct dwDemuxerParams
{
    /// Handle representing input stream.
    dwIOStreamHandle_t inputStream;
} dwDemuxerParams;

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EXPERIMENTAL_SENSORS_CODEC_DEMUXER_H_
