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
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_SENSORS_CONTAINERS_CONTAINER_H_
#define DW_SENSORS_CONTAINERS_CONTAINER_H_

#include <dw/core/base/Types.h>
#include <dw/sensors/codecs/Codec.h>
#include <dw/sensors/containers/Metadata.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DW_CONTAINER_MAX_TIMESTAMPS 8U
#define DW_CONTAINER_MAX_META_TYPES 8U

/// Enum representing a supported time domains.
typedef enum {
    DW_TIME_TYPE_NONE = 0,        //!< None time type
    DW_TIME_TYPE_HOST,            //!< Host time
    DW_TIME_TYPE_TSC,             //!< TSC time
    DW_TIME_TYPE_RELATIVE_PTS,    //!< PTS time
    DW_TIME_TYPE_RELATIVE_DTS,    //!< DTS time
    DW_TIME_TYPE_ABSOLUTE,        //!< Absolute time
    DW_TIME_TYPE_SERIALIZATION,   //!< Serialization time
    DW_TIME_TYPE_SEQUENCE_NUMBER, //!< Sequence Number
    DW_TIME_TYPE_POS_OFFSET       //!< Postition Offset
} dwTimeType;

/// Holds container track information
typedef struct dwContainerTrackInfo
{
    dwMediaType mediaType;                             //!< Media type defined in Codec.h.
    uint32_t timeTypesCount;                           //!< Number of time types.
    dwTimeType timeTypes[DW_CONTAINER_MAX_TIMESTAMPS]; //!< Array of time types.
    void const* metadata;                              //!< Serialized track metadata.
} dwContainerTrackInfo;

/// Enum representing a container frame flags.
typedef enum dwContainerFrameFlags {
    DW_CONTAINER_FRAME_SEEKABLE = 1 << 0, //!< This frame represents the beginning of a sequence resulting in a decodable frame.
    DW_CONTAINER_FRAME_COMPLETE = 1 << 1, //!< This frame represents the end of a sequence resulting in a decodable frame.
} dwContainerFrameFlags;

/// Enum representing a supported container types.
typedef enum dwContainerType {
    DW_CONTAINER_TYPE_UNKNOWN = 0, //!< Unknow Container
    DW_CONTAINER_TYPE_MP4     = 1, //!< MP4 Container
    DW_CONTAINER_TYPE_DWPOD   = 2, //!< Product Container
} dwContainerType;

/// Holds the container metadata configuration
typedef struct dwContainerMetaConfig
{
    dwContainerType containerType;                              //!< Container type.
    uint32_t metaTypesCount;                                    //!< Number of different metadata types.
    dwContainerMetaType metaTypes[DW_CONTAINER_MAX_META_TYPES]; //!< Types of metadata.
    uint32_t capacity[DW_CONTAINER_MAX_META_TYPES];             //!< Maximum metadata item count for each metadata type specified in @c metaTypes.
} dwContainerMetaConfig;

#ifdef __cplusplus
}
#endif

#endif //DW_SENSORS_CONTAINERS_CONTAINER_H_
