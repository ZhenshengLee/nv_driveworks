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

#ifndef DW_SENSORS_CONTAINERS_METADATA_H_
#define DW_SENSORS_CONTAINERS_METADATA_H_

#ifdef __cplusplus
extern "C" {
#endif

/// Enum representing supported metadata types
typedef enum {
    DW_CONTAINER_META_TYPE_UINT64, //!< NVP with a uint64_t as payload.
    DW_CONTAINER_META_TYPE_BOOL,   //!< NVP with a bool as payload.
    DW_CONTAINER_META_TYPE_STRING, //!< NVP with a string as payload.
} dwContainerMetaType;

/// Metadata type holding a string-uint64 name-value pair.
typedef struct dwContainerMetaUint64
{
    char const* name; //!< Field name.
    uint64_t value;   //!< Field value.
} dwContainerMetaUint64;

/// Metadata type holding a string-bool name-value pair.
typedef struct dwContainerMetaBool
{
    char const* name; //!< Field name.
    bool value;       //!< Field value.
} dwContainerMetaBool;

/// Metadata type holding a string-string name-value pair.
typedef struct dwContainerMetaString
{
    char const* name;  //!< Field name.
    char const* value; //!< Field value.
} dwContainerMetaString;

#ifdef __cplusplus
}
#endif

#endif //DW_SENSORS_CONTAINERS_METADATA_H_
