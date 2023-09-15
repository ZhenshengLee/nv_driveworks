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

#ifndef SENSOR_COMMON_IMPL_HPP_
#define SENSOR_COMMON_IMPL_HPP_

#include <dw/core/base/Types.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

// forward declare
struct dwRigObject;
typedef struct dwRigObject const* dwConstRigHandle_t;
struct dwSALObject;
typedef struct dwSALObject* dwSALHandle_t;
struct dwSensorObject;
typedef struct dwSensorObject* dwSensorHandle_t;

namespace dw
{
namespace framework
{
// forward declare
class dwLidarPacketsArray;

class SensorCommonImpl
{
    using FixedString1024 = dw::core::FixedString<1024>;

public:
    static FixedString1024 adjustPath(const char* paramString, uint64_t sensorId, const char* toFind, dwConstRigHandle_t& rig);
    // TODO(csketch): should not use string
    static dwSensorHandle_t createSensor(std::string sensorName, dwConstRigHandle_t rig, dwSALHandle_t sal);
};

/// define some useful functions to operate with raw C-structs for LiDAR data
class LidarPacketsArrayUtils
{
public:
    /*! there is a duplication of pointers inside 'dwLidarPacketsArray':
     *  - there is a cross-packet data array
     *  - there are individual data pointers inside every packet
     * the method updates data pointers inside single packets
     * and links them to the corresponding parts of the cross-packet data array */
    static void updateDataPointersForAllPackets(dwLidarPacketsArray& packetArray);
    /// update links for a single packets (for details - see 'updateDataPointersForAllPackets')
    static void updateDataPointersForPacket(dwLidarPacketsArray& packetArray, size_t const packetIndex);
    /// return the size of decoded packages without data (only C struct header), bytes
    static size_t getLidarDecodedPacketsSizeBytes(size_t const numberOfPackets);
    /// return the number of RTHI-points across all returns in the dwLidarPacketsArray
    static size_t getNumberOfReturnPointsRTHI(size_t const numberOfPackets, dwLidarPacketsArray const& packetArray);
    /// return the size of all RTHI-points across all returns in the dwLidarPacketsArray, bytes
    static size_t getReturnPointsRTHISizeBytes(size_t const numberOfPackets, dwLidarPacketsArray const& packetArray);
    /// return the size of a single aux-data items, bytes
    static size_t getSingleAuxDataItemSizeBytes(dwLidarPacketsArray const& packetArray, size_t iAuxType);
    /// return the full size of a aux-data items per return, bytes
    static size_t getSingleReturnAuxDataSizeBytes(dwLidarPacketsArray const& packetArray);
    /// return the full size of a aux-data items inside 'dwLidarPacketsArray', bytes
    static size_t getPacketReturnAuxDataOffset(size_t const numberOfPackets, dwLidarPacketsArray const& lidarPacketsArray);
    /// add a new packet to the 'dwLidarPacketsArray' with copying all underlying data
    static void addNewPacketAsDeepCopy(dwLidarPacketsArray& packetArray, dwLidarDecodedPacket const& newPacket);
    /// get size (in bytes) for each aux element
    static size_t getAuxElementSize(dwLidarDecodedReturn const& returnPacket, dwLidarAuxDataType auxType);
    static void validatePacket(dwLidarPacketsArray const& packetArray, dwLidarDecodedPacket const& packet);
    static void initPacketSettings(dwLidarPacketsArray& packetArray);
};

} // namespace framework
} // namespace dw

#endif // SENSOR_COMMON_IMPL_HPP_
