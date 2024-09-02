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

#ifndef DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPESL2PLUS_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPESL2PLUS_HPP_

#include <dw/sensors/lidar/Lidar.h>

namespace dw
{
namespace framework
{

// this struct combines multiple individual lidar packets into a single object
// all packet points are aggregated into a large array 'returnPointsRTHIArray'
// here we keep all data from all individual aggregated packets
// array pointers in the 'dwLidarDecodedPacket' are pointing to the right place in the corresponding large cross-packet array
// the util class 'LidarPacketsArrayUtils' implements multiple functions for managing 'dwLidarPacketsArray'
// like estimating sizes of all arrays, adding new packets and updating packet pointers
struct dwLidarPacketsArray
{
    // array of packets
    dwLidarDecodedPacket* packets;

    dwLidarPointRTHI* returnPointsRTHIArray; // size: maxPacketsPerSpin * maxNumberOfReturns * maxPointsPerReturn
    void* auxDataArray;                      // size: maxPacketsPerSpin * maxNumberOfReturns * maxPointsPerReturn * <size of all valid aux fields>
                                             // <size of all valid aux fields> can be derived from the validAuxInfos and LidarPacketsArrayUtils::getAuxElementSize()

    uint32_t maxPacketsPerSpin;  // max number of packets in this object
    uint32_t maxPointsPerPacket; // max number of points in a single packet
    uint32_t packetSize;         // the actual number of packets

    uint32_t maxNumberOfReturns; // the maximal number of returns
    uint32_t maxPointsPerReturn; // the max number of points in the 'returnPointsRTHIArray' per return
                                 // both values above as derived from the lidar properties using
                                 // the function 'dw::sensors::lidar::Lidar::getReturnSettingsFromProperties()'

    // Bitmask of valid aux info fields based on enum dwLidarAuxDataType
    // this mask defines the max number of aux elements per return
    uint64_t validAuxInfos;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPESL2PLUS_HPP_
