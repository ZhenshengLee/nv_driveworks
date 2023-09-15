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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_LIDARPOINTCLOUD_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_LIDARPOINTCLOUD_HPP_

#include <dw/core/context/Context.h>
#include <dw/pointcloudprocessing/pointcloud/LidarPointCloud.h>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwLidarPointCloud, DW_LIDAR_POINT_CLOUD);

namespace dw
{
namespace framework
{
///////////////////////////////////////////////////////////////////////////////////////
class LidarPointCloudPacket : public IChannelPacket
{
public:
    LidarPointCloudPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    ~LidarPointCloudPacket() override;
    GenericData getGenericData() final;

protected:
    dwLidarPointCloud m_lidarPointCloud{};
    dwLidarPointCloud m_dispatchLidarPointCloud{};
};

// LidarPointCloud packet for channel socket
class ChannelPacketLidarPointCloud : public LidarPointCloudPacket, public ChannelSocketPacketBase
{
public:
    ChannelPacketLidarPointCloud(const GenericData& specimen, dwContextHandle_t);
    size_t getMaxNumBytes() const;
    // Serializes the frame before transmission
    void serializeImpl() final;
    void serializeBuffer(void* dst, void* const src, size_t count, dwMemoryType type);
    void deserialize(size_t) final;
    void deserializeBufferCuda();
    void deserializeBufferNotCuda();
    void fixLayerPoints(dwPointCloud& pointCloud);
    static size_t getFormatSize(dwPointCloudFormat format);

private:
    size_t m_formatSize{};
    size_t m_maxNumBytes{};
};

} // namespace framework
} // namespace dw

///////////////////////////////////////////////////////////////////////////////////////
#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_LIDARPOINTCLOUD_HPP_
