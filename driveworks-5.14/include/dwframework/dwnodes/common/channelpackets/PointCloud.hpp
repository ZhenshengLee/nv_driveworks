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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_POINTCLOUD_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_POINTCLOUD_HPP_

#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwcgf/channel/Buffer.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwPointCloud, DW_POINT_CLOUD);

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
class PointCloudChannelPacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of point cloud
     *  @param[in] ctx driveworks context */
    PointCloudChannelPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    /*! Destructor */
    ~PointCloudChannelPacket() override;
    /*! Getter for generic data in channel packet
     *  @return  GenericData represention of point cloud */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_data);
    }

protected:
    dwPointCloud m_data{};
    dwPointCloud m_dataOri{};
};

// PointCloud packet for channel socket
class ChannelPacketPointCloud : public PointCloudChannelPacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of point cloud
     *  @param[in] ctx driveworks context */
    ChannelPacketPointCloud(const GenericData& specimen, dwContextHandle_t);
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame after transmission
     *  @param[in] unused */
    void deserialize(size_t) final;
    /*! Getter for size of point cloud format size
     *  @return size in bytes of point cloud format
     *  @throws std::runtime_error if unhandled type */
    static size_t getFormatSize(dwPointCloudFormat format);

private:
    size_t m_headerSize{};
    size_t m_objectSize{};
    size_t m_maxCount{};
};

class PointCloudNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 2U;

public:
    PointCloudNvSciPacket(const GenericData& specimen, dwContextHandle_t context = DW_NULL_HANDLE);
    uint32_t getNumBuffers() const final;
    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final;
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    void pack() final;
    void unpack() final;
    GenericData getGenericData() final;

private:
    void setPointers();
    void initReference();

private:
    size_t m_headerSize{};
    dw::core::VectorFixed<std::unique_ptr<Buffer>, NUM_BUFFERS> m_buffers{};
    dwPointCloud* m_header{};
    dwPointCloud m_reference{};
    dwPointCloud m_dispatch{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_POINTCLOUD_HPP_
