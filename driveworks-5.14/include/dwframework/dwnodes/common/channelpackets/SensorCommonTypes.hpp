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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_

#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dw::framework::SensorServiceNodeRawData, size_t, DW_SENSOR_SERVICE_RAW_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwFeatureNccScores, DW_FEATURE_NCC_SCORES);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLatency, DW_LATENCY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLidarPacketsArray, DW_LIDAR_PACKETS_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwSensorNodeRawData, DW_SENSOR_NODE_RAW_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwTraceNodeData, DW_TRACE_NODE_DATA);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCameraIntrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCalibratedRadarIntrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCodecMetadata);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwSensorNodeProperties);

namespace dw
{
namespace framework
{

// SensorServiceNodeRawData
///////////////////////////////////////////////////////////////////////////////////////
class SensorServiceNodeRawDataPacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of sensor service node raw data
     *  @param[in] ctx driveworks context */
    SensorServiceNodeRawDataPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of  sensor service node raw data */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_packet);
    }

protected:
    SensorServiceNodeRawData m_packet{};
    std::unique_ptr<uint8_t[]> m_dataBuffer{};
    size_t m_maxDataSize;
};

class ChannelPacketSensorServiceNodeRawData : public ChannelSocketPacketBase, public SensorServiceNodeRawDataPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of sensor service node raw data
     *  @param[in] ctx driveworks context */
    ChannelPacketSensorServiceNodeRawData(const GenericData& specimen, dwContextHandle_t);
    /*! Serializes the packet before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission
     *  @param[in] unused */
    void deserialize(size_t) final;

private:
    size_t m_headerSize;
};

class SensorServiceRawDataNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 1U;

public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of sensor service node raw data
     *  @param[in] ctx driveworks context */
    SensorServiceRawDataNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    /*! Fill out additional metadata prior to sending */
    void pack() final;
    /*! Fill out metadata upon receipt */
    void unpack() final;
    /*! Fill NvSciBuffAttrList used in constructing packet
     *  @param[in] bufferIndex unused because only one buffer exists
     *  @param[in,out] attrList list of nvsci buffer attributes */
    void fillNvSciBufAttributes(uint32_t /*bufferIndex*/, NvSciBufAttrList& attrList) const final;
    /*! Getter for number of Nvsci buffers
     *  @return number of nvsci buffers */
    uint32_t getNumBuffers() const final;
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of Image handle */
    GenericData getGenericData() final;

private:
    std::unique_ptr<Buffer> m_buffer;
    SensorServiceNodeRawData m_reference{};
    SensorServiceNodeRawData m_dispatch{};
    SensorServiceNodeRawData* m_header{};
    size_t m_headerSize;
    size_t m_maxDataSize{};
};

///////////////////////////////////////////////////////////////////////////////////////
class FeatureNccScoresPacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of feature normalized cross correlation scores
     *  @param[in] ctx driveworks context */
    FeatureNccScoresPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    /*! Destructor */
    ~FeatureNccScoresPacket() override;
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of normalized cross correlation scores */
    inline GenericData getGenericData() override
    {
        return GenericData(&m_dispatchNcc);
    }

protected:
    dwFeatureNccScores m_ncc{};
    dwFeatureNccScores m_dispatchNcc{};
};

class ChannelPacketFeatureNccScores : public FeatureNccScoresPacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of feature normalized cross correlation scores
     *  @param[in] ctx driveworks context */
    ChannelPacketFeatureNccScores(const GenericData& specimen, dwContextHandle_t);
    /*! Serializes the frame before transmission */
    void serializeImpl() override;
    /*! Deserializes the frame before transmission
     *  @param[in] unused */
    void deserialize(size_t) override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
class LatencyPacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of latency structure
     *  @param[in] ctx driveworks context */
    LatencyPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    /*! Getter for generic data in channel packet
     *  @return GenericData represention of latency structure */
    inline GenericData getGenericData() override
    {
        return GenericData(&m_data);
    }

protected:
    std::unique_ptr<uint8_t[]> m_allocation{};
    dwLatency m_data{};
    dwLatency m_dataOrig{};
};

// dwLatency packet for channel socket which is used for measurement of latency/data rate across tegras.
class ChannelPacketLatency : public LatencyPacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of latency structure
     *  @param[in] ctx driveworks context */
    ChannelPacketLatency(const GenericData& specimen, dwContextHandle_t);
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission
     *  @param[in] unused */
    void deserialize(size_t) final;

private:
    size_t m_headerSize;
    size_t m_objectSize;
    size_t m_maxCount;
};

///////////////////////////////////////////////////////////////////////////////////////
// dwLidarPacketsArray for channel socket
class ChannelPacketLidarPacketsArray : public ChannelPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of lidar packets array
     *  @param[in] ctx driveworks context */
    ChannelPacketLidarPacketsArray(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Constructor
     *  @param[in] ref pointer to lidar packets array to send with channel packet
     *  @param[in] ctx driveworks context */
    ChannelPacketLidarPacketsArray(dwLidarPacketsArray& ref, dwContextHandle_t);
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission
     *  @param[in] unused */
    void deserialize(size_t) final;
    /*! Getter for lidar packets array in channel packet
     *  @return pointer to data in channel packet */
    dwLidarPacketsArray* getFrame();

private:
    dwLidarPacketsArray m_data{};

    size_t m_headerSizeBytes{};

    dwLidarDecodedPacket* m_lidarPacketsPtr{};
    dwLidarPointRTHI* m_returnPointsRTHIPtr{};
    void* m_auxDataPtr{};
};

class LidarPacketsArrayNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 1U;

public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of lidar packets array
     *  @param[in] ctx driveworks context */
    LidarPacketsArrayNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    /*! Check that the dispatch buffer hasn't been corrupted from its original state by verifying
     *  that the pointers haven't been moved. If pointers were moved, then they likely no longer
     *  point to the shared allocation in the nvscibufobj therefore, throw an exception.
     *  @throws Exception with status DW_INVALID_ARGUMENT */
    void checkDispatchedArray() const;
    /*! Check that the dispatch packets haven't been corrupted from its original state */
    void checkDispatchedPackets() const;
    /*! Fill out additional metadata prior to sending */
    void pack() final;
    /*! Fill out metadata upon receipt */
    void unpack() final;
    /*! Fill NvSciBuffAttrList used in constructing packet
     *  @param[in] bufferIndex unused because only one buffer exists
     *  @param[in,out] attrList list of nvsci buffer attributes */
    void fillNvSciBufAttributes(uint32_t /*bufferIndex*/, NvSciBufAttrList& attrList) const final;
    /*! Getter for number of Nvsci buffers
     *  @return number of nvsci buffers */
    uint32_t getNumBuffers() const final;
    /*! Getter for generic data in channel packet
     *  @return GenericData represention of lidar packets array */
    GenericData getGenericData() final;

private:
    void setPointers();

    std::unique_ptr<Buffer> m_buffer;
    dwLidarPacketsArray* m_header = nullptr;
    dwLidarPacketsArray m_reference{};
    dwLidarPacketsArray m_dispatch{};

    size_t m_headerSizeBytes{};
    size_t m_lidarDecodedPacketsSizeBytes{};
    size_t m_returnPointsRTHISizeBytes{};
    size_t m_auxDataSizeBytes{};
};
///////////////////////////////////////////////////////////////////////////////////////
// dwSensorRawData packet for channel socket
class ChannelPacketSensorNodeRawData : public ChannelPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of sensor node raw data
     *  @param[in] ctx driveworks context */
    ChannelPacketSensorNodeRawData(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Constructor
     *  @param[in] ref pointer to sensor node raw data to send with channel packet
     *  @param[in] ctx driveworks context */
    ChannelPacketSensorNodeRawData(dwSensorNodeRawData& ref, dwContextHandle_t);
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame after transmission */
    void deserialize(size_t) final;
    /*! Setter for buffer size
     *  @param[in] size of buffer in bytes */
    inline void setBufferSize(size_t bufferSize)
    {
        m_bufferSize = bufferSize;
    }
    /*! Getter for sensor node raw data in channel packet
     *  @return pointer to data in channel packet */
    inline dwSensorNodeRawData* getFrame()
    {
        return &m_data;
    }

private:
    dwSensorNodeRawData m_data{};
};

///////////////////////////////////////////////////////////////////////////////////////
class ChannelPacketTraceNodeData : public ChannelPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of trace node data
     *  @param[in] ctx driveworks context */
    ChannelPacketTraceNodeData(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Constructor
     *  @param[in] ref pointer to trace node data to send with channel packet
     *  @param[in] ctx driveworks context */
    ChannelPacketTraceNodeData(dwTraceNodeData& ref, dwContextHandle_t);
    /*! Destructor */
    ~ChannelPacketTraceNodeData() override = default;
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission */
    void deserialize(size_t) final;
    /*! Getter for trace node data in channel packet
     *  @return pointer to data in channel packet */
    inline dwTraceNodeData* getFrame()
    {
        return &m_data;
    }

private:
    dwTraceNodeData m_data{};
    size_t m_headerSize{};
    size_t m_maxDataSize{};
    uint8_t* m_TraceData;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_
