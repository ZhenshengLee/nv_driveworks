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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_RADAR_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_RADAR_HPP_

#include <dw/sensors/radar/Radar.h>
#include <dwcgf/Exception.hpp>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwRadarScan, DW_RADAR_SCAN);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwRadarProperties);

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
// dwRadarScan packet for channel socket
class ChannelPacketRadarScan : public ChannelPacketBase
{
public:
    ChannelPacketRadarScan(const GenericData& specimen, dwContextHandle_t ctx);
    ChannelPacketRadarScan(dwRadarScan& ref, dwContextHandle_t);
    dwRadarScan* getFrame();
    void setBufferSize(size_t bufferSize);
    // Serializes the frame before transmission
    size_t serialize() final;
    // Deserializes the frame after transmission
    void deserialize(size_t) final;

private:
    size_t getRadarDataSize(const dwRadarScan& radarScan);
    dwRadarScan m_data{};
    std::unique_ptr<uint8_t[]> m_scanData;
    std::unique_ptr<uint8_t[]> m_scanDetectionMisc;
    std::unique_ptr<uint8_t[]> m_scanDetectionStdDev;
    std::unique_ptr<uint8_t[]> m_scanDetectionQuality;
    std::unique_ptr<uint8_t[]> m_scanDetectionProbability;
    std::unique_ptr<uint8_t[]> m_scanDetectionFFTPatch;
    std::unique_ptr<uint8_t[]> m_scanRadarSSI;
};

class RadarScanNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
private:
    static constexpr uint32_t NUM_BUFFERS{1U};

public:
    RadarScanNvSciPacket(const GenericData& specimen);
    uint32_t getNumBuffers() const final;
    void fillNvSciBufAttributes(uint32_t /*bufferIndex*/, NvSciBufAttrList& attrList) const final;
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    void pack() final;
    void unpack() final;
    GenericData getGenericData() final;

private:
    void setPointers();

    std::unique_ptr<Buffer> m_buffer;
    dwRadarScan m_reference{};
    dwRadarScan m_dispatch{};
    dwRadarScan* m_header{};
    size_t m_headerSize{};
    size_t m_numReturns{};
    size_t m_dataSize{};
    size_t m_detectionMiscSize{};
    size_t m_detectionStdDevSize{};
    size_t m_detectionQualitySize{};
    size_t m_detectionProbabilitySize{};
    size_t m_detectionFFTPatchSize{};
    size_t m_radarSSISize{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_RADAR_HPP_
