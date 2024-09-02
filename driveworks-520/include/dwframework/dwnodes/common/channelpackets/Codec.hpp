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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_CODEC_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_CODEC_HPP_

#include <dw/sensors/codecs/Codec.h>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwCodecPacket, size_t, DW_CODEC_PACKET);

namespace dw
{
namespace framework
{

// dwCodecPacket
///////////////////////////////////////////////////////////////////////////////////////
class CodecPacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of codec packet
     *  @param[in] ctx driveworks context */
    CodecPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of codec packet */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_packet);
    }

protected:
    dwCodecPacket m_packet{};
    std::unique_ptr<uint8_t[]> m_dataBuffer{};
    size_t m_maxDataSize;
};

// coverity[autosar_cpp14_a10_1_1_violation]
class ChannelPacketCodecPacket : public ChannelSocketPacketBase, public CodecPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of codec packet */
    ChannelPacketCodecPacket(const GenericData& specimen, dwContextHandle_t);
    /*! Serializes the packet before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission */
    void deserialize(size_t) final;

private:
    size_t m_headerSize;
};

class CodecNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 1U;

public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of codec packet */
    CodecNvSciPacket(const GenericData& specimen, dwContextHandle_t = DW_NULL_HANDLE);
    /*! Fill NvSciBuffAttrList used in constructing packet
     *  @param[in,out] attrList list of nvsci buffer attributes */
    void fillNvSciBufAttributes(uint32_t /*bufferIndex*/, NvSciBufAttrList& attrList) const final;
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    /*! Fill out additional metadata prior to sending */
    void pack() final;
    /*! Fill out metadata upon receipt */
    void unpack() final;
    /*! Getter for number of Nvsci buffers
     *  @return number of nvsci buffers */
    inline uint32_t getNumBuffers() const final
    {
        return NUM_BUFFERS;
    }
    inline GenericData getGenericData() final
    {
        return GenericData(&m_dispatch);
    }

private:
    std::unique_ptr<Buffer> m_buffer;
    dwCodecPacket m_reference{};
    dwCodecPacket m_dispatch{};
    dwCodecPacket* m_header{};
    size_t m_headerSize;
    size_t m_maxDataSize{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_CODEC_HPP_
