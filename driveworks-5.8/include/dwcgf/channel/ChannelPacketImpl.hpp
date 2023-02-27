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
// SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNEL_PACKET_HPP_
#define DW_FRAMEWORK_CHANNEL_PACKET_HPP_

#include <dwcgf/channel/IChannelPacket.hpp>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
class ChannelSocketPacketBase : public IChannelPacket::SocketCallbacks
{
public:
    uint8_t* getBuffer() override
    {
        return m_buffer.get();
    }

    size_t getBufferSize() override
    {
        return m_bufferSize;
    }

    size_t serialize() override
    {
        serializeImpl();
        return m_bufferSize;
    }

    // Serializes the frame before transmission
    virtual void serializeImpl()
    {
        throw Exception(DW_NOT_SUPPORTED, "ChannelPacketBase: serialize: not implemented");
    }

    // Deserializes the frame before transmission
    void deserialize(size_t) override
    {
        throw Exception(DW_NOT_SUPPORTED, "ChannelPacketBase: deserialize: not implemented");
    }

protected:
    ChannelSocketPacketBase() = default;

    ChannelSocketPacketBase(size_t bufferSize)
    {
        initBuffer(bufferSize);
    }

    void initBuffer(size_t bufferSize)
    {
        m_bufferSize = bufferSize;

        // TODO (ajayawardane): the extra uint32_t is to ensure there is enough room
        // to insert sync count after buffer. Figure out a better way to handle this
        m_buffer = std::make_unique<uint8_t[]>(m_bufferSize + sizeof(uint32_t));

        if (m_buffer == nullptr)
        {
            throw Exception(DW_BAD_ALLOC, "ChannelPacketBase: initBuffer: cannot allocate memory");
        }
    }

    // packet should have a buffer of type uint8_t
    // for serialization and de-serialization
    size_t m_bufferSize = 0;
    std::unique_ptr<uint8_t[]> m_buffer;
};

class ChannelPacketBase : public IChannelPacket, public ChannelSocketPacketBase
{
public:
    GenericData getGenericData() override
    {
        if (!m_frame)
        {
            throw Exception(DW_INTERNAL_ERROR, "ChannelPacketBase: getGenericData: not set by implementation.");
        }
        return m_frame.value();
    }

    using ChannelSocketPacketBase::ChannelSocketPacketBase;

protected:
    dw::core::Optional<GenericData> m_frame;
};

template <typename T>
class ChannelPacket
{
    virtual void notImplemented() = 0;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_PACKET_HPP_
