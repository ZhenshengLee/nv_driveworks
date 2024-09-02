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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dwshared/dwfoundation/dw/core/safety/Safety.hpp>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
// coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
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

    // coverity[autosar_cpp14_a2_10_5_violation]
    size_t serialize() override
    {
        serializeImpl();
        return m_bufferSize;
    }

    // Serializes the frame before transmission
    virtual void serializeImpl()
    {
        throw ExceptionWithStatus(DW_NOT_SUPPORTED, "ChannelPacketBase: serialize: not implemented");
    }

    // Deserializes the frame before transmission
    void deserialize(size_t) override = 0;

protected:
    ChannelSocketPacketBase() = default;

    explicit ChannelSocketPacketBase(size_t bufferSize)
        : IChannelPacket::SocketCallbacks()
    {
        initBuffer(bufferSize);
    }

    void initBuffer(size_t bufferSize)
    {
        m_bufferSize = bufferSize;

        // TODO (ajayawardane): the extra uint32_t is to ensure there is enough room
        // to insert sync count after buffer. Figure out a better way to handle this
        // the buffer size isn't close to the maximum value of size_t, hence no risk of overflow
        m_buffer = std::make_unique<uint8_t[]>(dw::core::safeAdd(m_bufferSize, sizeof(ChannelMetadata)).value());

        if (nullptr == m_buffer)
        {
            throw ExceptionWithStatus(DW_BAD_ALLOC, "ChannelPacketBase: initBuffer: cannot allocate memory");
        }
    }

    // packet should have a buffer of type uint8_t
    // for serialization and de-serialization
    size_t m_bufferSize{0U};
    std::unique_ptr<uint8_t[]> m_buffer;
};

// coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
// coverity[autosar_cpp14_a10_1_1_violation]
class ChannelPacketBase : public IChannelPacket, public ChannelSocketPacketBase
{
public:
    GenericData getGenericData() override
    {
        if (!m_frame.has_value())
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "ChannelPacketBase: getGenericData: not set by implementation.");
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
    static_assert(std::is_constructible<T>::value, "T must be constructible");
    static_assert(std::is_trivially_copyable<T>::value, "Type argument T is not trivially copyable");

    virtual void notImplemented() = 0;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_PACKET_HPP_
