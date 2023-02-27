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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_ICHANNEL_PACKET_HPP_
#define DW_FRAMEWORK_ICHANNEL_PACKET_HPP_

#include <typeinfo>
#include <cstddef>
#include <dwcgf/Types.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/ChannelParameters.hpp>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <dw/core/language/Function.hpp>

namespace dw
{
namespace framework
{

// Virtual functions for the channel to interact with packets opaquely
class IChannelPacket
{
public:
    // Enable ownership via this interface
    virtual ~IChannelPacket() = default;
    // Deserialize metadata at provided pointer
    virtual GenericData getGenericData() = 0;

    // Callbacks needed for socket channel
    class SocketCallbacks
    {
    public:
        // Get a pointer to the payload
        virtual uint8_t* getBuffer() = 0;
        // Get size of serialized data buffer
        virtual size_t getBufferSize() = 0;
        // Serialize the packet to internal buffer
        virtual size_t serialize() = 0;
        // Deserialize the packet from internal buffer
        virtual void deserialize(size_t) = 0;
    };

    // Callbacks needed for nvsci channel
    class NvSciCallbacks
    {
    public:
        // Init time callbacks

        // Get size of the metadata
        virtual uint32_t getNumBuffers() const = 0;
        // Get size of serialized data buffer
        virtual void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const = 0;
        // Initialize the packet from the given NvSciBufObjs
        virtual void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) = 0;

        // Runtime callbacks

        // Pack the API data type into the NvSciBufObjs
        virtual void pack() = 0;
        // Unpack the API data type from the NvSciBufObjs
        virtual void unpack() = 0;
    };

    // Get interface for channel socket, throws if not enabled or supported
    SocketCallbacks& getSocketCallbacks()
    {
        if (auto* ptr = dynamic_cast<SocketCallbacks*>(this))
        {
            return *ptr;
        }
        else
        {
            throw Exception(DW_NOT_SUPPORTED, "This packet interface does not implement socket callbacks");
        }
    }

    NvSciCallbacks& getNvSciCallbacks()
    {
        if (auto* ptr = dynamic_cast<NvSciCallbacks*>(this))
        {
            return *ptr;
        }
        else
        {
            throw Exception(DW_NOT_SUPPORTED, "This packet interface does not implement nvsci callbacks");
        }
    }
};

// TODO(chale): this should be made private to the framework once external entities no longer create
// Channel packets
class ChannelPacketDefaultBase : public IChannelPacket
{
public:
    ChannelPacketDefaultBase(size_t typeSize)
        : m_typeSize(typeSize)
        , m_buffer(new uint8_t[m_typeSize]())
        , m_data{m_buffer.get(), m_typeSize}
    {
    }

    GenericData getGenericData() final
    {
        return m_data;
    }

protected:
    size_t m_typeSize;
    std::unique_ptr<uint8_t[]> m_buffer;
    GenericData m_data;
};

class ChannelPacketDefault : public ChannelPacketDefaultBase, public IChannelPacket::SocketCallbacks
{
public:
    ChannelPacketDefault(size_t typeSize)
        : ChannelPacketDefaultBase(typeSize)
    {
    }

    uint8_t* getBuffer() final
    {
        return m_buffer.get();
    }

    size_t getBufferSize() final
    {
        return m_typeSize;
    }

    size_t serialize() final
    {
        return m_typeSize;
    }

    void deserialize(size_t) final
    {
    }
};

class ChannelNvSciPacketDefault : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr char LOG_TAG[] = "ChannelNvSciPacketDefault";

public:
    ChannelNvSciPacketDefault(size_t typeSize)
        : m_typeSize{typeSize}
    {
    }

    uint32_t getNumBuffers() const final
    {
        return 1U;
    }

    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final
    {
        if (bufferIndex != 0U)
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciPacketDefault: invalid buffer index");
        }

        fillCpuPacketDataAttributes(attrList);
    }

    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs)
    {
        if (bufs.size() != getNumBuffers())
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciPacketDefault: wrong number of buffers passed");
        }

        FRWK_CHECK_NVSCI_ERROR(NvSciBufObjGetCpuPtr(bufs[0], &m_data));
    }

    void pack() final
    {
        // noop
    }

    void unpack() final
    {
        // noop
    }

    GenericData getGenericData() final
    {
        return GenericData(m_data, m_typeSize);
    }

private:
    void fillCpuPacketDataAttributes(NvSciBufAttrList& output) const
    {
        const NvSciBufType bufferType               = NvSciBufType_RawBuffer;
        const bool cpuAccessFlag                    = true;
        const uint64_t rawAlign                     = 4;
        const NvSciBufAttrValAccessPerm permissions = NvSciBufAccessPerm_ReadWrite;

        dw::core::Array<NvSciBufAttrKeyValuePair, 5> rawBufferAttributes =
            {{{NvSciBufGeneralAttrKey_Types, &bufferType, sizeof(bufferType)},
              {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag, sizeof(cpuAccessFlag)},
              {NvSciBufGeneralAttrKey_RequiredPerm, &permissions, sizeof(permissions)},
              {NvSciBufRawBufferAttrKey_Align, &rawAlign, sizeof(rawAlign)},
              {NvSciBufRawBufferAttrKey_Size, &m_typeSize, sizeof(m_typeSize)}}};

        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(output,
                                                        rawBufferAttributes.data(),
                                                        rawBufferAttributes.size()));
    }

    size_t m_typeSize{};
    void* m_data{};
};

class IChannelPacketFactory
{
public:
    virtual std::unique_ptr<IChannelPacket> makePacket(ChannelPacketTypeID id, ChannelType channelType, GenericData ref) = 0;
};
using ChannelPacketFactoryPtr = std::shared_ptr<IChannelPacketFactory>;

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ICHANNEL_PACKET_HPP_
