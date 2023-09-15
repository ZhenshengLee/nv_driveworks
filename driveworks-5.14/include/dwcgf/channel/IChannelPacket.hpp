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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/ChannelParameters.hpp>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>

namespace dw
{
namespace framework
{

/**
 *  Abstract interface encapsulating a single channel packet.
 *  Applications must extend this interface to describe how the channel
 *  should allocate and transport its data types. Implementations of this
 *  class shall be allocated by factory functions which are registered to
 *  the ChannelFactory under a specific ChannelPacketTypeID and for a specific
 *  ChannelType.
 */
class IChannelPacket
{
public:
    // Enable ownership via this interface
    virtual ~IChannelPacket() = default;

    /**
     *  Get the data owned by the packet.
     */
    virtual GenericData getGenericData() = 0;

    /**
     * Additional interfaces needed by socket channels
     * to allocate and transport the data type.
     * A packet transported by socket must have the ability to
     * be serialized to and deserialized from a private buffer.
     */
    class SocketCallbacks
    {
    public:
        /**
         * Get a pointer to the serialized payload.
         */
        virtual uint8_t* getBuffer() = 0;
        /**
         * Get the maximum size of the serialized payload.
         * @return size_t
         */
        virtual size_t getBufferSize() = 0;
        /**
         * Serialize the packet contents to the payload.
         * @return size_t the size of the serialized payload
         */
        // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
        virtual size_t serialize() = 0;
        /**
         * Deserialize the packet contents from the payload.
         */
        virtual void deserialize(size_t) = 0;
    };

    /**
     * Additional interfaces needed by nvsci channels
     * to allocate and transport the data type.
     * A packet transported by NvSci channels must have the ability
     * to back its allocation with NvSciBufs and pack/unpack its
     * representation from the NvSciBufs.
     */
    class NvSciCallbacks
    {
    public:
        /**
         * Get the number of NvSciBufObjs needed.
         * @return uint32_t
         *
         * @note init and runtime
         */
        virtual uint32_t getNumBuffers() const = 0;
        /**
         * Fill the NvSciBufAttributes needed for the given NvSciBufObj.
         *
         * @param [in] bufferIndex the index of the NvSciBufObj
         * @param [out] attrList reference to the output attr list. The attr list is valid when passed by Channel
         *                       and can be filled in place or reassigned.
         *
         * @note init-time only
         */
        virtual void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const = 0;
        /**
         * Initialize the packet from the given NvSciBufObjs
         * @param bufs the span of NvSvciBufObjs to back the packet allocations.
         *
         * @note init-time only, will be called before pack()/unpack() is ever called.
         */
        virtual void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) = 0;

        /**
         * Pack the packet into the NvSciBufObjs so that the NvSciBufObjs contain the latest information from a producer.
         */
        virtual void pack() = 0;
        /**
         * Unpack the packet from the NvSciBufObjs so that the latest information is available to a consumer.
         */
        virtual void unpack() = 0;
    };

    /**
     * Get the socket-specific interfaces for the packet.
     *
     * @return SocketCallbacks&
     */
    SocketCallbacks& getSocketCallbacks();
    /**
     * Get the nvsci-specific interfaces for the packet.
     *
     * @return NvSciCallbacks&
     */
    NvSciCallbacks& getNvSciCallbacks();
};

// TODO(chale): this should be made private to the framework once external entities no longer create
// Channel packets
class ChannelPacketDefaultBase : public IChannelPacket
{
public:
    ChannelPacketDefaultBase(size_t typeSize);

    GenericData getGenericData() final;

protected:
    size_t m_typeSize;
    std::unique_ptr<uint8_t[]> m_buffer;
    GenericData m_data;
};

// coverity[autosar_cpp14_a0_1_6_violation]
class ChannelPacketDefault : public ChannelPacketDefaultBase, public IChannelPacket::SocketCallbacks
{
public:
    ChannelPacketDefault(size_t typeSize);

    uint8_t* getBuffer() final;

    size_t getBufferSize() final;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    size_t serialize() final;

    void deserialize(size_t) final;
};

// coverity[autosar_cpp14_a0_1_6_violation]
class ChannelNvSciPacketDefault : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    // coverity[autosar_cpp14_a0_1_1_violation]
    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    static constexpr char LOG_TAG[]{"ChannelNvSciPacketDefault"};

public:
    ChannelNvSciPacketDefault(size_t typeSize);

    uint32_t getNumBuffers() const final;
    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final;

    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs);

    void pack() final;

    void unpack() final;

    GenericData getGenericData() final;

private:
    void fillCpuPacketDataAttributes(NvSciBufAttrList& output) const;

    size_t m_typeSize;
    void* m_data;
};

/**
 * Interface to allocate packets
 */
class IChannelPacketFactory
{
public:
    /**
     * @brief Make a packet
     *
     * @param ref the parameters of the packet type.
     * @param channelType the channel type for which to make the packet.
     * @return std::unique_ptr<IChannelPacket>
     */
    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual std::unique_ptr<IChannelPacket> makePacket(const GenericDataReference& ref, ChannelType channelType) = 0;
};
// coverity[autosar_cpp14_a0_1_6_violation]
using ChannelPacketFactoryPtr = std::shared_ptr<IChannelPacketFactory>;

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ICHANNEL_PACKET_HPP_
