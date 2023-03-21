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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_SYNCPORTHELPER_HPP_
#define DW_FRAMEWORK_SYNCPORTHELPER_HPP_

#include <dw/core/container/HashContainer.hpp>
#include <dwcgf/channel/Channel.hpp>

namespace dw
{
namespace framework
{

using CycleCountFetcher = dw::core::Function<uint32_t(void)>;

// These classes are used to parse and handle indexed packets.
// TODO (ajayawardane) Move this logic into a separate port and change
// the port type for each pipelined node.
struct SyncPortHelper
{
public:
    SyncPortHelper()
        : m_dataSynced(false)
        , m_syncCount(0U)
        , m_dataOffset(0U)
        , m_syncCountRetriever(nullptr)
    {
    }
    void setSyncCount(uint32_t index);
    virtual void parseDataSynced(const ChannelParams& params);
    bool isDataSynced();
    dwStatus setSyncRetriever(const CycleCountFetcher& func);

protected:
    ChannelPacketTypeID getNewPacketID(ChannelPacketTypeID packetTypeID);
    void stampSyncCount(uint32_t& syncCountOut) const;

    bool m_dataSynced;
    uint32_t m_syncCount;
    uint32_t m_dataOffset;
    CycleCountFetcher m_syncCountRetriever;
};

// There are no specific requirements on the template type
// coverity[autosar_cpp14_a14_1_1_violation]
template <typename T>
class SyncPortHelperOutput : public SyncPortHelper
{
public:
    SyncPortHelperOutput()
        : SyncPortHelper()
    {
    }

protected:
    T* extractInternalPacket(GenericData genericData)
    {
        auto metadataPacket = genericData.template getData<MetadataPayload>();

        if (!metadataPacket)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperOutput extractInternalPacket: packet type mismatch");
        }

        auto packet = metadataPacket->data.template getData<T>();
        if (!packet)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperOutput extractInternalPacket: failed to extract underlying data");
        }

        m_metadataPayloadBuf[packet] = metadataPacket;
        return packet;
    }

    MetadataPayload* getMetadataPacket(T* frame)
    {
        MetadataPayload* metadataPacket = m_metadataPayloadBuf[frame];
        if (!metadataPacket)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperOutput getmetadataPacket: sync packet not found in packet buffer");
        }

        return metadataPacket;
    }

    void parseDataSynced(const ChannelParams& params) override
    {
        SyncPortHelper::parseDataSynced(params);
        m_metadataPayloadBuf = HeapHashMap<T*, MetadataPayload*>(params.getPoolCapacity());
    }

private:
    HeapHashMap<T*, MetadataPayload*> m_metadataPayloadBuf;
};

// There are no specific requirements on the template type
// coverity[autosar_cpp14_a14_1_1_violation]
template <typename T>
class SyncPortHelperInput : public SyncPortHelper
{
public:
    SyncPortHelperInput()
        : SyncPortHelper()
        , m_bufferedPacket()
        , m_dataBuffered(false)
    {
    }

protected:
    bool isPacketBuffered()
    {
        if (m_dataBuffered)
        {
            return true;
        }
        return false;
    }

    bool isValidPacketBuffered()
    {
        if (!isPacketBuffered())
        {
            return false;
        }

        auto packet = m_bufferedPacket.template getData<MetadataPayload>();

        if (!packet)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperInput isValidPacketBuffered: packet type mistmatch");
        }
        return validatePacket(*packet);
    }

    GenericData getBufferedPacket()
    {
        m_dataBuffered = false;
        return m_bufferedPacket;
    }

    T* extractSyncPacket(GenericData genericData)
    {
        auto metadataPacket = genericData.template getData<MetadataPayload>();

        if (!metadataPacket)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperInput extractSyncPacket: packet type mistmatch");
        }

        if (validatePacket(*metadataPacket))
        {
            auto packet                  = metadataPacket->data.template getData<T>();
            m_metadataPayloadBuf[packet] = metadataPacket;
            return packet;
        }
        else
        {
            m_bufferedPacket = genericData;
            m_dataBuffered   = true;
            return nullptr;
        }
    }

    T* extractInternalPacket(GenericData genericData)
    {
        auto metadataPacket = genericData.template getData<MetadataPayload>();

        if (!metadataPacket)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperInput extractInternalPacket: packet type mistmatch");
        }

        auto packet                  = metadataPacket->data.template getData<T>();
        m_metadataPayloadBuf[packet] = metadataPacket;
        return packet;
    }

    MetadataPayload* getMetadataPacket(T* frame)
    {
        MetadataPayload* metadataPacket = m_metadataPayloadBuf[frame];
        if (!metadataPacket)
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "SyncPortHelperInput getmetadataPacket: sync packet not found in packet buffer");
        }

        return metadataPacket;
    }

    void parseDataSynced(const ChannelParams& params) override
    {
        SyncPortHelper::parseDataSynced(params);
        m_metadataPayloadBuf = HeapHashMap<T*, MetadataPayload*>(params.getPoolCapacity());
    }

private:
    bool validatePacket(MetadataPayload& pkt)
    {
        // If a producer - consumer pair are across pipeline boundaries, they will
        // have non-zero data offsets; however, connections from that producer to
        // consumers not across the pipeline boundary must also have sync ports
        // (since if a producer is sync, all consumers must be sync). This check
        // is in place for cases where producer -> consumer pairs are in the same
        // pipeline boundary, and is basically a no-op for synchronization.
        if (m_dataOffset == 0)
        {
            return true;
        }

        uint32_t syncCount = (m_syncCountRetriever) ? m_syncCountRetriever() : m_syncCount;

        // Check if the packet is valid for consumption. The packet sync count represents
        // when the packet was produced and the m_syncCount is the current sync count.
        // The data offset is the offset between when it was produced and when it is
        // available for consumption.
        int validOffset = static_cast<int>(syncCount - pkt.header.iterationCount - m_dataOffset);

        if (validOffset >= 0)
        {
            return true;
        }

        return false;
    }

    HeapHashMap<T*, MetadataPayload*> m_metadataPayloadBuf;
    GenericData m_bufferedPacket;
    bool m_dataBuffered;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SYNCPORTHELPER_HPP_
