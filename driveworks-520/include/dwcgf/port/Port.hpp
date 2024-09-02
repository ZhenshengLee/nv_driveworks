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

#ifndef DW_FRAMEWORK_PORT_H_
#define DW_FRAMEWORK_PORT_H_

#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/Exception.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <dwcgf/port/MetadataHelper.hpp>

#include "SyncPortHelper.hpp"

#include <nvscisync.h>
#include <stdexcept>
#include <string>

namespace dw
{
namespace framework
{

namespace detail
{

template <typename T>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
T* getBufferTyped(GenericData buffer)
{
    MetadataPayload* metadataPacket{extractMetadata(buffer)};
    T* ptr{metadataPacket->data.template getData<T>()};

    if (nullptr == ptr)
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "getBufferTyped: type mismatch");
    }
    return ptr;
}

template <typename T>
struct vectorIterable
{
    static_assert(std::is_constructible<T>::value, "T must be constructible");

    explicit vectorIterable(dw::core::VectorFixed<GenericData> allBuffers)
        : m_allBuffers(std::move(allBuffers))
    {
    }

    /// Iterators
    template <class TT>
    class iterator : public dw::core::VectorFixed<GenericData>::iterator
    {
        static_assert(std::is_constructible<TT>::value, "TT must be constructible");

    public:
        using Base = dw::core::VectorFixed<GenericData>::iterator;
        // Same naming is used in dwshared, hence keeping the iterator name and its accessors for now
        iterator(Base&& base)
            : Base(std::move(base))
        {
        }

        const Base& baseFromThis() const
        {
            return *this;
        }

        TT* operator*() const
        {
            GenericData buffer{*baseFromThis()};
            return getBufferTyped<TT>(buffer);
        }
    };

    iterator<T> begin() { return iterator<T>(m_allBuffers.begin()); }

    iterator<T> end() { return iterator<T>(m_allBuffers.end()); }

private:
    dw::core::VectorFixed<GenericData> m_allBuffers;
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////
enum class PortDirection : uint8_t
{
    INPUT = 0,
    OUTPUT,
};

// coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
class PortBase
{
public:
    virtual ~PortBase() = default;
};

///////////////////////////////////////////////////////////////////////////////////////
class Port : public PortBase
{
public:
    virtual dwStatus bindChannel(ChannelObject* channel) = 0;
    virtual bool isBound()                               = 0;
    virtual ChannelObject* getChannel()
    {
        return m_channel;
    };

protected:
    ChannelObject* m_channel{nullptr};
};

///////////////////////////////////////////////////////////////////////////////////////
/**
 * PortOutput mimics an Output Block. It wrapps over a Producer Channel.
 * Provides services like
 * - binding(allocating channel resources, out of current scope),
 * - getting next free element
 * - sending data
 */
template <typename T>
// coverity[autosar_cpp14_a10_1_1_violation]
class PortOutput : public SyncPortHelperOutput<T>, public Port
{
    static_assert(std::is_constructible<T>::value, "T must be constructible");

public:
    static constexpr PortDirection DIRECTION = PortDirection::OUTPUT;
    // coverity[autosar_cpp14_a0_1_6_violation]
    using ApiDataTypeT   = T;
    using SpecimenT      = typename parameter_traits<T>::SpecimenT;
    using BaseSyncHelper = SyncPortHelperOutput<T>;

    static_assert(std::is_copy_constructible<SpecimenT>::value, "SpecimenT is not copy constructible");

    // coverity[autosar_cpp14_a2_10_5_violation]
    static constexpr char LOG_TAG[]{"PortOutput"};

private:
    ChannelObject::Producer* m_channelProducer;
    SpecimenT m_reference;
    OnSetSyncAttrs m_waiterAttrs;
    OnSetSyncAttrs m_signalerAttrs;
    void* m_onDataReadyOpaque;
    OnDataReady m_onDataReady;
    uint32_t m_sendSeqNum;

public:
    explicit PortOutput(SpecimenT const& ref)
        : PortOutput(ref, {})
    {
    }
    explicit PortOutput(SpecimenT&& ref)
        : SyncPortHelperOutput<T>()
        , Port()
        , m_channelProducer(nullptr)
        , m_reference(std::move(ref))
        , m_onDataReadyOpaque()
        , m_onDataReady()
        , m_sendSeqNum(0U)
    {
    }

    explicit PortOutput(SpecimenT const& ref,
                        OnSetSyncAttrs signalerAttrs,
                        OnSetSyncAttrs waiterAttrs = {})
        : SyncPortHelperOutput<T>()
        , Port()
        , m_channelProducer(nullptr)
        , m_reference(ref)
        , m_waiterAttrs(std::move(waiterAttrs))
        , m_signalerAttrs(std::move(signalerAttrs))
        , m_onDataReadyOpaque()
        , m_onDataReady()
        , m_sendSeqNum(0U)
    {
    }

    // Channel Bind
    dwStatus bindChannel(ChannelObject* channel) override
    {
        GenericDataReference ref{make_specimen<T>(&m_reference)};
        return bindChannelWithReference(channel, ref);
    }

    dwStatus bindChannelWithReference(ChannelObject* channel, GenericDataReference& ref)
    {
        return ExceptionGuard::guard(
            [&] {
                if (isBound())
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortOutput: bindChannel: port already bound");
                }
                if (nullptr == channel)
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortOutput: bindChannel: expected channel != nullptr");
                }
                m_channel = channel;
                BaseSyncHelper::parseDataSynced(channel->getParams());
                ref.packetTypeID          = BaseSyncHelper::getNewPacketID(ref.packetTypeID);
                ref.setWaiterAttributes   = m_waiterAttrs;
                ref.setSignalerAttributes = m_signalerAttrs;
                ref.onDataReadyOpaque     = m_onDataReadyOpaque;
                ref.onDataReady           = m_onDataReady;

                m_channelProducer = channel->getProducer(ref);
                if (nullptr == m_channelProducer)
                {
                    throw ExceptionWithStatus(DW_INTERNAL_ERROR, "PortOutput bindChannel: wrong channel implementations returned.");
                }
            },
            dw::core::Logger::Verbosity::DEBUG);
    }

    /**
     *  Bind channel to transfer packets of POD type, Pointers are considered
     *  POD so only shallow copy happens. It is expected to work for local
     *  share memory channel only.
     **/
    dwStatus bindChannelForPODTypePacket(ChannelObject* channel)
    {
        return ExceptionGuard::guard(
            [&] {
                if (isBound())
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortOutput: bindCbindChannelForPODTypePackethannel: port already bound");
                }
                if (nullptr == channel)
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortOutput: bindChannelForPODTypePacket: expected channel != nullptr");
                }
                if (ChannelType::SHMEM_LOCAL != channel->getParams().getType())
                {
                    throw dw::core::ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "PortOutput: bindChannelForPODTypePacket: setting channel to use POD type only allowed for local channels.");
                }

                BaseSyncHelper::parseDataSynced(channel->getParams());

                // coverity[autosar_cpp14_a20_8_4_violation] FP: nvbugs/4552679
                GenericDataReference ref{};
                ref.packetTypeID          = static_cast<dw::framework::ChannelPacketTypeID>(DWFRAMEWORK_METADATA_PACKET_TYPE_ID_OFFSET + static_cast<uint32_t>(DWFRAMEWORK_PACKET_ID_DEFAULT));
                ref.typeSize              = sizeof(T);
                ref.data                  = GenericData(static_cast<T*>(nullptr));
                ref.setWaiterAttributes   = m_waiterAttrs;
                ref.setSignalerAttributes = m_signalerAttrs;
                ref.onDataReadyOpaque     = m_onDataReadyOpaque;
                ref.onDataReady           = m_onDataReady;

                m_channelProducer = channel->getProducer(ref);
                if (nullptr == m_channelProducer)
                {
                    throw ExceptionWithStatus(DW_INTERNAL_ERROR, "PortOutput bindChannelForPODTypePacket: wrong channel implementations returned.");
                }
            },
            dw::core::Logger::Verbosity::DEBUG);
    }

    void setOnDataReady(void* opaque, OnDataReady onDataReady)
    {
        if (isBound())
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: setOnDataReady: channel already bound");
        }
        m_onDataReadyOpaque = opaque;
        m_onDataReady       = std::move(onDataReady);
    }

    bool isBound() final
    {
        return (nullptr != m_channelProducer);
    }

    dwStatus wait(dwTime_t timeout)
    {
        if (!isBound())
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: wait: no bound channel");
        }

        return m_channelProducer->wait(timeout);
    }

    // Node accessors
    // TODO(unknown): This function's prototype needs to change to properly propagate errors
    T* getFreeElement()
    {
        dwStatus status{DW_FAILURE};
        GenericData genericData{};
        if (m_channelProducer)
        {
            status = m_channelProducer->get(&genericData);
        }

        if (DW_SUCCESS != status)
        {
            return nullptr;
        }

        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        extractMetadata(genericData)->header.validFields = 0U;
        return BaseSyncHelper::extractInternalPacket(genericData);
    }

    // Tx Operations
    virtual dwStatus send(T* frame, const dwTime_t* publishTimestamp = nullptr)
    {
        if (!m_channelProducer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }

        MetadataPayload* payload{BaseSyncHelper::getMetadataPacket(frame)};
        populateDefaultMetadata(payload->header, publishTimestamp);
        return m_channelProducer->send(payload);
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    ChannelMetadata& getMetadata(T* frame)
    {
        if (!m_channelProducer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }

        MetadataPayload* payload{BaseSyncHelper::getMetadataPacket(frame)};
        return payload->header;
    }

    ChannelObject::SyncSignaler& getSyncSignaler()
    {
        if (!m_channelProducer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }
        return m_channelProducer->getSyncSignaler();
    }

    void setSignalFences(T* frame, dw::core::span<const NvSciSyncFence> fences)
    {
        getSyncSignaler().setSignalFences(BaseSyncHelper::getMetadataPacket(frame), fences);
    }

    ChannelObject::SyncWaiter& getSyncWaiter()
    {
        if (!m_channelProducer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }
        return m_channelProducer->getSyncWaiter();
    }

    void getWaitFences(T* frame, dw::core::span<NvSciSyncFence>& fences)
    {
        getSyncWaiter().getWaitFences(BaseSyncHelper::getMetadataPacket(frame), fences);
    }

    /**
     *  iterable for all the buffers in the output channel pool.
     **/
    detail::vectorIterable<T> getAllBufferIter()
    {
        return detail::vectorIterable<T>(m_channelProducer->getAllBuffers());
    }

protected:
    void populateDefaultMetadata(ChannelMetadata& header, const dwTime_t* publishTimestamp)
    {
        setSequenceNumber(header, m_sendSeqNum);
        if (m_sendSeqNum < std::numeric_limits<decltype(m_sendSeqNum)>::max())
        {
            m_sendSeqNum++;
        }
        else
        {
            m_sendSeqNum = std::numeric_limits<decltype(m_sendSeqNum)>::min();
        }
        setTimestamp(header, nullptr != publishTimestamp ? *publishTimestamp : m_channelProducer->getCurrentTime());
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        header.producerId = 0U;

        if (BaseSyncHelper::isDataSynced())
        {
            BaseSyncHelper::stampSyncCount(header.iterationCount);
            header.validFields |= static_cast<uint16_t>(MetadataFlags::METADATA_ITERATION_COUNT);
        }
    }
};

template <typename T>
constexpr char PortOutput<T>::LOG_TAG[];

///////////////////////////////////////////////////////////////////////////////////////
/**
 * PortInput mimics an Output Block. It wrapps over a Consumer Channel.
 * Provides services like
 * - binding(allocating channel resources, out of current scope),
 * - waiting for next event
 * - receiving packet
 */
template <typename T>
// coverity[autosar_cpp14_a10_1_1_violation]
class PortInput : public SyncPortHelperInput<T>, public Port
{
    static_assert(std::is_constructible<T>::value, "T must be constructible");
    static_assert(parameter_traits<T>::IsDeclared,
                  "Channel packet type not declared. Ensure channel packet type "
                  "handling is declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_POD "
                  "or DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION");
    // coverity[autosar_cpp14_a2_10_5_violation]
    static constexpr char LOG_TAG[]{"PortInput"};

public:
    static constexpr PortDirection DIRECTION = PortDirection::INPUT;
    // coverity[autosar_cpp14_a0_1_6_violation]
    using ApiDataTypeT   = T;
    using SpecimenT      = typename parameter_traits<T>::SpecimenT;
    using BaseSyncHelper = SyncPortHelperInput<T>;

    static_assert(std::is_copy_constructible<SpecimenT>::value, "SpecimenT is not copy constructible");

    explicit PortInput(SpecimenT const& ref)
        : PortInput(ref, OnSetSyncAttrs())
    {
    }
    explicit PortInput(SpecimenT&& ref)
        : SyncPortHelperInput<T>()
        , Port()
        , m_channelConsumer(nullptr)
        , m_reuse(false)
        , m_calledRecvImpl(RECV_API_CALLED_NONE)
        , m_lastTypedData(nullptr)
        , m_lastReleasePtr(nullptr)
        , m_existingUniquePtr(false)
        , m_reference(std::move(ref))
        , m_waiterAttrs()
        , m_signalerAttrs()
        , m_onDataReadyOpaque()
        , m_onDataReady()
    {
    }

    PortInput()
        : PortInput(OnSetSyncAttrs())
    {
    }

    explicit PortInput(OnSetSyncAttrs waiterAttrs,
                       OnSetSyncAttrs signalerAttrs = {})
        : SyncPortHelperInput<T>()
        , Port()
        , m_channelConsumer(nullptr)
        , m_reuse(false)
        , m_calledRecvImpl(RECV_API_CALLED_NONE)
        , m_lastTypedData(nullptr)
        , m_lastReleasePtr(nullptr)
        , m_existingUniquePtr(false)
        , m_waiterAttrs(std::move(waiterAttrs))
        , m_signalerAttrs(std::move(signalerAttrs))
        , m_onDataReadyOpaque()
        , m_onDataReady()
    {
    }

    explicit PortInput(SpecimenT const& ref,
                       OnSetSyncAttrs waiterAttrs,
                       OnSetSyncAttrs signalerAttrs = {})
        : SyncPortHelperInput<T>()
        , Port()
        , m_channelConsumer(nullptr)
        , m_reuse(false)
        , m_calledRecvImpl(RECV_API_CALLED_NONE)
        , m_lastTypedData(nullptr)
        , m_lastReleasePtr(nullptr)
        , m_existingUniquePtr(false)
        , m_reference(ref)
        , m_waiterAttrs(std::move(waiterAttrs))
        , m_signalerAttrs(std::move(signalerAttrs))
        , m_onDataReadyOpaque()
        , m_onDataReady()
    {
    }

    ~PortInput() override
    {
        // release data cached for reuse
        if (nullptr != m_channelConsumer && nullptr != m_lastReleasePtr)
        {
            if (m_existingUniquePtr.load())
            {
                DW_LOGE << dw::core::StringView{"~PortInput: Cannot release reused packet since the unique_ptr has not been returned by caller yet"} << Logger::State::endl;
            }
            else
            {
                static_cast<void>(m_channelConsumer->release(m_lastReleasePtr));
            }
        }
    }

    // Channel Bind
    dwStatus bindChannel(ChannelObject* channel) override
    {
        return ExceptionGuard::guard(
            [&] {
                if (isBound())
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortInput: bindChannel: port already bound");
                }
                if (nullptr == channel)
                {
                    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "PortInput: bindChannel: expected channel != nullptr");
                }
                m_channel = channel;

                BaseSyncHelper::parseDataSynced(channel->getParams());
                GenericDataReference ref{make_specimen<T>(nullptr)};

                if (m_reference.has_value())
                {
                    ref = make_specimen<T>(&m_reference.value());
                }

                ref.packetTypeID          = BaseSyncHelper::getNewPacketID(ref.packetTypeID);
                ref.setWaiterAttributes   = m_waiterAttrs;
                ref.setSignalerAttributes = m_signalerAttrs;
                ref.onDataReadyOpaque     = m_onDataReadyOpaque;
                ref.onDataReady           = m_onDataReady;

                m_channelConsumer = channel->getConsumer(ref);
                if (nullptr == m_channelConsumer)
                {
                    throw ExceptionWithStatus(DW_INTERNAL_ERROR, "PortInput bindChannel: wrong channel implementations returned.");
                }
                m_reuse = channel->getParams().getReuseEnabled();
            },
            dw::core::Logger::Verbosity::DEBUG);
    }

    bool isBound() override
    {
        return !(nullptr == m_channelConsumer);
    }

    void setOnDataReady(void* opaque, OnDataReady onDataReady)
    {
        if (isBound())
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortInput: setOnDataReady: channel already bound");
        }
        m_onDataReadyOpaque = opaque;
        m_onDataReady       = std::move(onDataReady);
    }

    // Rx Operations
    dwStatus wait(dwTime_t timeout)
    {
        if (!isBound())
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortInput: wait: no bound channel");
        }

        // For synced packets, the wait can return DW_NOT_AVAILABLE or DW_SUCCESS
        // if there are no packets to consume. This is because you need to consume
        // a packet to make sure it's valid or not.
        if (BaseSyncHelper::isValidPacketBuffered())
        {
            return DW_SUCCESS;
        }
        else if (BaseSyncHelper::isPacketBuffered())
        {
            return DW_NOT_AVAILABLE;
        }
        else if (BaseSyncHelper::isDataSynced())
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            timeout = 0;
        }

        bool hasLast{(nullptr != m_last.get()) || (nullptr != m_lastTypedData)};
        dwTime_t waitTime{hasLast ? 0 : timeout};
        dwStatus status{m_channelConsumer->wait(waitTime)};
        if (hasLast && (DW_TIME_OUT == status || DW_NOT_AVAILABLE == status))
        {
            return DW_SUCCESS;
        }

        return status;
    }

    /// @deprecated Use recvUnique() instead.
    virtual std::shared_ptr<T> recv()
    {
        if (RECV_API_CALLED_RECV_UNIQUE == m_calledRecvImpl)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "PortInput: recv() can't be called after calling recvUnique() before");
        }
        m_calledRecvImpl = RECV_API_CALLED_RECV;

        GenericData data{};
        std::shared_ptr<T> result{};
        if (!isBound())
        {
            return nullptr;
        }

        // coverity[autosar_cpp14_a0_1_1_violation]
        T* typedData{nullptr};
        // coverity[autosar_cpp14_a0_1_1_violation]
        void* releasePtr{nullptr};

        if (BaseSyncHelper::isValidPacketBuffered())
        {
            // There is a valid packet to consume
            data       = BaseSyncHelper::getBufferedPacket();
            releasePtr = data.getPointer();
            typedData  = BaseSyncHelper::extractInternalPacket(data);
        }
        else if (BaseSyncHelper::isPacketBuffered())
        {
            // There is a buffered packet, but it's not ready to be consumed.
            return nullptr;
        }
        else
        {
            dwStatus status{m_channelConsumer->recv(&data)};
            if (DW_SUCCESS != status)
            {
                if (nullptr != m_last)
                {
                    return m_last;
                }
                else
                {
                    return nullptr;
                }
            }
            if (BaseSyncHelper::isDataSynced())
            {
                typedData = BaseSyncHelper::extractSyncPacket(data);
                if (!typedData)
                {
                    return nullptr;
                }
            }
            else
            {
                typedData = BaseSyncHelper::extractInternalPacket(data);
            }
            releasePtr = data.getPointer();
        }

        // don't rely on this class's member when releasing packet
        ChannelObject::Consumer* channelConsumer{m_channelConsumer};
        // coverity[autosar_cpp14_a5_1_9_violation] FP: nvbugs/4347682
        result = std::shared_ptr<T>(typedData, [channelConsumer, releasePtr](T*) {
            static_cast<void>(channelConsumer->release(releasePtr));
        });
        if (m_reuse)
        {
            m_last = result;
        }

        return result;
    }

    struct PacketDeleter
    {
        void operator()(T* p) const
        {
            if (nullptr == port)
            {
                // coverity[autosar_cpp14_a18_5_2_violation] RFD Accepted: TID-2417
                delete p;
            }
            else
            {
                if (!port->m_reuse)
                {
                    static_cast<void>(port->m_channelConsumer->release(releasePtr));
                }
                else
                {
                    port->m_existingUniquePtr = false;
                }
            }
        }
        PortInput* port;
        void* releasePtr;
    };

    using UniquePacketPtr = std::unique_ptr<T, PacketDeleter>;

    virtual UniquePacketPtr recvUnique()
    {
        if (RECV_API_CALLED_RECV == m_calledRecvImpl)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "PortInput: recvUnique() can't be called after calling recv() before");
        }
        m_calledRecvImpl = RECV_API_CALLED_RECV_UNIQUE;

        if (!isBound())
        {
            return nullptr;
        }
        GenericData data{};

        // coverity[autosar_cpp14_a0_1_1_violation]
        T* typedData{nullptr};
        // coverity[autosar_cpp14_a0_1_1_violation]
        void* releasePtr{nullptr};
        if (BaseSyncHelper::isValidPacketBuffered())
        {
            // There is a valid packet to consume
            data       = BaseSyncHelper::getBufferedPacket();
            releasePtr = data.getPointer();
            typedData  = BaseSyncHelper::extractInternalPacket(data);
        }
        else if (BaseSyncHelper::isPacketBuffered())
        {
            // There is a buffered packet, but it's not ready to be consumed.
            return nullptr;
        }
        else
        {
            dwStatus status{m_channelConsumer->recv(&data)};
            if (DW_SUCCESS != status)
            {
                return makeUniquePtr();
            }
            if (BaseSyncHelper::isDataSynced())
            {
                typedData = BaseSyncHelper::extractSyncPacket(data);
                if (!typedData)
                {
                    return nullptr;
                }
            }
            else
            {
                typedData = BaseSyncHelper::extractInternalPacket(data);
            }
            releasePtr = data.getPointer();
        }

        return makeUniquePtr(typedData, releasePtr);
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    ChannelMetadata& getMetadata(T* frame)
    {
        if (!m_channelConsumer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortInput: channel not bound");
        }

        MetadataPayload* payload{BaseSyncHelper::getMetadataPacket(frame)};
        return payload->header;
    }

    ChannelObject::SyncSignaler& getSyncSignaler()
    {
        if (!m_channelConsumer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortInput: channel not bound");
        }
        return m_channelConsumer->getSyncSignaler();
    }

    void setSignalFences(T* frame, dw::core::span<const NvSciSyncFence> fences)
    {
        getSyncSignaler().setSignalFences(BaseSyncHelper::getMetadataPacket(frame), fences);
    }

    ChannelObject::SyncWaiter& getSyncWaiter()
    {
        if (!m_channelConsumer)
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "PortInput: channel not bound");
        }
        return m_channelConsumer->getSyncWaiter();
    }

    void getWaitFences(T* frame, dw::core::span<NvSciSyncFence>& fences)
    {
        getSyncWaiter().getWaitFences(BaseSyncHelper::getMetadataPacket(frame), fences);
    }

    /**
     *  iterable for all the buffers in the output channel pool.
     **/
    detail::vectorIterable<T> getAllBufferIter()
    {
        return detail::vectorIterable<T>(m_channelConsumer->getAllBuffers());
    }

private:
    UniquePacketPtr makeUniquePtr(T* typedData = nullptr, void* releasePtr = nullptr)
    {
        if (!m_reuse)
        {
            return UniquePacketPtr(typedData, PacketDeleter{this, releasePtr});
        }

        if (m_existingUniquePtr.load())
        {
            // never hand out more than one shared_ptr when reuse is enabled
            // the caller must release the previous shared_ptr before requesting a new one with recv()
            // indenpendent if the new unique_ptr refers to the same reused packet or a new one
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "Cannot return unique_ptr of reused packet since previous unique_ptr has not been returned");
        }

        if (nullptr == typedData && nullptr == m_lastTypedData)
        {
            return nullptr;
        }

        if (nullptr != typedData)
        {
            if (nullptr != m_lastTypedData)
            {
                // release previous data when new data has been received
                static_cast<void>(m_channelConsumer->release(m_lastReleasePtr));
            }
            m_lastTypedData  = typedData;
            m_lastReleasePtr = releasePtr;
        }

        m_existingUniquePtr = true;
        return std::move(UniquePacketPtr(m_lastTypedData, PacketDeleter{this, releasePtr}));
    }

    /// @deprecated See recv()
    static constexpr uint8_t RECV_API_CALLED_NONE{0U};
    /// @deprecated See recv()
    static constexpr uint8_t RECV_API_CALLED_RECV{1U};
    /// @deprecated See recv()
    static constexpr uint8_t RECV_API_CALLED_RECV_UNIQUE{2U};

    ChannelObject::Consumer* m_channelConsumer;
    bool m_reuse;
    /// Track the first called recv function to prevent calling the other recv function afterwards.
    uint8_t m_calledRecvImpl;
    /// @deprecated See recv()
    std::shared_ptr<T> m_last;
    /// The typed data pointer of the last packet being available for reuse.
    T* m_lastTypedData;
    /// The pointer of the last packet available for reuse to be returned to the channel.
    void* m_lastReleasePtr;
    /// Track lifetime of unique_ptr returned by recvUnique() since only one can exist at a time for reuse.
    std::atomic<bool> m_existingUniquePtr;
    dw::core::Optional<SpecimenT> m_reference;
    OnSetSyncAttrs m_waiterAttrs;
    OnSetSyncAttrs m_signalerAttrs;
    void* m_onDataReadyOpaque;
    OnDataReady m_onDataReady;
};

template <typename T>
constexpr char PortInput<T>::LOG_TAG[];

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PORT_H_
