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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "SyncPortHelper.hpp"

#include <stdexcept>
#include <string>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
enum class PortDirection : uint8_t
{
    INPUT = 0,
    OUTPUT,
    COUNT,
};

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
    virtual dwStatus initialize() { return DW_SUCCESS; }
    virtual bool isBound() = 0;
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
class PortOutput : public SyncPortHelperOutput<T>, public Port
{
public:
    static constexpr PortDirection DIRECTION = PortDirection::OUTPUT;
    using ApiDataTypeT                       = T;
    using SpecimenT                          = typename parameter_traits<T>::SpecimenT;
    using BaseSyncHelper                     = SyncPortHelperOutput<T>;

    static_assert(std::is_copy_constructible<SpecimenT>::value, "SpecimenT is not copy constructible");

    static constexpr char LOG_TAG[] = "PortOutput";

private:
    ChannelObject::Producer* m_channelProducer{};
    SpecimenT m_reference{};
    OnSetSyncAttrs m_waiterAttrs{};
    OnSetSyncAttrs m_signalerAttrs{};

public:
    explicit PortOutput(SpecimenT const& ref)
        : m_reference(ref)
    {
    }
    explicit PortOutput(SpecimenT&& ref)
        : m_reference(std::move(ref))
    {
    }

    explicit PortOutput(SpecimenT const& ref,
                        OnSetSyncAttrs signalerAttrs,
                        OnSetSyncAttrs waiterAttrs = {})
        : m_reference(ref)
        , m_waiterAttrs(waiterAttrs)
        , m_signalerAttrs(signalerAttrs)
    {
    }

    // Channel Bind
    dwStatus bindChannel(ChannelObject* channel) override
    {
        auto ref = make_specimen<T>(&m_reference);
        return bindChannelWithReference(channel, ref);
    }

    dwStatus bindChannelWithReference(ChannelObject* channel, GenericDataReference& ref)
    {
        return Exception::guard([&] {
            if (isBound())
            {
                // TODO(chale): this should be an Exception but applications are currently
                // doing this. Those applications should be fixed.
                FRWK_LOGE << "PortOutput: bindChannel: attempted to bind the same port twice, ignoring this bind!" << Logger::State::endl;
                return;
            }
            if (channel == nullptr)
            {
                throw Exception(DW_INVALID_ARGUMENT, "PortOutput: bindChannel: expected channel != nullptr");
            }

            BaseSyncHelper::parseDataSynced(channel->getParams());

            if (BaseSyncHelper::isDataSynced())
            {
                ref.packetTypeID = BaseSyncHelper::getNewPacketID(ref.packetTypeID);
            }

            ref.setWaiterAttributes   = m_waiterAttrs;
            ref.setSignalerAttributes = m_signalerAttrs;

            m_channelProducer = channel->getProducer(ref);
            if (m_channelProducer == nullptr)
            {
                throw Exception(DW_INTERNAL_ERROR, "PortOutput bindChannel: wrong channel implementations returned.");
            }
        });
    }

    void setOnDataReady(void* opaque, ChannelObject::PacketPool::OnDataReady onDataReady)
    {
        if (!isBound())
        {
            throw Exception(DW_NOT_AVAILABLE, "PortOutput: setOnDataReady: no bound channel");
        }
        m_channelProducer->setOnDataReady(opaque, onDataReady);
    }

    bool isBound() final
    {
        return (m_channelProducer != nullptr);
    }

    dwStatus wait(dwTime_t timeout)
    {
        if (!isBound())
        {
            throw Exception(DW_NOT_AVAILABLE, "PortInput: wait: no bound channel");
        }

        return m_channelProducer->wait(timeout);
    }

    // Node accessors
    // TODO(unknown): This function's prototype needs to change to properly propagate errors
    T* getFreeElement()
    {
        dwStatus status = DW_FAILURE;
        GenericData genericData{};
        if (m_channelProducer)
        {
            status = m_channelProducer->get(&genericData);
        }

        if (status != DW_SUCCESS)
        {
            return nullptr;
        }

        if (BaseSyncHelper::isDataSynced())
        {
            return BaseSyncHelper::extractInternalPacket(genericData);
        }

        return genericData.template getData<T>();
    }

    // Tx Operations
    virtual dwStatus send(T* frame)
    {
        if (!m_channelProducer)
        {
            throw Exception(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }

        if (BaseSyncHelper::isDataSynced())
        {
            return m_channelProducer->send(BaseSyncHelper::getSyncPacket(frame));
        }

        return m_channelProducer->send(frame);
    }

    ChannelObject::SyncSignaler& getSyncSignaler()
    {
        if (!m_channelProducer)
        {
            throw Exception(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }
        return m_channelProducer->getSyncSignaler();
    }

    void setSignalFences(T* frame, dw::core::span<NvSciSyncFence> fences)
    {
        if (BaseSyncHelper::isDataSynced())
        {
            m_channelProducer->getSyncSignaler().setSignalFences(BaseSyncHelper::getSyncPacket(frame), fences);
        }
        else
        {
            m_channelProducer->getSyncSignaler().setSignalFences(frame, fences);
        }
    }

    ChannelObject::SyncWaiter& getSyncWaiter()
    {
        if (!m_channelProducer)
        {
            throw Exception(DW_NOT_AVAILABLE, "PortOutput: channel not bound");
        }
        return m_channelProducer->getSyncWaiter();
    }

    void getWaitFences(T* frame, dw::core::span<NvSciSyncFence> fences)
    {
        if (BaseSyncHelper::isDataSynced())
        {
            m_channelProducer->getSyncWaiter().getWaitFences(BaseSyncHelper::getSyncPacket(frame), fences);
        }
        else
        {
            m_channelProducer->getSyncWaiter().getWaitFences(frame, fences);
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
class PortInput : public SyncPortHelperInput<T>, public Port
{
    static constexpr char LOG_TAG[] = "PortInput";

public:
    static constexpr PortDirection DIRECTION = PortDirection::INPUT;
    using ApiDataTypeT                       = T;
    using SpecimenT                          = typename parameter_traits<T>::SpecimenT;
    using BaseSyncHelper                     = SyncPortHelperInput<T>;

    static_assert(std::is_copy_constructible<SpecimenT>::value, "SpecimenT is not copy constructible");

    explicit PortInput(SpecimenT const& ref)
        : m_reference(ref)
    {
    }
    explicit PortInput(SpecimenT&& ref)
        : m_reference(std::move(ref))
    {
    }

    PortInput()
    {
    }

    explicit PortInput(OnSetSyncAttrs waiterAttrs,
                       OnSetSyncAttrs signalerAttrs = {})
        : m_waiterAttrs(waiterAttrs)
        , m_signalerAttrs(signalerAttrs)
    {
    }

    explicit PortInput(SpecimenT const& ref,
                       OnSetSyncAttrs waiterAttrs,
                       OnSetSyncAttrs signalerAttrs = {})
        : m_reference(ref)
        , m_waiterAttrs(waiterAttrs)
        , m_signalerAttrs(signalerAttrs)
    {
    }

    ~PortInput() override = default;

    // Channel Bind
    dwStatus bindChannel(ChannelObject* channel) override
    {
        return Exception::guard([&] {
            if (isBound())
            {
                // TODO(chale): this should be an Exception but applications are currently
                // doing this. Those applications should be fixed.
                FRWK_LOGE << "PortInput: bindChannel: attempted to bind the same port twice, ignoring this bind!" << Logger::State::endl;
                return;
            }
            if (channel == nullptr)
            {
                throw Exception(DW_INVALID_ARGUMENT, "PortInput: bindChannel: expected channel != nullptr");
            }

            BaseSyncHelper::parseDataSynced(channel->getParams());
            auto ref = make_specimen<T>(nullptr);

            if (m_reference)
            {
                ref = make_specimen<T>(&m_reference.value());
            }

            if (BaseSyncHelper::isDataSynced())
            {
                ref.packetTypeID = BaseSyncHelper::getNewPacketID(ref.packetTypeID);
            }

            ref.setWaiterAttributes   = m_waiterAttrs;
            ref.setSignalerAttributes = m_signalerAttrs;

            m_channelConsumer = channel->getConsumer(ref);
            if (m_channelConsumer == nullptr)
            {
                throw Exception(DW_INTERNAL_ERROR, "PortInput bindChannel: wrong channel implementations returned.");
            }
            m_reuse = channel->getParams().getReuseEnabled();
        });
    }

    bool isBound() override
    {
        return !(m_channelConsumer == nullptr);
    }

    void setOnDataReady(void* opaque, ChannelObject::PacketPool::OnDataReady onDataReady)
    {
        if (!isBound())
        {
            throw Exception(DW_NOT_AVAILABLE, "PortInput: setOnDataReady: no bound channel");
        }
        m_channelConsumer->setOnDataReady(opaque, onDataReady);
    }

    // Rx Operations
    dwStatus wait(dwTime_t timeout)
    {
        if (!isBound())
        {
            throw Exception(DW_NOT_AVAILABLE, "PortInput: wait: no bound channel");
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
            timeout = 0;
        }

        dwTime_t waitTime = m_last ? 0 : timeout;
        dwStatus status   = m_channelConsumer->wait(waitTime);
        if (m_last && (status == DW_TIME_OUT || status == DW_NOT_AVAILABLE))
        {
            return DW_SUCCESS;
        }

        return status;
    }

    // TODO(unknown): This function's prototype needs to change to properly propagate errors
    virtual std::shared_ptr<T> recv()
    {
        GenericData data{};
        std::shared_ptr<T> result;
        if (!isBound())
        {
            return nullptr;
        }

        T* typedData     = nullptr;
        void* releasePtr = nullptr;

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
            dwStatus status = m_channelConsumer->recv(&data);
            if (status != DW_SUCCESS)
            {
                if (m_last != nullptr)
                {
                    return m_last;
                }
                else
                {
                    return nullptr;
                }
            }
            releasePtr = data.getPointer();

            if (BaseSyncHelper::isDataSynced())
            {
                typedData = BaseSyncHelper::extractInternalPacket(data);
                if (!typedData)
                {
                    return nullptr;
                }
            }
            else
            {
                typedData = data.template getData<T>();
            }
        }

        // don't rely on this class's member when releasing packet
        auto* channelConsumer = m_channelConsumer;
        result                = std::shared_ptr<T>(typedData, [channelConsumer, releasePtr](T*) {
            channelConsumer->release(releasePtr);
        });
        if (m_reuse)
        {
            m_last = result;
        }

        return result;
    }

    ChannelObject::SyncSignaler& getSyncSignaler()
    {
        if (!m_channelConsumer)
        {
            throw Exception(DW_NOT_AVAILABLE, "PortInput: channel not bound");
        }
        return m_channelConsumer->getSyncSignaler();
    }

    void setSignalFences(T* frame, dw::core::span<NvSciSyncFence> fences)
    {
        if (BaseSyncHelper::isDataSynced())
        {
            throw Exception(DW_NOT_SUPPORTED, "PortInput: not supported");
        }
        else
        {
            m_channelConsumer->getSyncSignaler().setSignalFences(frame, fences);
        }
    }

    ChannelObject::SyncWaiter& getSyncWaiter()
    {
        if (!m_channelConsumer)
        {
            throw Exception(DW_NOT_AVAILABLE, "PortInput: channel not bound");
        }
        return m_channelConsumer->getSyncWaiter();
    }

    void getWaitFences(T* frame, dw::core::span<NvSciSyncFence> fences)
    {
        if (BaseSyncHelper::isDataSynced())
        {
            throw Exception(DW_NOT_SUPPORTED, "PortInput: not supported");
        }
        else
        {
            m_channelConsumer->getSyncWaiter().getWaitFences(frame, fences);
        }
    }

private:
    ChannelObject::Consumer* m_channelConsumer{};
    bool m_reuse{};
    std::shared_ptr<T> m_last{};
    dw::core::Optional<SpecimenT> m_reference{};
    OnSetSyncAttrs m_waiterAttrs{};
    OnSetSyncAttrs m_signalerAttrs{};
};

template <typename T>
constexpr char PortInput<T>::LOG_TAG[];

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PORT_H_
