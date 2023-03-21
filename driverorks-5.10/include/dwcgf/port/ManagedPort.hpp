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

#ifndef DW_FRAMEWORK_MANAGEDPORT_HPP_
#define DW_FRAMEWORK_MANAGEDPORT_HPP_

#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/MetadataHelper.hpp>
#include <dwcgf/lockstep/ILockstepSyncClient.hpp>
#include <dw/core/language/Optional.hpp>
#include <dw/core/container/RingBuffer.hpp>
#include <dw/core/language/Function.hpp>
#include <type_traits>

namespace dw
{
namespace framework
{

class ManagedPortBase : public PortBase
{
public:
    ManagedPortBase();

    virtual ~ManagedPortBase() = default;

    /**
     *  All derived Port classes cannot be copied since the channel client interface and
     *  its packets are non-copyable.
     **/
    ManagedPortBase(const ManagedPortBase& other) = delete;
    ManagedPortBase& operator=(const ManagedPortBase& other) = delete;
    ManagedPortBase(ManagedPortBase&& other)                 = delete;
    ManagedPortBase& operator=(ManagedPortBase&& other) = delete;

    /**
     *  Bind channel to the port. Indicates to channel the data type that
     *  is needed by the port and passes channel any reference information
     *  required to allocate packets.
     *  @param [in] channel the channel to bind.
     *  @note throws if channel is invalid, port is already bound, or request to channel fails.
     **/
    virtual void bindChannel(ChannelObject* channel) = 0;

    /**
     *  Bind lockstep sync client to the port.
     *  @param [in] syncClient the sync client to bind.
     *  @note throws if syncClient is invalid or port is already bound to a sync client.
     */
    void bindLockstepSyncClient(dw::framework::lockstep::ILockstepSyncClient* syncClient);

    /**
     *  @return true if Port has successfully been bound to channel already.
     **/
    virtual bool isBound() const noexcept = 0;

    /**
     *  @param [in] cycleCount the current cycle count representing the number of times the DAG has been executed.
     **/
    void setCycleCount(uint32_t cycleCount);

    /**
     *  @param [in] period the period of the node/application that created the port.
     **/
    void setPeriod(uint32_t period);

    /**
     *  Resets the port's cycle count to 0.
     *  @note does not drop the bound channel.
     **/
    virtual void reset();

    /**
     *  Getter for the channel
     *  @return the channel if bound else nullptr
     **/
    ChannelObject* getChannel();

protected:
    uint32_t m_cycleCount;
    uint32_t m_period;
    ChannelObject* m_channel;
    dw::framework::lockstep::ILockstepSyncClient* m_lockstepSyncClient;
};

/**
 *  Base class encapsulates ownership of buffers and interactions with channel in type-agnostic way.
 **/
class ManagedPortOutputBase : public ManagedPortBase
{
public:
    struct ConstructProperties
    {
    };

    struct BoundProperties
    {
        /**
        * Whether this port is a sync port and will stamp the current cycle on all
        * sync packets it produces.
        **/
        bool syncEnabled = false;
    };

    struct Properties
    {
        ConstructProperties constructProperties;
        BoundProperties boundProperties;
    };
    /**
     *  @return true if the Port is holding a buffer acquired from the channel.
     **/
    bool isBufferAvailable() const noexcept;

    /**
     *  Acquire a buffer from the channel.
     *  @note only a single buffer may be acquired from channel at any given time.
     *        if the port is already holding a buffer this call is a no-op.
     *  @note client should check after with isBufferAvailable() to know if buffer was actually acquired.
     *  @throws if channel returns unexpected error.
     **/
    void acquire();

    /**
     *  Send the buffer to downstream clients over the channel.
     *  @note if the Port has no acquired buffer this call is a no-op.
     *  @note after this call the sent buffer will no longer be available.
     *        any pointers deduced from the previously held buffer should not be used.
     *  @throws if channel returns unexpected errors.
     **/
    void send();

    const Properties& getProperties() const noexcept;

    /**
     *  Set callback function to be invoked before sending the data.
     **/
    void setCallbackBeforeSend(dw::core::Function<dwStatus()> callback);

    // Implemented inherited methods
    void bindChannel(ChannelObject* channel) override;
    bool isBound() const noexcept override;

    /**
     *  Resets the packet sequence counter
     *  @note does not drop the bound channel.
     **/
    void reset() override;

    /**
     *   Get the metadata of the acquired buffer.
     *   @return header of the acquired buffer
     **/
    ChannelMetadata& getMetadata();

    /**
     *   Send an advance packet if a lockstep sync client is bound.
     **/
    void sendAdvTimestamp();

protected:
    explicit ManagedPortOutputBase(ConstructProperties props, GenericDataReference&& ref);
    GenericData getBufferGeneric();

private:
    static BoundProperties getBoundProperties(const ChannelObject& channel);
    void populateDefaultMetadata(ChannelMetadata& header);
    GenericDataReference m_ref;
    ChannelObject::Producer* m_channelProducer;
    Properties m_props;
    dw::core::Optional<GenericData> m_buffer;
    dw::core::Function<dwStatus()> m_callbackBeforeSend;
    uint32_t m_sendSeqNum;
};

/**
 *  Base class encapsulates ownership of buffers and interactions with channel in type-agnostic way.
 **/
class ManagedPortInputBase : public ManagedPortBase
{
public:
    using RingBuffer = dw::core::RingBuffer<GenericData>;

    struct ConstructProperties
    {
        /**
         *  The maximum number of buffers this port can acquire at once.
         **/
        uint32_t maxBuffers = 1U;
        /**
         *  When recv() is called, calling thread will block at least waitTime number of us for
         *  a packet to arrive on the channel.
         *  TODO(chale): move to BoundProperties once it is supported.
         **/
        dwTime_t waitTime = 0;
    };

    struct BoundProperties
    {
        /**
         *  If all buffers have been released and there are no new buffers
         *  available on channel since recv() was last called, enableReuse will allow
         *  the last buffer acquired from the channel to be returned.
         *  Only 1 buffer will be re-usable in case multiple buffers were last acquired.
         *
         *  TODO(chale): move this to ConstructProperties once properties are passed by node instead of
         *  via channel.
         **/
        bool enableReuse = false;

        /**
        * Whether this port is a sync port and will validate a packet's cycle counter.
        **/
        bool syncEnabled = false;
        /**
        * Offset from when an incoming packet was produced to the current cycle that must
        * elapse before a packet can be releaesd to a consumer.
        **/
        uint32_t dataOffset = 0U;
    };

    struct Properties
    {
        /**
         *  Properties deduced when port is constructed.
         **/
        ConstructProperties constructProperties;
        /**
         *  Properties deduced once port is bound to channel.
         **/
        BoundProperties boundProperties;
    };

    /**
     *  @return Get the properties to initialize the port.
     **/
    const Properties& getProperties() const noexcept;

    /**
     *  Receive buffers from the channel
     *  @note client should check after with isBufferAvailable() to know if a buffer was actually received.
     *  @note when no buffers are available on the channel, this call will block waiting a buffer to
     *        arrive. (wait time given by properties)
     *  @note once a buffers has arrived on channel, port will attempt to receive buffers until
     *        there are no more available or the maximum number has been received (max number given by properties)
     *  @throws when unexpected error is received from channel or when buffers were already received (and not released).
     **/
    void recv();

    /**
     *  Release buffers to the channel
     *  @note when no buffers had previously been received this operation is a no-op.
     *  @throws when unexpected error is received from channel.
     **/
    void release();

    /**
     *  Return all buffers to the channel as release() does, but also returns buffers held for re-use.
     *  @note does not drop the bound channel.  Resets the input's port cycle count to 0.
     **/
    void reset() override;

    /**
     *  @return true if the Port is holding a buffer acquired from the channel.
     **/
    bool isBufferAvailable() const noexcept;

    // Implemented inherited methods
    void bindChannel(ChannelObject* channel) override;
    bool isBound() const noexcept override;

    /**
     *  Set callback function to be invoked after receiving the new data.
     *  Note that the callback won't be called if the data is reused.
     **/
    void setCallbackAfterRecv(dw::core::Function<dwStatus()> callback);

    /**
     *   Get the metadata of the acquired buffer.
     *   @return header of the acquired buffer
     **/
    const ChannelMetadata& getMetadata();

    /**
     *   Send an advance packet if a lockstep sync client is bound.
     **/
    void sendAdvTimestamp();

protected:
    ManagedPortInputBase(ConstructProperties props, GenericDataReference&& ref);
    GenericData getBufferGeneric() const;
    GenericData popBufferGeneric();
    void releaseToChannel(void* data);

private:
    static BoundProperties getBoundProperties(const ChannelObject& channel);
    bool recvSingle(dwTime_t waitTime);
    bool stashConsumed();
    bool packetStashed(GenericData packet);
    void handleReuseDrop();

    void releaseSingle();
    bool postProcessLockstepReplayData(GenericData packet);
    dwTime_t getWaitTime();
    bool recvData();
    bool waitForData();

    Properties m_props;
    GenericDataReference m_ref;
    ChannelObject::Consumer* m_channelConsumer;
    bool m_shouldDropFirstBuffer;
    dw::core::Function<dwStatus()> m_callbackAfterRecv;

protected:
    RingBuffer m_buffers;
    GenericData m_stashedFuturePacket;
    bool m_stashValid;
};

namespace detail
{

template <typename T>
T* getBufferTyped(GenericData buffer)
{
    auto metadataPacket = extractMetadata(buffer);
    T* ptr              = metadataPacket->data.template getData<T>();

    if (ptr == nullptr)
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "getBufferTyped: type mismatch");
    }
    return ptr;
}

} // namespace detail

/**
 *  Derived ManagedPortOutput<T> provides type-specific interfaces for accessing buffers.
 **/
template <typename T>
class ManagedPortOutput : public ManagedPortOutputBase
{
    static_assert(parameter_traits<T>::IsDeclared,
                  "Channel packet type not declared. Ensure channel packet type "
                  "handling is declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION");

    using SpecimenT = typename parameter_traits<T>::SpecimenT;

public:
    /**
     *   @param [in] props the properties of the output Port.
     *   @param [in] ref   the reference data to be used by channel to allocate
     *                     packets.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID != DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(ConstructProperties props, SpecimenT& ref)
        : ManagedPortOutputBase(std::move(props), make_specimen<T>(&ref))
    {
    }

    /**
     *   @param [in] ref   the reference data to be used by channel to allocate
     *                     packets.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID != DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(SpecimenT& ref)
        : ManagedPortOutputBase({}, make_specimen<T>(&ref))
    {
    }

    /**
     *   @param [in] props the properties of the output Port.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID == DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(ConstructProperties props)
        : ManagedPortOutputBase(std::move(props), make_specimen<T>(nullptr))
    {
    }

    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID == DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput()
        : ManagedPortOutputBase({}, make_specimen<T>(nullptr))
    {
    }

    /**
     *   Get a pointer to the acquired buffer.
     *   @note returned pointer is never nullptr
     *   @throws when the buffer cannot be cast to T.
     **/
    auto getBuffer()
    {
        return detail::getBufferTyped<T>(getBufferGeneric());
    }
};

/**
 *  Derived ManagedPortInput<T> provides type-specific interfaces for accessing buffers.
 **/
template <typename T>
class ManagedPortInput : public ManagedPortInputBase
{
    static_assert(parameter_traits<T>::IsDeclared,
                  "Channel packet type not declared. Ensure channel packet type "
                  "handling is declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION");
    using SpecimenT = typename parameter_traits<T>::SpecimenT;

    struct PacketDeleter;

public:
    /**
     *   @param [in] props the properties of the input Port.
     **/
    ManagedPortInput(ConstructProperties props)
        : ManagedPortInputBase(std::move(props), make_specimen<T>(nullptr))
    {
    }

    ManagedPortInput()
        : ManagedPortInputBase({}, make_specimen<T>(nullptr))
    {
    }

    /**
     *   Get iterator over received buffers
     *   @note returned pointers are never nullptr
     *   @note iterator access throws if buffer cannot be cast to T.
     **/
    auto getBufferIter()
    {
        return iterable(*this);
    }

    /**
     *   Get a pointer to the first received buffer.
     *   @note returned pointer is never nullptr
     *   @throws when the buffer cannot be cast to T.
     **/
    auto getBuffer()
    {
        return detail::getBufferTyped<T>(getBufferGeneric());
    }

    /**
     *   For bind optional and buffer optional case
     *   Get a pointer to the first received buffer if the buffer available.
     *   Otherwise return nullptr.
     *   @throws when the buffer cannot be cast to T.
     **/
    auto getOptionalBuffer()
    {
        return isBufferAvailable() ? getBuffer() : nullptr;
    }

    using UniquePacketPtr = std::unique_ptr<T, PacketDeleter>;

    auto takeOwnership()
    {
        GenericData packet = popBufferGeneric();
        auto* ptr          = detail::getBufferTyped<T>(packet);
        void* releasePtr   = packet.getPointer();
        return UniquePacketPtr(ptr, PacketDeleter{this, releasePtr});
    }

private:
    struct PacketDeleter
    {
        void operator()(T* p)
        {
            (void)p;
            port->releaseToChannel(releasePtr);
        }
        ManagedPortInput* port;
        void* releasePtr;
    };

    struct iterable
    {
        explicit iterable(ManagedPortInput<T>& port)
            : m_port(port)
        {
        }

        /// Iterators
        // There are no specific requirements on the template type
        // coverity[autosar_cpp14_a14_1_1_violation]
        template <class TT>
        class iterator : public ManagedPortInputBase::RingBuffer::iterator
        {
            using Base = ManagedPortInputBase::RingBuffer::iterator;

        public:
            // Same naming is used in dwshared, hence keeping the iterator name and its accessors for now
            // coverity[cert_dcl51_cpp_violation]
            iterator(Base&& base, ManagedPortInput<T>& port)
                : Base(base)
                , m_port(port)
            {
            }

            const Base& baseFromThis() const
            {
                return *this;
            }

            auto operator*() const
            {
                GenericData buffer = *baseFromThis();
                return detail::getBufferTyped<TT>(buffer);
            }

        private:
            ManagedPortInput<T>& m_port;
        };

        // coverity[cert_dcl51_cpp_violation]
        iterator<T> begin() { return iterator<T>(m_port.m_buffers.begin(), m_port); }

        // coverity[cert_dcl51_cpp_violation]
        iterator<T> end() { return iterator<T>(m_port.m_buffers.end(), m_port); }

        // coverity[cert_dcl51_cpp_violation]
        iterator<const T> begin() const { return iterator<const T>(m_port.m_buffers.begin(), m_port); }

        // coverity[cert_dcl51_cpp_violation]
        iterator<const T> end() const { return iterator<const T>(m_port.m_buffers.end(), m_port); }

    private:
        ManagedPortInput<T>& m_port;
    };
};

template <typename T>
using UniquePacketPtr = typename ManagedPortInput<T>::UniquePacketPtr;

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_MANAGEDPORT_HPP_
