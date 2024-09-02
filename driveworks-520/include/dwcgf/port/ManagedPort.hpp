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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/port/MetadataHelper.hpp>
#include <dwcgf/lockstep/ILockstepSyncClient.hpp>
#include <dwshared/dwfoundation/dw/core/language/Optional.hpp>
#include <dwshared/dwfoundation/dw/core/container/RingBuffer.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>
#include <type_traits>
#include <typeindex>
#include <utility>

#include <fiu/fiu.hpp>
#include <fiu/FaultRegistry.hpp>

namespace dw
{
namespace framework
{

class ManagedPortBase : public PortBase
{
public:
    ManagedPortBase(const dw::core::StringView& name);

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

    /// Get the name of the port.
    const dw::core::StringView& getName() const;

    /// Get the name of the node this port belongs to.
    const dw::core::StringView& getNodeName() const;

    /// Set the name of the node this port belongs to.
    virtual void setNodeName(const dw::core::StringView& nodeName) noexcept;

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
    /// The unique name within set of ports with the same direction.
    dw::core::StringView m_name;
    /// The name of the node this port belongs to.
    dw::core::StringView m_nodeName;
    uint32_t m_cycleCount;
    uint32_t m_period;
    ChannelObject* m_channel;
    dw::framework::lockstep::ILockstepSyncClient* m_lockstepSyncClient;
    FI_DECLARE_INSTANCE_SET_HANDLE(m_fiHandle);
};

/**
 *  Base class encapsulates ownership of buffers and interactions with channel in type-agnostic way.
 **/
class ManagedPortOutputBase : public ManagedPortBase
{
public:
    // Due to single TU checking
    // coverity[autosar_cpp14_a0_1_1_violation]
    static constexpr char LOG_TAG[]{"ManagedPortOutputBase"};

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
     *  @return vector of all GenericData allocated in the channel memory pool.
     **/
    dw::core::VectorFixed<GenericData> getAllBuffers();

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
     *  @param [in] publishTimestamp if no timestamp is passed, the current time is used to populate the metadata
     *  @note if the Port has no acquired buffer this call is a no-op.
     *  @note after this call the sent buffer will no longer be available.
     *        any pointers deduced from the previously held buffer should not be used.
     *  @throws if channel returns unexpected errors.
     **/
    void send(const dwTime_t* publishTimestamp = nullptr);

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

    void setNodeName(const dw::core::StringView& nodeName) noexcept override;

    ChannelObject::Producer* getChannelProducer()
    {
        return m_channelProducer;
    }

protected:
    explicit ManagedPortOutputBase(const dw::core::StringView& name, ConstructProperties props, GenericDataReference&& ref, std::type_index typeIndex);
    GenericData getBufferGeneric();

private:
    static BoundProperties getBoundProperties(const ChannelObject& channel);
    void preSend(ChannelMetadata& header, const dwTime_t* publishTimestamp);
    void checkFiInstances(bool& sendMessage);
    void checkFiInstanceInvalidateMessageHeaders();
    void checkFiInstanceZeroSequenceNumber();
    void checkFiInstanceDropMessages(bool& sendMessage);
    dwStatus maybeSendMessage(bool sendMessage);
    GenericDataReference m_ref;
    ChannelObject::Producer* m_channelProducer;
    Properties m_props;
    dw::core::Optional<GenericData> m_buffer;
    dw::core::Function<dwStatus()> m_callbackBeforeSend;
    uint32_t m_sendSeqNum;
    std::type_index m_typeIndex;
    bool m_isNvsci{false};
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

    /**
     *  @return vector of all GenericData allocated in the channel memory pool.
     **/
    dw::core::VectorFixed<GenericData> getAllBuffers();

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
    ManagedPortInputBase(const dw::core::StringView& name, ConstructProperties props, GenericDataReference&& ref);
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
    void handleWaitFailure(dwStatus status);
    void handleCallbackAfterRecv();

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

/**
 *  Derived ManagedPortOutput<T> provides type-specific interfaces for accessing buffers.
 **/
template <typename T>
class ManagedPortOutput : public ManagedPortOutputBase
{
    static_assert(std::is_constructible<T>::value, "T must be constructible");
    static_assert(parameter_traits<T>::IsDeclared,
                  "Channel packet type not declared. Ensure channel packet type "
                  "handling is declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_POD "
                  "or DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION");

    using SpecimenT = typename parameter_traits<T>::SpecimenT;

public:
    /**
     *   @param [in] name  the name of the output port.
     *   @param [in] props the properties of the output Port.
     *   @param [in] ref   the reference data to be used by channel to allocate
     *                     packets.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID != DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(const dw::core::StringView& name, ConstructProperties props, SpecimenT& ref)
        : ManagedPortOutputBase(name, std::move(props), make_specimen<T>(&ref), std::type_index(typeid(T)))
    {
    }

    /**
     *   @param [in] name  the name of the output port.
     *   @param [in] ref   the reference data to be used by channel to allocate
     *                     packets.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID != DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(const dw::core::StringView& name, SpecimenT& ref)
        : ManagedPortOutput(name, {}, ref)
    {
    }

    /**
     *   @param [in] name  the name of the output port.
     *   @param [in] props the properties of the output Port.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID == DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(const dw::core::StringView& name, ConstructProperties props)
        : ManagedPortOutputBase(name, std::move(props), make_specimen<T>(nullptr), std::type_index(typeid(T)))
    {
    }

    /**
     *   @param [in] name the name of the output port.
     **/
    template <typename T2 = T, typename std::enable_if_t<parameter_traits<T2>::PacketTID == DWFRAMEWORK_PACKET_ID_DEFAULT, void>* = nullptr>
    ManagedPortOutput(const dw::core::StringView& name)
        : ManagedPortOutput(name, {})
    {
    }

    /**
     *  iterable for all the buffers in the output channel pool.
     **/
    detail::vectorIterable<T> getAllBufferIter()
    {
        return detail::vectorIterable<T>(getAllBuffers());
    }

    /**
     *   Get a pointer to the acquired buffer.
     *   @note returned pointer is never nullptr
     *   @throws when the buffer cannot be cast to T.
     **/
    T* getBuffer()
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
    static_assert(std::is_constructible<T>::value, "T must be constructible");
    static_assert(parameter_traits<T>::IsDeclared,
                  "Channel packet type not declared. Ensure channel packet type "
                  "handling is declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_POD "
                  "or DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION");
    // coverity[autosar_cpp14_a0_1_6_violation]
    using SpecimenT = typename parameter_traits<T>::SpecimenT;

    struct PacketDeleter;

public:
    struct iterable;

    /**
     *   @param [in] name  the name of the input Port.
     *   @param [in] props the properties of the input Port.
     **/
    ManagedPortInput(const dw::core::StringView& name, ConstructProperties props)
        : ManagedPortInputBase(name, std::move(props), make_specimen<T>(nullptr))
    {
    }

    ManagedPortInput(const dw::core::StringView& name)
        : ManagedPortInput(name, ConstructProperties())
    {
    }

    ManagedPortInput(const dw::core::StringView& name, ConstructProperties props, SpecimenT specimen)
        : ManagedPortInputBase(name, std::move(props), make_specimen<T>(&specimen))
    {
    }

    ManagedPortInput(const dw::core::StringView& name, SpecimenT specimen)
        : ManagedPortInput(name, {}, specimen)
    {
    }

    /**
     *  iterable for all the buffers in the output channel pool.
     **/
    detail::vectorIterable<T> getAllBufferIter()
    {
        return detail::vectorIterable<T>(getAllBuffers());
    }

    /**
     *   Get iterator over received buffers
     *   @note returned pointers are never nullptr
     *   @note iterator access throws if buffer cannot be cast to T.
     **/
    iterable getBufferIter()
    {
        return iterable(*this);
    }

    /**
     *   Get a pointer to the first received buffer.
     *   @note returned pointer is never nullptr
     *   @throws when the buffer cannot be cast to T.
     **/
    auto getBuffer() -> T*
    {
        return detail::getBufferTyped<T>(getBufferGeneric());
    }

    /**
     *   For bind value if it is valid
     *   Get a pointer to the first received buffer if the buffer is valid or validity is not set.
     *   Otherwise return nullptr.
     **/
    auto getBufferIfAvailableAndValid() -> T*
    {
        if (isBufferAvailable())
        {
            const dw::core::Optional<dwValidityStatus> valid{getValidityStatus(getMetadata())};
            // At the moment if validity is not set it counts as valid
            // Once all the publishers set the validity signal it can be changed
            const bool isValid{!valid.has_value() || (DW_VALIDITY_VALID == valid->validity)};
            if (isValid)
            {
                return getBuffer();
            }
        }
        return nullptr;
    }

    /**
     *   For bind optional and buffer optional case
     *   Get a pointer to the first received buffer if the buffer available.
     *   Otherwise return nullptr.
     *   @throws when the buffer cannot be cast to T.
     **/
    auto getOptionalBuffer() -> T*
    {
        return isBufferAvailable() ? getBuffer() : nullptr;
    }

    using UniquePacketPtr = std::unique_ptr<T, PacketDeleter>;

    UniquePacketPtr takeOwnership()
    {
        GenericData packet{popBufferGeneric()};
        T* ptr{detail::getBufferTyped<T>(packet)};
        void* releasePtr{packet.getPointer()};
        return UniquePacketPtr(ptr, PacketDeleter{this, releasePtr});
    }

    struct iterable
    {
        explicit iterable(ManagedPortInput<T>& port)
            : m_port(port)
        {
        }

        /// Iterators
        template <class TT>
        class iterator : public ManagedPortInputBase::RingBuffer::iterator
        {
            static_assert(std::is_constructible<TT>::value, "TT must be constructible");

            using Base = ManagedPortInputBase::RingBuffer::iterator;

        public:
            // Same naming is used in dwshared, hence keeping the iterator name and its accessors for now
            iterator(Base&& base, ManagedPortInput<T>& port)
                : Base(std::move(base))
                , m_port(port)
            {
            }

            const Base& baseFromThis() const
            {
                return *this;
            }

            auto operator*() const -> TT*
            {
                GenericData buffer{*baseFromThis()};
                return detail::getBufferTyped<TT>(buffer);
            }

        private:
            ManagedPortInput<T>& m_port;
        };

        iterator<T> begin() { return iterator<T>(m_port.m_buffers.begin(), m_port); }

        iterator<T> end() { return iterator<T>(m_port.m_buffers.end(), m_port); }

        iterator<const T> begin() const { return iterator<const T>(m_port.m_buffers.begin(), m_port); }

        iterator<const T> end() const { return iterator<const T>(m_port.m_buffers.end(), m_port); }

    private:
        ManagedPortInput<T>& m_port;
    };

private:
    struct PacketDeleter
    {
        void operator()(T* p)
        {
            static_cast<void>(p);
            port->releaseToChannel(releasePtr);
        }
        ManagedPortInput* port;
        void* releasePtr;
    };
};

template <typename T>
using UniquePacketPtr = typename ManagedPortInput<T>::UniquePacketPtr;

// Create a port type specimen for a given port index
namespace detail
{
template <
    typename NodeT,
    PortDirection Direction,
    uint64_t DescriptorIndex>
struct IsOutputNonPOD : std::integral_constant<bool, Direction == PortDirection::OUTPUT && parameter_traits<decltype(portDescriptorType<NodeT, Direction, DescriptorIndex>())>::PacketTID != DWFRAMEWORK_PACKET_ID_DEFAULT>
{
    // static_assert(!std::is_pod<decltype(portDescriptorType<NodeT, Direction, DescriptorIndex>())>::value && parameter_traits<decltype(portDescriptorType<NodeT, Direction, DescriptorIndex>())>::IsDeclared, "The packet type is not yet declared.");
    static_assert(DescriptorIndex < portDescriptorSize<NodeT, Direction>(), "Invalid PortIndex.");
};

template <
    typename NodeT,
    PortDirection Direction,
    uint64_t DescriptorIndex>
typename std::enable_if<
    !IsOutputNonPOD<NodeT, Direction, DescriptorIndex>::value,
    GenericDataReference>::type
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    createPortSpecimenByStaticIndex()
{
    GenericDataReference ref{make_specimen<decltype(portDescriptorType<NodeT, Direction, DescriptorIndex>())>(nullptr)};
    ref.packetTypeID = dw::core::safeAdd(static_cast<dw::framework::ChannelPacketTypeID>(DWFRAMEWORK_METADATA_PACKET_TYPE_ID_OFFSET), static_cast<uint32_t>(ref.packetTypeID)).value();
    return ref;
}

template <
    typename NodeT,
    PortDirection Direction,
    uint64_t DescriptorIndex>
typename std::enable_if<
    IsOutputNonPOD<NodeT, Direction, DescriptorIndex>::value,
    GenericDataReference>::type
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    createPortSpecimenByStaticIndex()
{
    throw ExceptionWithStatus(DW_NOT_SUPPORTED, "createPortSpecimenByStaticIndex: Non POD output port is not supported");
}

template <
    typename NodeT,
    PortDirection Direction,
    uint64_t... Idx>
typename std::enable_if<(sizeof...(Idx) > 1), GenericDataReference>::type
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/4040101
    createPortSpecimenImpl(size_t descriptorIndex, std::index_sequence<Idx...>)
{
    constexpr size_t ArraySize{sizeof...(Idx)};
    if (descriptorIndex < ArraySize)
    {
        // coverity[autosar_cpp14_a20_8_4_violation] FP: nvbugs/4552679
        std::array<GenericDataReference, ArraySize> specimens{
            (Idx == descriptorIndex ? createPortSpecimenByStaticIndex<NodeT, Direction, Idx>() : GenericDataReference{})...};
        return specimens[descriptorIndex];
    }
    throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "createPortSpecimenImpl: index out of bound.");
}

// Above createPortSpecimenImpl actually covers correctly for all sizeof...(Idx)
// but AutoSAR complaining about dead code and unreachable code branch when sizeof...(Idx) == 1.
// Thus split it into three different implementations.
template <
    typename NodeT,
    PortDirection Direction,
    uint64_t... Idx>
typename std::enable_if<(sizeof...(Idx) == 1), GenericDataReference>::type
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/4040101
    createPortSpecimenImpl(size_t descriptorIndex, std::index_sequence<Idx...>)
{
    constexpr size_t ArraySize{sizeof...(Idx)};
    if (descriptorIndex < ArraySize)
    {
        return createPortSpecimenByStaticIndex<NodeT, Direction, 0>();
    }
    throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "createPortSpecimenImpl: index out of bound.");
}

template <
    typename NodeT,
    PortDirection Direction,
    uint64_t... Idx>
typename std::enable_if<(sizeof...(Idx) == 0), GenericDataReference>::type
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/4040101
    createPortSpecimenImpl(size_t, std::index_sequence<Idx...>)
{
    throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "createPortSpecimenImpl: index out of bound.");
}

template <
    typename NodeT,
    PortDirection Direction>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/4040101
GenericDataReference createPortSpecimen(size_t descriptorIndex)
{
    return detail::createPortSpecimenImpl<NodeT, Direction>(
        descriptorIndex,
        std::make_index_sequence<portDescriptorSize<NodeT, Direction>()>{});
}
} // namespace detail

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_MANAGEDPORT_HPP_
