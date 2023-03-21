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

#ifndef DW_FRAMEWORK_ICHANNELFACTORY_HPP_
#define DW_FRAMEWORK_ICHANNELFACTORY_HPP_

#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dw/core/health/HealthSignals.h>
#include <nvscisync.h>

namespace dw
{
namespace framework
{

using ChannelPacketConstructor             = dw::core::Function<std::unique_ptr<IChannelPacket>(GenericData dataSpecimen, dwContextHandle_t context)>;
using ChannelPacketConstructorSignature    = std::pair<ChannelPacketTypeID, ChannelType>;
using ChannelPacketConstructorRegistration = std::pair<ChannelPacketConstructorSignature, ChannelPacketConstructor>;

class ChannelFactoryImpl;

enum class ChannelEventType
{
    CONNECTED,    ///< Channel became connected
    DISCONNECTED, ///< Channel became disconnected
    ERROR,        ///< Channel encountered error
    READY         ///< Channel is ready
};

struct ChannelEvent
{
    ChannelEventType type;                              ///< The type of the event
    uint32_t uid;                                       ///< unique identifier of the channel
    void* opaque;                                       ///< The opaque pointer set for the channel tthat is ready. Only valid for type=READY
    ChannelObject::PacketPool::OnDataReady onDataReady; ///< The callback set for the channel that is ready. Only valid for type=READY
    dwErrorSignal errorSignal;                          ///< The error signal. Only valid for type=ERROR
};

/**
 * Shared allocator and context for Channels.
 */ class ChannelFactory
{
public:
    /**
     * Register a packet constructor to the factory.
     *
     * @note A packet class must be registered to the ChannelFactory before a corresponding ChannelObject which the application intends
     *       to use with said packet class is created.
     * @note Thread-safe
     * @note init-time only
     *
     * @param signature The signature under chich to register the constructor.
     * @param constructor The constructor to register.
     */
    static void registerPacketConstructor(const ChannelPacketConstructorSignature& signature, const ChannelPacketConstructor& constructor);
    /**
     * Unregister a packet constructor from the factory.
     *
     * @note Thread-safe
     * @note deinit-time only
     *
     * @param signature The signature to unregister.
     */
    static void unregisterPacketConstructor(const ChannelPacketConstructorSignature& signature);

    /**
     * @brief Construct a new Channel Factory object
     * 
     * @param ctx the DriveWorks context
     */
    ChannelFactory(dwContextHandle_t ctx = DW_NULL_HANDLE);
    virtual ~ChannelFactory()            = default;

    /**
     * Create a Channel
     *
     * @param channelParams the parameters of the Channel
     * @return std::shared_ptr<ChannelObject>
     */
    std::shared_ptr<ChannelObject> makeChannel(const char* channelParams);

    // Get the packet factory so registered packet constructors can be accessed directly
    ChannelPacketFactoryPtr getPacketFactory();

    // Get the nvscisync module used by underlying NvSciStream channels. This NvSciSyncModule is owned
    // internally by channels and shall not be freed by application.
    NvSciSyncModule getNvSciSyncModule();

    /**
     * Start the background services of the Channels.
     *
     * @note Until services are started, Channels may not properly connect or get notified when new data is ready.
     * @note Performance to create Channels will be very slow due to contention of background services.
     *       Hence it is recommended to create all Channels before starting services.
     *
     * @note Not Yet Implemented.
     */
    void startServices();

    /**
     * Stop the background services of the Channels.
     *
     * @note While services are stopped, Channels may not properly be able to get notified when new data is ready.
     * @note Performance to delete Channels will be very slow due to contention of background services.
     *       Hence it is recommended to stop services before deleting any Channels.
     */
    void stopServices();

    using OnDispatchDataReady = dw::core::Function<void(void* opaque, ChannelObject::PacketPool::OnDataReady)>;
    void setOnDispatchDataReady(OnDispatchDataReady dispatchDataReady);

    /**
     * Get the oldest event.
     *
     * @note this method is not thread-safe.
     *
     * @param [out] event   event to wait for
     * @param [in] timeout  timeout for the wait
     * @return true if successful
     *         false if no event came
     */
    bool getEvent(ChannelEvent& event, dwTime_t timeout);

private:
    std::shared_ptr<ChannelFactoryImpl> m_pimpl;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ICHANNELFACTORY_HPP_
