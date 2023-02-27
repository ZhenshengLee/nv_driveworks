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
#include <dwcgf/channel/ChannelTrace.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <nvscisync.h>

namespace dw
{
namespace framework
{

using ChannelPacketConstructor             = dw::core::Function<std::unique_ptr<IChannelPacket>(GenericData dataSpecimen, dwContextHandle_t context)>;
using ChannelPacketConstructorSignature    = std::pair<ChannelPacketTypeID, ChannelType>;
using ChannelPacketConstructorRegistration = std::pair<ChannelPacketConstructorSignature, ChannelPacketConstructor>;

class ChannelFactoryImpl;

class ChannelFactory
{
public:
    static void registerPacketConstructor(const ChannelPacketConstructorSignature& signature, const ChannelPacketConstructor& constructor);
    static void unregisterPacketConstructor(const ChannelPacketConstructorSignature& signature);

    ChannelFactory(dwContextHandle_t ctx = DW_NULL_HANDLE);
    virtual ~ChannelFactory()            = default;
    // Create a channel with given channelParams
    std::shared_ptr<ChannelObject> makeChannel(const char* channelParams);

    using OnDispatchDataReady = dw::core::Function<void(void* opaque, ChannelObject::PacketPool::OnDataReady)>;
    void setOnDispatchDataReady(OnDispatchDataReady dispatchDataReady);

    void setTraceMode(ChannelTraceMode traceMode);

    void setOnRegisterTraceWriter(ChannelOnRegisterTraceWriter onRegisterTraceWriter);
    void setOnRegisterTraceReader(ChannelOnRegisterTraceReader onRegisterTraceReader);

    // Get the packet factory so registered packet constructors can be accessed directly
    ChannelPacketFactoryPtr getPacketFactory();

    // Get the nvscisync module used by underlying NvSciStream channels. This NvSciSyncModule is owned
    // internally by channels and shall not be freed by application.
    NvSciSyncModule getNvSciSyncModule();

    void stopServices();

private:
    std::shared_ptr<ChannelFactoryImpl> m_pimpl;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ICHANNELFACTORY_HPP_
