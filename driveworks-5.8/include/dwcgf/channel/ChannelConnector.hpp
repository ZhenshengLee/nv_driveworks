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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNELCONNECTOR_HPP_
#define DW_FRAMEWORK_CHANNELCONNECTOR_HPP_

#include <dw/core/base/Types.h>

#include <memory>
#include <dwcgf/channel/Channel.hpp>

namespace dw
{
namespace framework
{

class ChannelConnectorImpl;

class ChannelConnector
{
public:
    static constexpr uint32_t DEFAULT_MAX_CHANNELS = 512;
    using OnChannelsConnected                      = dw::core::Function<void()>;

    explicit ChannelConnector(size_t numChannels = DEFAULT_MAX_CHANNELS);

    ~ChannelConnector();

    dwStatus addChannel(std::shared_ptr<ChannelObject> channel);

    void start();

    void stop();

    // blocking wait until channels connected. Returns true if channels connected
    bool waitUntilConnected(dwTime_t timeout);

    // set callback on all channels connected.
    void setOnChannelsConnected(OnChannelsConnected onChannelsConnected);

    void logUnconnectedChannels();

private:
    std::unique_ptr<ChannelConnectorImpl> m_impl;
};

} // namespace framework
} // namespace dw

#endif
