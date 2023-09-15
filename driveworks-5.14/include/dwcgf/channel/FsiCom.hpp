///////////////////////////////////////////////////////////////////////////////////////

// Copyright (c) 2023 Mercedes-Benz AG. All rights reserved.
//
// Mercedes-Benz AG as copyright 2023 owner and NVIDIA Corporation as licensor retain
// all intellectual property and proprietary rights in and to this software
// and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement is strictly prohibited.
//
// This code contains Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE ARE MADE.
//
// No responsibility is assumed for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights. No third party distribution is allowed unless
// expressly authorized.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// Products are not authorized for use as critical
// components in life support devices or systems without express written approval.
///////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_FSI_COM_HPP_
#define DW_FRAMEWORK_FSI_COM_HPP_

#include <dw/core/base/BasicTypes.h>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

// coverity[autosar_cpp14_a1_1_1_violation]
#include <errno.h>

namespace dw
{
namespace framework
{

/**
 * Interface for FSI communication
 */
class IChannelFsiCom
{
public:
    // coverity[autosar_cpp14_a0_1_1_violation]
    // coverity[autosar_cpp14_a2_10_5_violation]
    static constexpr char LOG_TAG[]{"IChannelFsiCom"};

    /**
     * @brief Connect to a specified FSI endpoint
     * @return true if no errors were reported during connection, false otherwise
     */
    virtual bool connect() = 0;

    /**
     * @brief Check if FSI endpoint is connected
     * @return true if connected, false otherwise
     */
    virtual bool isConnected() = 0;

    /**
     * @brief Disconnect from a connected FSI endpoint
     */
    virtual void disconnect() = 0;

    /**
     * @brief Write to a connected FSI endpoint
     *
     * @param[in] buff the buffer to be written to FSI
     * @param[in] size the size of the buffer
     * @param[out] writtenBytes bytes written to FSI
     *
     * @return POSIX errno indicating if write is successful
     */
    virtual int32_t write(void* buff, uint32_t size, uint32_t& writtenBytes) = 0;

    /**
     * @brief Wait for a buffer to be available to read
     *
     * @param[in] timeout_us the amount of time to wait for event
     *
     * @return POSIX errno indicating if there is data available
     */
    virtual int32_t waitForEvent(dwTime_t timeout_us) = 0;

    /**
     * @brief Read from a connected FSI endpoint
     *
     * @param[inout] buff the buffer to be filled
     * @param[in] size the size of the buffer
     * @param[out] readBytes bytes read from FSI
     *
     * @return POSIX errno indicating if write is successful
     */
    virtual int32_t read(void* buff, uint32_t size, uint32_t& readBytes) = 0;

    using createFactory = std::function<std::shared_ptr<IChannelFsiCom>(uint8_t, const char*)>;
    /**
     * @brief Create the FSI channel
     *
     * @param[in] numChannels the numChannel parameter for FSI
     * @param[in] compat compat string for FSI
     * @param[in] overrider if provided, this custom function will be used for creating the channel
     *
     * @return the channel object
     */
    static std::shared_ptr<IChannelFsiCom> create(uint8_t numChannels, const char* compat, createFactory& overrider);

    /**
     * @brief Register a FSI connection.
     *
     * @param[in] consumer indicate if the channel to be registered is a consumer
     *
     * @return true if registration is successful. Registration is only successful if only a single consumer or producer is registered.
     */
    bool registerClient(bool consumer);

    /**
     * @brief Unregister a FSI connection.
     *
     * @param[in] consumer indicate if the channel to be unregistered is a consumer
     */
    void unregisterClient(bool consumer);

private:
    bool m_consumerRegistered{false};
    bool m_producerRegistered{false};
};

} // namespace framework
} // namespace dw

#endif
