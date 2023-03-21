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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_EXCEPTIONSAFENODE_HPP_
#define DW_FRAMEWORK_EXCEPTIONSAFENODE_HPP_

#include <dwcgf/node/Node.hpp>

namespace dw
{
namespace framework
{
class ExceptionSafeProcessNode : public Node
{
public:
    explicit ExceptionSafeProcessNode(std::unique_ptr<Node> impl);

    ~ExceptionSafeProcessNode() override = default;

    dwStatus reset() override;

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override;

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID, dwSerializationType dataType) override;

    dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) override;

    dwStatus validate() override;

    dwStatus run() override;

    size_t getPassCount() const noexcept override;

    dwStatus runPass(size_t passIndex) override;

    dwStatus getPass(Pass** pass, uint8_t index) override;

    dwStatus getPasses(VectorFixed<Pass*>& passList) override;

    dwStatus getPasses(VectorFixed<Pass*>& passList,
                       dwProcessorType processorType,
                       dwProcessType processType) override;

    dwStatus setName(const char* name) override;

    dwStatus getName(const char** name) override;

    dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) override;

    dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignal, bool updateFromModule = false) override;

    dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) override;

    dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) override;

    dwStatus setIterationCount(uint32_t iterationCount) override final;

    dwStatus setNodePeriod(uint32_t period) override final;

    dwStatus setState(const char* state) override;

    void resetPorts() override;

    dwStatus getInputChannel(const uint8_t portID, ChannelObject*& channel) const override;

    dwStatus getOutputChannel(const uint8_t portID, ChannelObject*& channel) const override;

    dwStatus getInputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

    dwStatus getOutputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

protected:
    std::unique_ptr<Node> m_impl;
};

class ExceptionSafeSensorNode : public Node, public ISensorNode
{
public:
    explicit ExceptionSafeSensorNode(std::unique_ptr<Node> impl);

    ~ExceptionSafeSensorNode() override = default;

    dwStatus reset() override;

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override;

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID, dwSerializationType dataType) override;

    dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) override;

    dwStatus validate() override;

    dwStatus start() override;

    dwStatus stop() override;

    dwStatus setAffinityMask(uint mask) override;

    dwStatus setThreadPriority(int prio) override;

    dwStatus setStartTime(dwTime_t startTime) override;

    dwStatus setEndTime(dwTime_t endTime) override;

    dwStatus run() override;

    size_t getPassCount() const noexcept override;

    dwStatus runPass(size_t passIndex) override;

    dwStatus getPass(Pass** pass, uint8_t index) override;

    dwStatus getPasses(VectorFixed<Pass*>& passList) override;

    dwStatus getPasses(VectorFixed<Pass*>& passList,
                       dwProcessorType processorType,
                       dwProcessType processType) override;

    dwStatus setName(const char* name) override;

    dwStatus getName(const char** name) override;

    dwStatus isVirtual(bool* isVirtualBool) override;

    dwStatus setDataEventReadCallback(DataEventReadCallback cb) override;

    dwStatus setDataEventWriteCallback(DataEventWriteCallback cb) override;

    dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) override;

    dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignal, bool updateFromModule = false) override;

    dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) override;

    dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) override;

    dwStatus setIterationCount(uint32_t iterationCount) override final;

    dwStatus setNodePeriod(uint32_t period) override final;

    dwStatus setState(const char* state) override;

    void resetPorts() override;

    dwStatus getInputChannel(const uint8_t portID, ChannelObject*& channel) const override;

    dwStatus getOutputChannel(const uint8_t portID, ChannelObject*& channel) const override;
    void setNodeImple(std::unique_ptr<Node> impl);

    dwStatus getInputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

    dwStatus getOutputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

    dwStatus setLockstepDeterministicMode(bool enable) final;

    dwStatus getNextTimestamp(dwTime_t& nextTimestamp) final;

protected:
    std::unique_ptr<Node> m_impl;
    ISensorNode* m_sensorNodeImpl;
};
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_EXCEPTIONSAFENODE_HPP_
