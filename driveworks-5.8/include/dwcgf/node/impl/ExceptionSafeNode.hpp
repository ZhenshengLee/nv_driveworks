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

#include <dwcgf/Exception.hpp>

namespace dw
{
namespace framework
{
class ExceptionSafeProcessNode : public Node
{
public:
    explicit ExceptionSafeProcessNode(std::unique_ptr<Node> impl)
        : m_impl(std::move(impl))
    {
    }

    ~ExceptionSafeProcessNode() override = default;

    dwStatus reset() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reset();
        });
    }

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setInputChannel(channel, portID);
        });
    }

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID, dwSerializationType dataType) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setInputChannel(channel, portID, dataType);
        });
    }

    dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setOutputChannel(channel, portID);
        });
    }

    dwStatus validate() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->validate();
        });
    }

    dwStatus run() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->run();
        });
    }

    size_t getPassCount() const noexcept override
    {
        return m_impl->getPassCount();
    }

    dwStatus runPassByID(uint8_t passID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->runPassByID(passID);
        });
    }

    dwStatus runPass(size_t passIndex) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->runPass(passIndex);
        },
                                          dw::core::Logger::Verbosity::WARN);
    }

    dwStatus getPasses(VectorFixed<Pass*>& passList) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getPasses(passList);
        });
    }

    dwStatus getPasses(VectorFixed<Pass*>& passList,
                       dwProcessorType processorType,
                       dwProcessType processType) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getPasses(passList, processorType, processType);
        });
    }

    dwStatus setName(const char* name) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setName(name);
        });
    }

    dwStatus getName(const char** name) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getName(name);
        });
    }

    dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getErrorSignal(errorSignal);
        });
    }

    dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignal, bool updateFromModule = false) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getHealthSignal(healthSignal, updateFromModule);
        });
    }

    dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reportCurrentErrorSignal(signal);
        });
    }

    dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reportCurrentHealthSignal(signal);
        });
    }

    dwStatus setIterationCount(uint32_t iterationCount) override final
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setIterationCount(iterationCount);
        });
    }

    dwStatus setState(const char* state) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setState(state);
        });
    }

    void resetPorts() override
    {
        Exception::guard([&]() {
            return m_impl->resetPorts();
        });
    }

protected:
    std::unique_ptr<Node> m_impl;
};

class ExceptionSafeSensorNode : public Node, public ISensorNode
{
public:
    explicit ExceptionSafeSensorNode(std::unique_ptr<Node> impl)
        : m_impl(std::move(impl))
        , m_sensorNodeImpl(dynamic_cast<ISensorNode*>(m_impl.get()))
    {
        if (m_sensorNodeImpl == nullptr)
        {
            throw Exception(DW_INVALID_ARGUMENT, "Not a sensor node");
        }
    }

    ~ExceptionSafeSensorNode() override = default;

    dwStatus reset() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reset();
        });
    }

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setInputChannel(channel, portID);
        });
    }

    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID, dwSerializationType dataType) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setInputChannel(channel, portID, dataType);
        });
    }

    dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setOutputChannel(channel, portID);
        });
    }

    dwStatus validate() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->validate();
        });
    }

    dwStatus start() override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->start();
        });
    }

    dwStatus stop() override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->stop();
        });
    }

    dwStatus setAffinityMask(uint mask) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setAffinityMask(mask);
        });
    }

    dwStatus setThreadPriority(int prio) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setThreadPriority(prio);
        });
    }

    dwStatus setStartTime(dwTime_t startTime) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setStartTime(startTime);
        });
    }

    dwStatus setEndTime(dwTime_t endTime) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setEndTime(endTime);
        });
    }

    dwStatus run() override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->run();
        });
    }

    size_t getPassCount() const noexcept override
    {
        return m_impl->getPassCount();
    }

    dwStatus runPassByID(uint8_t passID) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->runPassByID(passID);
        });
    }

    dwStatus runPass(size_t passIndex) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->runPass(passIndex);
        },
                                          dw::core::Logger::Verbosity::WARN);
    }

    dwStatus getPasses(VectorFixed<Pass*>& passList) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getPasses(passList);
        });
    }

    dwStatus getPasses(VectorFixed<Pass*>& passList,
                       dwProcessorType processorType,
                       dwProcessType processType) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getPasses(passList, processorType, processType);
        });
    }

    dwStatus setName(const char* name) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setName(name);
        });
    }

    dwStatus getName(const char** name) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getName(name);
        });
    }

    dwStatus isVirtual(bool* isVirtualBool) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->isVirtual(isVirtualBool);
        });
    }

    dwStatus setDataEventReadCallback(DataEventReadCallback cb) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setDataEventReadCallback(cb);
        });
    }

    dwStatus setDataEventWriteCallback(DataEventWriteCallback cb) override
    {
        return Exception::guardWithReturn([&]() {
            return m_sensorNodeImpl->setDataEventWriteCallback(cb);
        });
    }

    dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getErrorSignal(errorSignal);
        });
    }

    dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignal, bool updateFromModule = false) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->getHealthSignal(healthSignal, updateFromModule);
        });
    }

    dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reportCurrentErrorSignal(signal);
        });
    }

    dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->reportCurrentHealthSignal(signal);
        });
    }

    dwStatus setIterationCount(uint32_t iterationCount) override final
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setIterationCount(iterationCount);
        });
    }

    dwStatus setState(const char* state) override
    {
        return Exception::guardWithReturn([&]() {
            return m_impl->setState(state);
        });
    }

    void resetPorts() override
    {
        Exception::guard([&]() {
            return m_impl->resetPorts();
        });
    }

protected:
    std::unique_ptr<Node> m_impl;
    ISensorNode* m_sensorNodeImpl;
};
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_EXCEPTIONSAFENODE_HPP_
