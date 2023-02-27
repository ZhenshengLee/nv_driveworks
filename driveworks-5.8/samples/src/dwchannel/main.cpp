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

#include <stdlib.h>
#include <thread>
#include <iostream>
#include <string>
#include <numeric>
#include <unistd.h>
#include <ctime>
#include <future>
#include <sstream>

#include <framework/ProgramArguments.hpp>

#include <dw/core/context/Context.h>
#include <dw/core/base/Version.h>

#include <dwcgf/channel/ChannelFactory.hpp>
#include <dwcgf/channel/ChannelConnector.hpp>
#include <dwcgf/port/Port.hpp>

#include "ChannelPacketTypes.hpp"
#include "CustomRawBuffer.hpp"

#include <dwframework/dwnodes/common/factories/DWChannelFactory.hpp>
#include <framework/Log.hpp>

using namespace dw::framework;

static constexpr size_t MAX_DOWNSTREAM_CONSUMERS = 64U;

static constexpr char LOG_TAG[] = "DWChannelSample";

#define CHECK_DW_ERROR(x)                                                                                                                                                                                \
    {                                                                                                                                                                                                    \
        dwStatus RESULT = x;                                                                                                                                                                             \
        if (RESULT != DW_SUCCESS)                                                                                                                                                                        \
        {                                                                                                                                                                                                \
            throw std::runtime_error(std::string("DW Error ") + dwGetStatusName(RESULT) + std::string(" executing DW function:\n " #x) + std::string("\n at " __FILE__ ":") + std::to_string(__LINE__)); \
        }                                                                                                                                                                                                \
    };

static inline std::string getNvSciErrorStr(NvSciError sciError)
{
    std::stringstream ss{};
    ss << static_cast<uint32_t>(sciError);
    std::string s;
    ss >> s;
    return s;
}

#define CHECK_NVSCI_ERROR(x)                                                                                                                                                                                    \
    {                                                                                                                                                                                                           \
        NvSciError RESULT = x;                                                                                                                                                                                  \
        if (RESULT != NvSciError_Success)                                                                                                                                                                       \
        {                                                                                                                                                                                                       \
            throw std::runtime_error(std::string("NvSci Error ") + getNvSciErrorStr(RESULT) + std::string(" executing NvSci function:\n " #x) + std::string("\n at " __FILE__ ":") + std::to_string(__LINE__)); \
        }                                                                                                                                                                                                       \
    };

// Define simple template for handling the supported data types
// of this sample app, either IntWithTimestamp, dwImage, or CustomRawPacket.
// That way the main app class can be re-used for each of the data types.
template <typename PacketT>
struct TypeCallbacks;

struct IntWithTimestamp
{
    int count;
    dwTime_t timestamp;
};

template <>
struct TypeCallbacks<IntWithTimestamp>
{
    static IntWithTimestamp getSpecimen()
    {
        return {};
    }

    static size_t getSize()
    {
        return sizeof(IntWithTimestamp);
    }

    static dwTime_t getTimestamp(IntWithTimestamp data)
    {
        return data.timestamp;
    }

    static void setTimestamp(IntWithTimestamp& data, dwTime_t time)
    {
        data.timestamp = time;
    }
};

template <>
struct TypeCallbacks<dwImageHandle_t>
{
    static constexpr size_t IMAGE_WIDTH  = 3860;
    static constexpr size_t IMAGE_HEIGHT = 2160;

    static dwImageProperties getSpecimen()
    {
        // This can be changed to change the parameters of the images to
        // be streamed.
        dwImageProperties prop{.type   = DW_IMAGE_CUDA,
                               .width  = IMAGE_WIDTH,
                               .height = IMAGE_HEIGHT,
                               .format = DW_IMAGE_FORMAT_YUV420_UINT8_PLANAR};
        prop.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH;

        return prop;
    }

    static size_t getSize()
    {
        auto prop          = getSpecimen();
        size_t elementSize = 0;
        size_t planeCount  = 0;

        uint32_t planeChannelCount[DW_MAX_IMAGE_PLANES] = {0};
        dwVector2ui planeSize[DW_MAX_IMAGE_PLANES]      = {0};

        size_t imageSize = 0;

        CHECK_DW_ERROR(dwImage_getDataLayout(&elementSize, &planeCount, planeChannelCount, planeSize, &prop));

        for (size_t i = 0; i < planeCount; i++)
        {
            imageSize += elementSize * planeSize[i].x * planeChannelCount[i] * planeSize[i].y;
        }

        return imageSize;
    }

    static dwTime_t getTimestamp(dwImageHandle_t img)
    {
        dwTime_t packetTime;
        CHECK_DW_ERROR(dwImage_getTimestamp(&packetTime, img));
        return packetTime;
    }

    static void setTimestamp(dwImageHandle_t img, dwTime_t time)
    {
        CHECK_DW_ERROR(dwImage_setTimestamp(time, img));
    }
};

template <>
struct TypeCallbacks<CustomRawBuffer>
{
    static CustomRawBuffer getSpecimen()
    {
        // This can be changed to change the properties of the CustomRawBuffer being streamed.
        CustomRawBuffer specimen{};
        specimen.capacity   = 256;
        specimen.memoryType = MemoryType::CPU;
        return specimen;
    }

    static size_t getSize()
    {
        return sizeof(CustomRawBuffer);
    }

    static dwTime_t getTimestamp(CustomRawBuffer& data)
    {
        return data.timestamp;
    }

    static void setTimestamp(CustomRawBuffer& data, dwTime_t time)
    {
        data.timestamp = time;
    }
};

static void setCpuWaiterAttributes(NvSciSyncAttrList attrList)
{
    const bool cpuAccess           = true;
    const NvSciSyncAccessPerm perm = NvSciSyncAccessPerm_WaitOnly;
    dw::core::Array<NvSciSyncAttrKeyValuePair, 2U> pairArray = {{{NvSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
                                                                 {NvSciSyncAttrKey_RequiredPerm, &perm, sizeof(perm)}}};
    CHECK_NVSCI_ERROR(NvSciSyncAttrListSetAttrs(attrList, &pairArray[0], pairArray.size()));
}

static void setCpuSignalerAttributes(NvSciSyncAttrList attrList)
{
    const bool cpuAccess           = true;
    const NvSciSyncAccessPerm perm = NvSciSyncAccessPerm_SignalOnly;
    dw::core::Array<NvSciSyncAttrKeyValuePair, 2U> pairArray = {{{NvSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
                                                                 {NvSciSyncAttrKey_RequiredPerm, &perm, sizeof(perm)}}};
    CHECK_NVSCI_ERROR(NvSciSyncAttrListSetAttrs(attrList, &pairArray[0], pairArray.size()));
}

enum class SyncMode
{
    NONE,
    P2C,
    C2P,
    BOTH
};

template <typename PacketT>
class DWChannelSample
{
    using PacketTSpec = typename parameter_traits<PacketT>::SpecimenT;

private:
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwImageProperties m_props{};

    bool m_hasProducer = true;

    uint32_t m_numConsumers = 1;
    uint32_t m_downStreams  = 1;

    std::string m_portId{};
    std::string m_ipAddr{};
    std::string m_type{};
    std::string m_channelMode{};
    std::string m_prodReaches{};
    std::string m_prodStreamNames{};
    std::vector<std::string> m_consReaches{};
    std::vector<std::string> m_consStreamNames{};

    bool m_isMailBox   = false;
    bool m_isReuse     = false;
    int32_t m_fifoSize = -1;
    uint32_t m_maxFrameNumber{128U};
    SyncMode m_syncMode{};

    NvSciSyncCpuWaitContext m_cpuWaitContext{};

    static constexpr size_t SKIP_FRAME_NUMBER = 0;
    static constexpr size_t MAX_CONSUMER      = 4;
    static constexpr size_t WAIT_TIMEOUT_US   = 10'000'000;

    // Channel factory
    std::unique_ptr<ChannelFactory> m_channelFactory{};
    std::unique_ptr<ChannelConnector> m_channelConnector{};

    // Channels
    std::shared_ptr<ChannelObject> m_inputChannels[MAX_CONSUMER]{};
    std::shared_ptr<ChannelObject> m_outputChannel{};

    // Ports
    std::unique_ptr<dw::framework::PortInput<PacketT>> m_inputPorts[MAX_CONSUMER]{};
    std::unique_ptr<dw::framework::PortOutput<PacketT>> m_outputPort{};

    std::vector<std::thread> m_consumerThreads{};
    std::thread m_producerThread{};

    std::vector<dwTime_t> m_sendLatencyArr[MAX_CONSUMER]{};

    NvSciSyncObj m_producerSignalerSyncObj{};
    std::vector<NvSciSyncObj> m_consumerWaiterSyncObjs{};
    std::vector<NvSciSyncObj> m_producerWaiterSyncObjs{};
    std::vector<NvSciSyncObj> m_consumerSignalerSyncObjs{};
    std::vector<NvSciSyncFence> m_producerWaitFences{};

    dwTime_t m_prodElapsed{};
    dwTime_t m_consElapsed[MAX_CONSUMER]{};

    ProgramArguments m_args{};

    const std::string& getArgument(const char* name) const
    {
        return m_args.get(name);
    }

    void initContext()
    {
        // initialize logger to print info message on console
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, nullptr));
    }

    std::vector<std::string> splitString(std::string str, char splitChar)
    {
        std::vector<std::string> result{};
        std::istringstream ss(str);
        for (std::string line; std::getline(ss, line, splitChar);)
        {
            result.push_back(line);
        }
        return result;
    }

    SyncMode parseSyncMode(std::string syncMode)
    {
        SyncMode out;
        if (syncMode == "p2c")
        {
            out = SyncMode::P2C;
        }
        else if (syncMode == "c2p")
        {
            out = SyncMode::C2P;
        }
        else if (syncMode == "both")
        {
            out = SyncMode::BOTH;
        }
        else if (syncMode == "none")
        {
            out = SyncMode::NONE;
        }
        else
        {
            throw std::runtime_error("Sync mode not recognized");
        }

        return out;
    }

    bool p2cEnabled()
    {
        return m_syncMode == SyncMode::BOTH || m_syncMode == SyncMode::P2C;
    }

    bool c2pEnabled()
    {
        return m_syncMode == SyncMode::BOTH || m_syncMode == SyncMode::C2P;
    }

    void parseAndCheckArguments()
    {
        // parse and check channel type
        m_type = getArgument("type");
        if (m_type == "NVSCI")
        {
            m_prodReaches     = getArgument("prod-reaches");
            m_prodStreamNames = getArgument("prod-stream-names");
            m_consReaches     = splitString(getArgument("cons-reaches"), ':');
            m_consStreamNames = splitString(getArgument("cons-stream-names"), ':');
            if (m_consReaches.size() != m_consStreamNames.size())
            {
                throw std::runtime_error("Size of cons-reaches does not match cons-stream-names. They must be equal");
            }
            m_hasProducer  = !m_prodReaches.empty();
            m_numConsumers = m_consReaches.size();
        }
        else
        {
            uint32_t prod = std::atoi(getArgument("prod").c_str());
            if (prod > 1)
            {
                throw std::runtime_error("Number of Producer must be 0 or 1.");
            }

            // parse and check producer with downstream.
            m_hasProducer = static_cast<bool>(prod);
            m_downStreams = std::atoi(getArgument("downstreams").c_str());
            if ((m_hasProducer && m_downStreams == 0) || (!m_hasProducer && m_downStreams >= 1))
            {
                throw std::runtime_error("downstreams argument must with prod.");
            }

            // parse and check number of consumer.
            m_numConsumers = std::atoi(getArgument("cons").c_str());
            if (m_hasProducer && (m_numConsumers > m_downStreams))
            {
                throw std::runtime_error("Number of consumers cannot exclude downStreams.");
            }
            if (m_numConsumers > MAX_CONSUMER)
            {
                throw std::runtime_error("Number of consumers cannot exclude to maximum number of consumer this sample.");
            }

            if (m_type == "SHMEM_LOCAL")
            {
                if (m_numConsumers != m_downStreams)
                {
                    throw std::runtime_error("Number of downStreams (downstream) and Number of Consumers (cons) must equal when use SHMEM, as consumers and producers are in a same process.");
                }
            }
            else if (m_type != "SOCKET")
            {
                throw std::runtime_error("Invalid ChannelType.");
            }
        }

        // parse and check channel mode
        m_channelMode = getArgument("mode");
        if (m_channelMode == "mailbox")
        {
            m_isMailBox = true;
        }
        else if (m_channelMode == "reuse")
        {
            m_isMailBox = true;
            m_isReuse   = true;
        }
        else if (std::atoi(m_channelMode.c_str()) > 0)
        {
            m_fifoSize = std::atoi(m_channelMode.c_str());
        }
        else
        {
            throw std::runtime_error("Invalid ChannelMode");
        }

        m_ipAddr = getArgument("ip");
        m_portId = getArgument("port");

        m_maxFrameNumber = atoi(getArgument("frames").c_str());

        m_syncMode = parseSyncMode(getArgument("sync-mode"));

        return;
    }

    dw::framework::OnSetSyncAttrs getCpuWaiterAttrs(bool isProducer)
    {
        if (m_syncMode == SyncMode::BOTH || ((m_syncMode == SyncMode::C2P) && isProducer) || ((m_syncMode == SyncMode::P2C) && !isProducer))
        {
            return [](NvSciSyncAttrList attrList) {
                setCpuWaiterAttributes(attrList);
            };
        }
        return {};
    }

    dw::framework::OnSetSyncAttrs getCpuSignalerAttrs(bool isProducer)
    {
        if (m_syncMode == SyncMode::BOTH || ((m_syncMode == SyncMode::P2C) && isProducer) || ((m_syncMode == SyncMode::C2P) && !isProducer))
        {
            return [](NvSciSyncAttrList attrList) {
                setCpuSignalerAttributes(attrList);
            };
        }
        return {};
    }

    void initChannels()
    {
        // Create Factory and Connector
        m_channelFactory   = std::make_unique<dw::framework::DWChannelFactory>(m_context);
        m_channelConnector = std::make_unique<dw::framework::ChannelConnector>();

        // Create channels, ports and dataPtr
        for (size_t consId = 0; consId < m_numConsumers; consId++)
        {
            std::string channelParam = "role=consumer,type=" + m_type + ",ip=" + m_ipAddr + ",id=" + m_portId + ",timeout=1000";
            if (m_isMailBox)
            {
                channelParam += ",mode=mailbox";
                if (m_isReuse)
                {
                    channelParam += ",reuse=true";
                }
            }
            else
            {
                channelParam += (",fifo-size=" + m_channelMode);
            }
            if (m_type == "NVSCI")
            {
                channelParam += (",streamName=" + m_consStreamNames[consId]);
                channelParam += (",reach=" + m_consReaches[consId]);
            }
            std::cout << "Creating channel with parameters: " << channelParam << std::endl;
            m_inputChannels[consId] = m_channelFactory->makeChannel(channelParam.c_str());
            PacketTSpec ref         = TypeCallbacks<PacketT>::getSpecimen();
            m_inputPorts[consId]    = std::make_unique<dw::framework::PortInput<PacketT>>(ref, getCpuWaiterAttrs(false), getCpuSignalerAttrs(false));
            CHECK_DW_ERROR(m_inputPorts[consId]->bindChannel(m_inputChannels[consId].get()));
            m_channelConnector->addChannel(m_inputChannels[consId]);
        }

        if (m_hasProducer)
        {
            std::string channelParam = "role=producer,type=" + m_type + ",ip=" + m_ipAddr + ",id=" + m_portId + ",downstreams=" + std::string(getArgument("downstreams")) + ",producer-fifo=1,fifo-size=4,timeout=1000";
            if (m_type == "NVSCI")
            {
                channelParam += (",streamName=" + m_prodStreamNames);
                channelParam += (",reach=" + m_prodReaches);
            }
            std::cout << "Creating channel with parameters: " << channelParam << std::endl;
            m_outputChannel = m_channelFactory->makeChannel(channelParam.c_str());
            PacketTSpec ref = TypeCallbacks<PacketT>::getSpecimen();
            m_outputPort    = std::make_unique<dw::framework::PortOutput<PacketT>>(ref, getCpuSignalerAttrs(true), getCpuWaiterAttrs(true));
            CHECK_DW_ERROR(m_outputPort->bindChannel(m_outputChannel.get()));
            m_channelConnector->addChannel(m_outputChannel);
        }

        if (m_syncMode != SyncMode::NONE)
        {
            CHECK_NVSCI_ERROR(NvSciSyncCpuWaitContextAlloc(m_channelFactory->getNvSciSyncModule(), &m_cpuWaitContext));
        }
    }

    void connectChannels()
    {
        // start connection thread
        m_channelConnector->start();
        // Break when: 1. Each producer connect at least one consumer. 2.All consumers are connected. Are both satisfied.
        while (!m_channelConnector->waitUntilConnected(0))
        {
            usleep(100);
        }
        // stop connection thread
        m_channelConnector->stop();

        if (p2cEnabled())
        {
            if (m_hasProducer)
            {
                auto span = dw::core::make_span<NvSciSyncObj>(&m_producerSignalerSyncObj, 1);
                m_outputPort->getSyncSignaler().getSyncObjs(span);
            }

            for (uint32_t i = 0U; i < m_numConsumers; i++)
            {
                m_consumerWaiterSyncObjs.push_back(nullptr);
                auto span = dw::core::make_span<NvSciSyncObj>(&m_consumerWaiterSyncObjs.back(), 1);
                m_inputPorts[i]->getSyncWaiter().getSyncObjs(span);
            }
        }

        if (c2pEnabled())
        {
            if (m_hasProducer)
            {
                m_producerWaiterSyncObjs.resize(MAX_DOWNSTREAM_CONSUMERS, nullptr);
                m_producerWaitFences.resize(MAX_DOWNSTREAM_CONSUMERS, NvSciSyncFenceInitializer);
                auto span = dw::core::make_span<NvSciSyncObj>(&m_producerWaiterSyncObjs[0], MAX_DOWNSTREAM_CONSUMERS);
                m_outputPort->getSyncWaiter().getSyncObjs(span);
            }

            for (uint32_t i = 0U; i < m_numConsumers; i++)
            {
                m_consumerSignalerSyncObjs.push_back(nullptr);
                auto span = dw::core::make_span<NvSciSyncObj>(&m_consumerSignalerSyncObjs.back(), 1);
                m_inputPorts[i]->getSyncSignaler().getSyncObjs(span);
            }
        }
    }

    dwTime_t getClockRealtime()
    {
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);
        dwTime_t result = time.tv_sec * 1000000000 + time.tv_nsec;
        return result;
    }

    dwTime_t getClockMonotonic()
    {
        struct timespec time;
        clock_gettime(CLOCK_MONOTONIC, &time);
        dwTime_t result = time.tv_sec * 1000000000 + time.tv_nsec;
        return result;
    }

    void runConsumers(size_t consId)
    {
        const auto& inputPort = m_inputPorts[consId];

        // make sure the port is bound to a channel
        if (!inputPort->isBound())
        {
            std::cout << "Consumer[" << consId << "] doesn't bind port." << std::endl;
            return;
        }

        // get threadStart time for perf measurement
        dwTime_t threadStartTime = getClockMonotonic();

        std::future<void> future{};

        size_t currFrame = 0;
        while (currFrame < m_maxFrameNumber)
        {
            // recv slot
            dwStatus ret = inputPort->wait(WAIT_TIMEOUT_US);
            if (ret == DW_TIME_OUT)
            {
                std::cout << "Consumer[" << consId << "] exit by timeout." << std::endl;
                break;
            }

            std::shared_ptr<PacketT> inputPtr = inputPort->recv();

            if (inputPtr == nullptr)
            {
                throw std::runtime_error("Wait Success but recv failure.");
            }

            NvSciSyncFence waitFence = NvSciSyncFenceInitializer;
            if (p2cEnabled())
            {
                auto span = dw::core::make_span<NvSciSyncFence>(&waitFence, 1);
                inputPort->getWaitFences(inputPtr.get(), span);
            }

            if (c2pEnabled())
            {
                // set signal fence
                NvSciSyncFence fence = NvSciSyncFenceInitializer;
                CHECK_NVSCI_ERROR(NvSciSyncObjGenerateFence(m_consumerSignalerSyncObjs[consId], &fence));
                auto span = dw::core::make_span<NvSciSyncFence>(&fence, 1);
                inputPort->setSignalFences(inputPtr.get(), span);
            }

            auto work = [ this, inputPacket = inputPtr.get(), waitFence, consId ]()
            {
                if (p2cEnabled())
                {
                    CHECK_NVSCI_ERROR(NvSciSyncFenceWait(&waitFence, m_cpuWaitContext, -1));
                }

                // collect receive latency
                dwTime_t packetTime = TypeCallbacks<PacketT>::getTimestamp(*inputPacket);
                dwTime_t current    = getClockRealtime();
                m_sendLatencyArr[consId].push_back((current - packetTime) / 1000);

                if (c2pEnabled())
                {
                    // signal the sync object that frame is done
                    CHECK_NVSCI_ERROR(NvSciSyncObjSignal(m_consumerSignalerSyncObjs[consId]));
                }
            };

            if (c2pEnabled())
            {
                if (future.valid())
                {
                    future.wait();
                }
                future = std::async(std::launch::async, work);
            }
            else
            {
                work(); // must wait for work to complete before we return packet.
            }

            // release slot
            inputPtr = nullptr;
            currFrame++;
        }

        dwTime_t threadEndTime = getClockMonotonic();
        m_consElapsed[consId]  = threadEndTime - threadStartTime;
    }

    void runProducer()
    {
        // get threadStart time for perf measurement
        dwTime_t threadStartTime = getClockMonotonic();

        std::future<void> future{};

        size_t currFrame = 0;
        while (currFrame < m_maxFrameNumber)
        {
            // get slot from channel memory pool
            dwStatus status = m_outputPort->wait(2000000);
            if (status != DW_SUCCESS)
            {
                throw std::runtime_error("Producer timeout when acquiring packet.");
            }
            PacketT* outputFreePacket = m_outputPort->getFreeElement();

            if (outputFreePacket == nullptr)
            {
                throw std::runtime_error("Producer doesn't have enough empty slot.");
            }

            if (c2pEnabled())
            {
                // get wait fences
                auto span = dw::core::make_span<NvSciSyncFence>(&m_producerWaitFences[0], m_producerWaitFences.size());
                m_outputPort->getWaitFences(outputFreePacket, span);
                if (span.size() != m_producerWaitFences.size())
                {
                    throw std::runtime_error("Producer wait fence span size mismatch");
                }
            }

            if (p2cEnabled())
            {
                // set signal fences
                NvSciSyncFence fence = NvSciSyncFenceInitializer;
                CHECK_NVSCI_ERROR(NvSciSyncObjGenerateFence(m_producerSignalerSyncObj, &fence));
                auto span = dw::core::make_span<NvSciSyncFence>(&fence, 1);
                m_outputPort->setSignalFences(outputFreePacket, span);
            }

            auto work = [this, outputFreePacket]() {
                if (c2pEnabled())
                {
                    for (uint32_t i = 0U; i < m_producerWaitFences.size(); i++)
                    {
                        CHECK_NVSCI_ERROR(NvSciSyncFenceWait(&m_producerWaitFences[i], m_cpuWaitContext, -1));
                    }
                }

                usleep(10000); // sleep to emulate busy writing

                // get send time for perf measurement
                dwTime_t time = getClockRealtime();
                TypeCallbacks<PacketT>::setTimestamp(*outputFreePacket, time);

                if (p2cEnabled())
                {
                    // signal the sync object that frame is done
                    CHECK_NVSCI_ERROR(NvSciSyncObjSignal(m_producerSignalerSyncObj));
                }
            };

            if (p2cEnabled())
            {
                if (future.valid())
                {
                    future.wait();
                }
                future = std::async(std::launch::async, work);
            }
            else
            {
                work(); // work must be completed before sending
            }

            // send data from memory slot
            CHECK_DW_ERROR(m_outputPort->send(outputFreePacket));
            currFrame++;
        }

        dwTime_t threadEndTime = getClockMonotonic();
        m_prodElapsed          = threadEndTime - threadStartTime;
    }

    void startProducerThread()
    {
        if (m_hasProducer)
        {
            m_producerThread = std::thread(&DWChannelSample::runProducer, this);
        }
    }

    void startConsumerThreads()
    {
        for (size_t consId = 0; consId < m_numConsumers; consId++)
        {
            m_consumerThreads.push_back(std::thread(&DWChannelSample::runConsumers, this, consId));
        }
    }

    void stopProducerThread()
    {
        if (m_hasProducer && m_producerThread.joinable())
        {
            m_producerThread.join();
        }
    }

    void stopConsumerThreads()
    {
        for (size_t consId = 0; consId < m_numConsumers; consId++)
        {
            if (m_consumerThreads[consId].joinable())
            {
                m_consumerThreads[consId].join();
            }
        }
    }

    void printAvgLatency()
    {
        // Cannot cacluate Latency and BW under reuse channel in this sample.
        if (m_isReuse)
        {
            return;
        }

        size_t imageSize = TypeCallbacks<PacketT>::getSize();

        if (m_hasProducer)
        {
            std::cout << "Producer Send BandWidth: " << (((imageSize * m_maxFrameNumber) / (1024 * 1024)) / (m_prodElapsed / 1000000000.0f)) << "MB/s" << std::endl;
        }

        for (size_t consId = 0; consId < m_numConsumers; ++consId)
        {
            dwTime_t accumulateLatency = 0;
            if (m_sendLatencyArr[consId].size() != m_maxFrameNumber)
            {
                std::cout << m_sendLatencyArr[consId].size() << std::endl;
                std::cout << "Consumer Idx[" << consId << "] Packets Number doesn't match with number of frames, use mailbox mode ? " << std::endl;
            }

            if (m_sendLatencyArr[consId].size() <= SKIP_FRAME_NUMBER)
            {
                continue;
            }

            accumulateLatency += std::accumulate(m_sendLatencyArr[consId].begin() + SKIP_FRAME_NUMBER, m_sendLatencyArr[consId].end(), 0);
            std::cout << "Consumer Idx[" << consId << "] Latency: " << (accumulateLatency / (m_maxFrameNumber - SKIP_FRAME_NUMBER)) << "us" << std::endl;
            std::cout << "Consumer Idx[" << consId << "] Recv BandWidth: " << (((imageSize * m_sendLatencyArr[consId].size()) / (1024 * 1024)) / (m_consElapsed[consId] / 1000000000.0f)) << "MB/s" << std::endl;
        }

        return;
    }

public:
    DWChannelSample(const ProgramArguments& args)
        : m_args(args)
    {
    }

    bool initialize()
    {
        initContext();

        parseAndCheckArguments();
        initChannels();

        connectChannels();

        return true;
    }

    void process()
    {
        startConsumerThreads();
        startProducerThread();

        stopConsumerThreads();
        stopProducerThread();

        return;
    }

    void release()
    {
        printAvgLatency();

        m_channelFactory->stopServices();

        if (m_cpuWaitContext != nullptr)
        {
            NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
        }

        for (size_t consId = 0; consId < m_numConsumers; consId++)
        {
            m_inputPorts[consId].reset();
            m_inputChannels[consId].reset();
        }

        if (m_hasProducer)
        {
            m_outputChannel.reset();
            m_outputPort.reset();
        }

        m_channelConnector.reset();
        m_channelFactory.reset();

        dwRelease(m_context);
    }
};

int main(int argc, const char** argv)
{
    // -------------------
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("prod", "1", "Have a producer in this process (0/1)."),
                              ProgramArguments::Option_t("downstreams", "1", "Number of the producer's downstreams [1,4]."),
                              ProgramArguments::Option_t("cons", "1", "Number of consumers in this process [0,4]"),
                              ProgramArguments::Option_t("ip", "127.0.0.1", "Source IP Address"),
                              ProgramArguments::Option_t("prod-stream-names", "", "colon-separated list of producer nvsciipc endpoints, for NVSCI mode only"),
                              ProgramArguments::Option_t("prod-reaches", "", "colon-separated list of producer reaches (process|chip), for NVSCI mode only"),
                              ProgramArguments::Option_t("cons-stream-names", "", "colon-separated list of consumer nvsciipc endpoints, for NVSCI mode only"),
                              ProgramArguments::Option_t("cons-reaches", "", "colon-separated list of consumer reaches (process|chip), for NVSCI mode only"),
                              ProgramArguments::Option_t("sync-mode", "none", "Type of synchronization (none|p2c|c2p|both), for NVSCI mode only"),
                              ProgramArguments::Option_t("dataType", "dwImage", "the type of data to be transfered (int|dwImage|custom)"),
                              ProgramArguments::Option_t("port", "40002", "PortId"),
                              ProgramArguments::Option_t("mode", "4", "ChannelMode (mailbox/reuse/[N])"),
                              ProgramArguments::Option_t("type", "SOCKET", "ChannelType (SOCKET/SHMEM_LOCAL/NVSCI)"),
                              ProgramArguments::Option_t("frames", "128", "number of frames to run the sample"),
                          },
                          "DW Channel Sample.");

    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    auto runApp = [](auto app) {
        app.initialize();
        app.process();
        app.release();
    };

    auto dataType = args.get("dataType");
    if (dataType == "dwImage")
    {
        runApp(DWChannelSample<dwImageHandle_t>(args));
    }
    else if (dataType == "int")
    {
        runApp(DWChannelSample<IntWithTimestamp>(args));
    }
    else if (dataType == "custom")
    {
        runApp(DWChannelSample<CustomRawBuffer>(args));
    }
    else
    {
        throw std::runtime_error("dataType arg is not one of (dwImage|int|custom)");
    }

    return 0;
}
