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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_FEATURELIST_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_FEATURELIST_HPP_

#include <dw/core/context/Context.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwFeatureArray, DW_FEATURE_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwFeatureHistoryArray, DW_FEATURE_HISTORY_ARRAY);

namespace dw
{
namespace framework
{
///////////////////////////////////////////////////////////////////////////////////////
class FeatureArrayPacket : public IChannelPacket
{
public:
    FeatureArrayPacket(const GenericData& specimen, dwContextHandle_t ctx);
    ~FeatureArrayPacket() override;
    GenericData getGenericData() override;

protected:
    dwFeatureArray m_featureArray{};
    dwFeatureArray m_featureArrayOrig{};
    cudaStream_t m_stream{};
};

// coverity[autosar_cpp14_a10_1_1_violation] : TODO(trushton) AVRR-4261 fix a10_1_1 violations in dwframework
class ChannelPacketFeatureArray : public FeatureArrayPacket, public ChannelSocketPacketBase
{
public:
    ChannelPacketFeatureArray(const GenericData& specimen, dwContextHandle_t ctx);
    // Serializes the frame before transmission
    void serializeImpl() override;
    // Deserializes the frame before transmission
    void deserialize(size_t) override;

private:
    static constexpr uint32_t NUM_PROPS{9u};
    size_t m_propIndex[NUM_PROPS] = {};
};

///////////////////////////////////////////////////////////////////////////////////////
class FeatureHistoryArrayPacket : public IChannelPacket
{
public:
    FeatureHistoryArrayPacket(const GenericData& specimen, dwContextHandle_t ctx);
    ~FeatureHistoryArrayPacket() override;
    GenericData getGenericData() final;

protected:
    dwFeatureHistoryArray m_featureHistoryArray{};
    dwFeatureHistoryArray m_featureHistoryArrayOrig{};
    cudaStream_t m_stream{};
};

// coverity[autosar_cpp14_a10_1_1_violation] : Class shall not be derived from more than one base class which is not an interface class.
class ChannelPacketFeatureHistoryArray : public FeatureHistoryArrayPacket, public ChannelSocketPacketBase
{
public:
    ChannelPacketFeatureHistoryArray(const GenericData& specimen, dwContextHandle_t ctx);
    // Serializes the frame before transmission
    void serializeImpl() override;
    // Deserializes the frame before transmission
    void deserialize(size_t) override;
};

class FeatureHistoryArrayNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks

{
    static constexpr uint32_t NUM_BUFFERS{2U};

public:
    FeatureHistoryArrayNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx = DW_NULL_HANDLE);
    uint32_t getNumBuffers() const final;
    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final;
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    void pack() final;
    void unpack() final;
    GenericData getGenericData() final;

    // get the pointer into the allocation buffer at given offset
    void* getAllocationPtr(size_t offset);

private:
    // set the pointers that point into the allocation buffer
    void setPointers();

    dw::core::VectorFixed<std::unique_ptr<Buffer>, NUM_BUFFERS> m_buffers{};
    dwFeatureHistoryArray* m_header{};
    dwFeatureHistoryArray m_reference{};
    dwFeatureHistoryArray m_dispatch{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_FEATURELIST_HPP_
