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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_EGOMOTIONSTATE_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_EGOMOTIONSTATE_HPP_

#include <dw/egomotion/2.0/Egomotion2.h>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwEgomotionStateHandle_t, dwEgomotionStateParams, DW_EGOMOTION_STATE_HANDLE);

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
class EgomotionStateHandlePacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData representation of egomotion state handle
     *  @param[in] ctx driveworks context */
    EgomotionStateHandlePacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~EgomotionStateHandlePacket();
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of egomotion state handle */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_dispatchEgomotionState);
    }

protected:
    dwEgomotionStateHandle_t m_egomotionState         = DW_NULL_HANDLE;
    dwEgomotionStateHandle_t m_dispatchEgomotionState = DW_NULL_HANDLE;
};

// coverity[autosar_cpp14_a10_1_1_violation] : TODO(trushton) AVRR-4261 fix a10_1_1 violations in dwframework
class ChannelPacketEgomotionState : public EgomotionStateHandlePacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData representation of egomotion state handle
     *  @param[in] ctx driveworks context */
    ChannelPacketEgomotionState(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Constructor
     *  @param[in] params egomotion state handle to send with channel packet
     *  @param[in] ctx driveworks context */
    ChannelPacketEgomotionState(dwEgomotionStateParams& params, dwContextHandle_t ctx);
    /*! Getter for number of bytes of buffer in channel packet */
    size_t getNumBytes();
    /*! Serializes the packet before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame before transmission */
    void deserialize(size_t) final;
    /*! Getter for egomotion state handle in channel packet
     *  @return pointer to egomotion state handle in channel packet */
    inline dwEgomotionStateHandle_t* getFrame()
    {
        return &m_dispatchEgomotionState;
    }

private:
    size_t m_numBytes = 0;
};

class EgomotionStateHandleNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 1U;

public:
    /*! Constructor
     *  @param[in] specimen GenericData representation of egomotion state handle
     *  @param[in] ctx driveworks context */
    EgomotionStateHandleNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~EgomotionStateHandleNvSciPacket() override;
    /*! Getter for number of Nvsci buffers
     *  @return number of nvsci buffers */
    inline uint32_t getNumBuffers() const final
    {
        return NUM_BUFFERS;
    }
    /*! Fill NvSciBuffAttrList at given index with history size used in constructing packet
     *  @param[in] bufferIndex index between 0 and NUM_BUFFERS-1
     *  @param[in,out] attrList list of nvsci buffer attributes
     *  @throws Exception with status DW_OUT_OF_BOUNDS if buffer index is >= NUM_BUFFERS */
    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final;
    /*! Initialize channel packets using the nvsci buffers passed
     *  @param[in] bufs NvSciBufObj to bind to */
    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final;
    /*! Fill out additional metadata prior to sending */
    void pack() final;
    /*! Fill out metadata upon receipt */
    void unpack() final;
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of egomotion state handle */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_dispatch);
    }

private:
    dwEgomotionStateHandle_t m_dispatch{};
    dwEgomotionStateHandle_t m_orig{};

    dwEgomotionStateParams m_reference{};
    dwContextHandle_t m_ctx{};
};

class EgomotionStateHandleRemoteShmemPacket : public IChannelPacket, public IChannelPacket::RemoteShmemCallbacks
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData representation of egomotion state handle
     *  @param[in] ctx driveworks context */
    EgomotionStateHandleRemoteShmemPacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~EgomotionStateHandleRemoteShmemPacket() override;

    /// Getter for number of bytes in the channel packet
    /// @throws ExceptionWithStatus with DW_INTERNAL_ERROR if the buffer size would overflow size_t
    /// @return number of bytes in the channel packet
    size_t getBufferSize() final;
    size_t getPoolIndex() final;
    void initializeFromMemory(void* ptr, size_t index) final;
    void pack() final;
    void unpack() final;
    GenericData getGenericData() final;

private:
    size_t m_poolIndex{};

    dwEgomotionStateHandle_t m_dispatch{};
    dwEgomotionStateHandle_t m_orig{};

    dwEgomotionStateParams m_reference{};
    dwContextHandle_t m_ctx{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_EGOMOTIONSTATE_HPP_
