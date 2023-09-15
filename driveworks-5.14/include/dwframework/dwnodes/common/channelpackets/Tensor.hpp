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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSOR_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSOR_HPP_

#include <dw/core/context/Context.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/interop/streamer/TensorStreamer.h>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwframework/dwnodes/common/channelpackets/TensorBase.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include <dwshared/dwfoundation/dw/core/matrix/BaseMatrix.hpp>
#include <nvscibuf.h>

#include "TensorUtils.hpp"

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwDNNTensorHandle_t, dwDNNTensorProperties, DW_TENSOR_HANDLE);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwDNNTensorHandle_t*);

///////////////////////////////////////////////////////////////////////////////////////

namespace dw
{
namespace framework
{
class TensorHandlePacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Tensor handle
     *  @param[in] ctx driveworks context */
    TensorHandlePacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    virtual ~TensorHandlePacket();
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of Tensor handle */
    inline GenericData getGenericData() override
    {
        return GenericData(&m_dispatchTensor);
    }

protected:
    dwDNNTensorHandle_t m_dispatchTensor = DW_NULL_HANDLE;

private:
    dwDNNTensorHandle_t m_tensorHandle = DW_NULL_HANDLE;
};

class ChannelPacketTensorHandle : public TensorHandlePacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Tensor handle
     *  @param[in] ctx driveworks context */
    ChannelPacketTensorHandle(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    virtual ~ChannelPacketTensorHandle() = default;

    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame after transmission
     *  @param[in] unused */
    void deserialize(size_t) final;

private:
    dwContextHandle_t m_ctx = DW_NULL_HANDLE;

    std::unique_ptr<TensorSerializer> m_tensorSerializer = nullptr;
};

using TensorHandleNvSciPacket = TensorNvSciPacketBase<dwDNNTensorHandle_t>;

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSOR_HPP_
