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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORUTILS_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORUTILS_HPP_

#include <dw/core/context/Context.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/interop/streamer/TensorStreamer.h>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dwcgf/Exception.hpp>

namespace dw
{
namespace framework
{

/*
* The class that aids the serialization of a tensor into a linear buffer on host.
* The tensor could be stored either on host or on device. 
*
* Definition of the Tensor is available in src/dw/dnn/tensor/Tensor.h and src/dw/dnn/tensor/Tensor.hpp.
* Representation of the buffer is described as the "buffer" in src/dwcgf/channel/ChannelPacketImpl.hpp. 
*/
class TensorSerializer
{
public:
    /*! CTOR
     *  @param[in] src handle of the tensor object that this TensorSerializer serves to serialize 
     *  @param[in] ctx DW context
     */
    TensorSerializer(dwDNNTensorHandle_t src, dwContextHandle_t ctx);

    ~TensorSerializer()
    {
        if (m_streamerToCPU)
        {
            dwDNNTensorStreamer_release(m_streamerToCPU);
        }
    };

    // Getter of the tensor raw data size, EXCLUDING metadata.
    size_t getTensorRawDataSize() const
    {
        return m_totalElementSizeBytes;
    };

    // Getter of the tensor total data size, INCLUDING metadata.
    size_t getTensorTotalSize() const
    {
        return getTensorRawDataSize() + sizeof(dwDNNTensor);
    };

    /*! Tensor serialization function to serialize a Tensor into a linear buffer on host.
     *  @param[out] dst destination buffer 
     *  @param[in] srcHandle handle of the tensor
     *  @param[in] bufferSize maximum size of buffer copied
     *  @return bytes serialized */
    size_t serialize(uint8_t* dst, dwDNNTensorHandle_t srcHandle, size_t bufferSize)
    {
        if (nullptr == srcHandle || nullptr == dst)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "srcHandle or dst is a null pointer");
        }

        if (srcHandle != m_srcHandle)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorSerializer::serialize(): The input source handle is not the handle of the tensor object that this serializer serves.");
        }

        const size_t BUFFER_HEAD_POS = 0U;
        size_t offset                = serializeMetadata(dst, srcHandle, bufferSize, BUFFER_HEAD_POS);

        dwDNNTensor cpuTensor{};
        dwDNNTensorHandle_t recvTensor{nullptr};

        if (m_srcProps.tensorType != DW_DNN_TENSOR_TYPE_CPU)
        {
            dwTime_t time{};
            dwDNNTensor_getTimestamp(&time, srcHandle);
            static constexpr uint32_t TIMEOUT{1000};
            FRWK_CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(srcHandle, m_streamerToCPU));
            FRWK_CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&recvTensor, TIMEOUT, m_streamerToCPU));

            dwDNNTensor_setTimestamp(time, recvTensor);

            FRWK_CHECK_DW_ERROR(dwDNNTensor_getTensor(&cpuTensor, recvTensor));
        }
        else
        {
            FRWK_CHECK_DW_ERROR(dwDNNTensor_getTensor(&cpuTensor, srcHandle));
        }

        offset = serializeRawData(dst, &cpuTensor, bufferSize, offset);

        if (m_srcProps.tensorType != DW_DNN_TENSOR_TYPE_CPU)
        {
            dwDNNTensorHandle_t* returnedTensor{nullptr};
            FRWK_CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&recvTensor, m_streamerToCPU));
            FRWK_CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(returnedTensor, 0, m_streamerToCPU));
        }

        return offset;
    };

private:
    /*! Helper function to serialize the raw tensor data.
     *  @param[out] dst destination buffer 
     *  @param[in] src handle of the tensor
     *  @param[in] bufferSize maximum size of buffer copied
     *  @param[in] offset bytes in the buffer that are already used
     *  @return bytes serialized */
    size_t serializeRawData(uint8_t* dst, dwDNNTensor* src, size_t bufferSize, size_t offset)
    {
        if (offset + m_totalElementSizeBytes > bufferSize)
        {
            throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "TensorSerializer: serializeRawData: elementSizeBytes: ", m_totalElementSizeBytes, " buffer offset: ", offset, " bufferSize: ", bufferSize, " tensor exceeds buffer size.");
        }
        memcpy(dst + offset, src->ptr, m_totalElementSizeBytes);

        return offset + m_totalElementSizeBytes;
    }

    /*! Helper function to serialize the tensor metadata.
     *  @param[out] dst destination buffer 
     *  @param[in] srcHandle handle of the tensor
     *  @param[in] bufferSize maximum size of buffer copied
     *  @param[in] offset bytes in the buffer that are already used
     *  @return bytes serialized */
    size_t serializeMetadata(uint8_t* dst, dwDNNTensorHandle_t srcHandle, size_t bufferSize, size_t offset)
    {
        constexpr size_t METADATA_BUFFER_OFFSET{sizeof(dwDNNTensor)};

        dwDNNTensor metadata{};
        FRWK_CHECK_DW_ERROR(dwDNNTensor_getTensor(&metadata, srcHandle));

        if (offset + METADATA_BUFFER_OFFSET > bufferSize)
        {
            throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "TensorSerializer: serializeMetadata: METADATA_BUFFER_OFFSET: ", METADATA_BUFFER_OFFSET, " exceeds bufferSize: ", bufferSize);
        }

        memcpy(dst, &metadata, METADATA_BUFFER_OFFSET);

        return offset + METADATA_BUFFER_OFFSET;
    }

    dwContextHandle_t m_ctx                     = DW_NULL_HANDLE;
    dwDNNTensorHandle_t m_srcHandle             = DW_NULL_HANDLE;
    dwDNNTensorStreamerHandle_t m_streamerToCPU = DW_NULL_HANDLE;
    dwDNNTensorProperties m_srcProps{};
    size_t m_totalElementSizeBytes{0U};
};

/*
* The class that aids the deserialization of a tensor from a linear buffer on host.
* The tensor could be stored either on host or on device. 
*
* Definition of the Tensor is available in src/dw/dnn/tensor/Tensor.h and src/dw/dnn/tensor/Tensor.hpp.
* Representation of the buffer is described as the "buffer" in src/dwcgf/channel/ChannelPacketImpl.hpp. 
*/
class TensorDeserializer
{
public:
    TensorDeserializer() = default;

    ~TensorDeserializer() = default;

    /*! Deserialization function to deserialize the data in a linear buffer on host to the destination tensor.
     *  @param[out] dstTensor destination tensor 
     *  @param[in] srcBuffer linear buffer on host
     *  @param[in] elementSizeBytes total size of the serialized raw tensor data 
     *  @param[in] bufferSize maximum size of buffer copied
     *  @return bytes deserialized */
    static size_t deserialize(dwDNNTensorHandle_t dstTensor, uint8_t* srcBuffer, size_t elementSizeBytes, size_t bufferSize);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORUTILS_HPP_