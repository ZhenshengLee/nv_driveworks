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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORBASE_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORBASE_HPP_

#include <dw/dnn/tensor/Tensor.h>
#include <dwcgf/Exception.hpp>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwcgf/channel/NvSciHelper.hpp>
#include <dwshared/dwfoundation/dw/core/utility/Constants.hpp>

using dw::core::util::ONE_U;
using dw::core::util::TWO_U;
using dw::core::util::ZERO_U;

namespace dw
{
namespace framework
{

// forward declaration
BufferFlags GetBufferFlagsFromMemoryType(dwMemoryType memType);
void* GetBufferPointerWithMemoryType(Buffer& buffer, dwMemoryType memoryType, size_t offset);

///////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class TensorPacketAccessor
{
public:
    /**
     * Set the tensor handle of a specified tensor struct.
     *
     * @param[out] data The specified tensor struct
     * @param[in] tensor The updated tensor handle
     */
    static void setTensorHandle(T& data, dwDNNTensorHandle_t tensor);

    /**
     * Get the tensor handle of a specified tensor struct.
     *
     * @param[in] data The specified tensor struct
     * @return The tensor handle
     */
    static dwDNNTensorHandle_t getTensorHandle(const T& data);
};

template <typename T>
class TensorNvSciPacketBase : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS = 2U;

public:
    static constexpr const char* LOG_TAG = "TensorNvSciPacketBase";

    /*! Constructor with specimen
     *  @param[in] specimen GenericData represention of dwDNNTensorProperties
     */
    TensorNvSciPacketBase(const GenericData& specimen, dwContextHandle_t /*ctx*/)
        : TensorNvSciPacketBase()
    {
        auto* props = specimen.getData<dwDNNTensorProperties>();
        if (!props)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorNvSciPacketBase: invalid dwDNNTensorProperties provided");
        }

        m_props = *props;
    }

    uint32_t getNumBuffers() const final
    {
        return NUM_BUFFERS;
    }

    void fillNvSciBufAttributes(uint32_t bufferIndex, NvSciBufAttrList& attrList) const final
    {
        if (bufferIndex >= NUM_BUFFERS)
        {
            throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "TensorNvSciPacketBase: Buffer index ", bufferIndex, " is greater than buffers available: ", NUM_BUFFERS);
        }

        if (bufferIndex == ZERO_U)
        {
            m_headerBuffer->fillNvSciBufAttrs(attrList);
        }
        else
        {
            fillNvSciBufAttrsForTensor(attrList);
        }
    }

    static NvSciBufAttrValDataType getNvSciDataType(dwTrivialDataType dataType)
    {
        NvSciBufAttrValDataType nvSciDataType = NvSciDataType_UpperBound;
        switch (dataType)
        {
        case DW_TYPE_BOOL:
        {
            nvSciDataType = NvSciDataType_Bool;
            break;
        }
        case DW_TYPE_INT8:
        {
            nvSciDataType = NvSciDataType_Int8;
            break;
        }
        case DW_TYPE_INT16:
        {
            nvSciDataType = NvSciDataType_Int16;
            break;
        }
        case DW_TYPE_INT32:
        {
            nvSciDataType = NvSciDataType_Int32;
            break;
        }
        case DW_TYPE_UINT8:
        {
            nvSciDataType = NvSciDataType_Uint8;
            break;
        }
        case DW_TYPE_UINT16:
        {
            nvSciDataType = NvSciDataType_Uint16;
            break;
        }
        case DW_TYPE_UINT32:
        {
            nvSciDataType = NvSciDataType_Uint32;
            break;
        }
        case DW_TYPE_FLOAT32:
        {
            nvSciDataType = NvSciDataType_Float32;
            break;
        }
        case DW_TYPE_FLOAT16:
        {
            nvSciDataType = NvSciDataType_Float16;
            break;
        }
        case DW_TYPE_CHAR8:
        {
            nvSciDataType = NvSciDataType_Int8;
            break;
        }
        case DW_TYPE_INT64:
        case DW_TYPE_UINT64:
        case DW_TYPE_FLOAT64:
            throw ExceptionWithStatus(DW_NOT_SUPPORTED, "TensorNvSciPacketBase: unsupported dwTrivialDataType ", dataType);
        case DW_TYPE_UNKNOWN:
        default:
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorNvSciPacketBase: unknown dwTrivialDataType ", dataType);
        }
        return nvSciDataType;
    }

    static uint32_t getHeightDimIdx(dwDNNTensorLayout const tensorLayout)
    {
        uint32_t heightIdx{ZERO_U};
        switch (tensorLayout)
        {
        case DW_DNN_TENSOR_LAYOUT_NCHW:
            heightIdx = ONE_U;
            break;
        case DW_DNN_TENSOR_LAYOUT_NCHWx:
            heightIdx = TWO_U;
            break;
        case DW_DNN_TENSOR_LAYOUT_NHWC:
            heightIdx = TWO_U;
            break;
        default:
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorNvSciPacketBase: invalid tensor layout ", tensorLayout);
        }
        return heightIdx;
    }

    void fillNvSciBufAttrsForTensor(NvSciBufAttrList& attrList) const
    {
        const NvSciBufType bufferType{NvSciBufType_Tensor};
        const NvSciBufAttrValAccessPerm permissions{NvSciBufAccessPerm_ReadWrite};
        const uint64_t baseAddrAlign{512U};
        const NvSciBufAttrValDataType dataType = getNvSciDataType(m_props.dataType);
        const uint32_t numDims                 = m_props.numDimensions;
        if (numDims > DW_DNN_TENSOR_MAX_DIMENSIONS)
        {
            throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "TensorNvSciPacketBase: dimension number ", numDims, " is larger than DW_DNN_TENSOR_MAX_DIMENSIONS: ", DW_DNN_TENSOR_MAX_DIMENSIONS);
        }

        uint64_t sizePerDim[DW_DNN_TENSOR_MAX_DIMENSIONS]{};
        uint32_t alignmentPerDim[DW_DNN_TENSOR_MAX_DIMENSIONS]{};
        for (uint32_t idx = 0; idx < numDims; ++idx)
        {
            // Note: the dimension order in NvSciBufTensor is the reverse of the dimension order in dwDNNTensorProperties
            uint32_t idxInNvSciBufTensor    = numDims - 1 - idx;
            sizePerDim[idxInNvSciBufTensor] = m_props.dimensionSize[idx];
            // No special alignment requirement for now
            alignmentPerDim[idxInNvSciBufTensor] = ONE_U;
        }

        const uint32_t bufferAttrNum{7U};
        dw::core::Array<NvSciBufAttrKeyValuePair, bufferAttrNum> tensorBufferAttributes{
            {{NvSciBufGeneralAttrKey_Types, &bufferType, sizeof(decltype(bufferType))},
             {NvSciBufGeneralAttrKey_RequiredPerm, &permissions, sizeof(decltype(permissions))},
             {NvSciBufTensorAttrKey_DataType, &dataType, sizeof(decltype(dataType))},
             {NvSciBufTensorAttrKey_NumDims, &numDims, sizeof(numDims)},
             {NvSciBufTensorAttrKey_SizePerDim, sizePerDim, sizeof(uint64_t) * numDims},
             {NvSciBufTensorAttrKey_AlignmentPerDim, alignmentPerDim, sizeof(uint32_t) * numDims},
             {NvSciBufTensorAttrKey_BaseAddrAlign, &baseAddrAlign, sizeof(baseAddrAlign)}}};

        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                        tensorBufferAttributes.data(),
                                                        tensorBufferAttributes.size()));

        fillNvSciBufAttrsForTensorType(attrList);
    }

    void fillNvSciBufAttrsForTensorType(NvSciBufAttrList& attrList) const
    {
        if (m_props.tensorType == DW_DNN_TENSOR_TYPE_CPU)
        {
            const bool cpuAccessFlag{true};

            Array<NvSciBufAttrKeyValuePair, 1> cpuBufferAttributes{
                {{NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag, sizeof(cpuAccessFlag)}}};

            FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                            cpuBufferAttributes.data(),
                                                            cpuBufferAttributes.size()));
        }
        else if (m_props.tensorType == DW_DNN_TENSOR_TYPE_CUDA)
        {
            int32_t cudaDevice{};
            FRWK_CHECK_CUDA_ERROR(cudaGetDevice(&cudaDevice));
            CUuuid uuid{};
            FRWK_CHECK_CUDA_ERROR(cuDeviceGetUuid(&uuid, cudaDevice));
            NvSciRmGpuId gpuIds[]{{0U}};
            static_assert(sizeof(uuid) == sizeof(gpuIds[0]), "TensorNvSciPacketBase: cuda uuid size does not match size of NvSciRmGpuId");
            memcpy(static_cast<void*>(&gpuIds[0]), static_cast<void*>(&uuid), sizeof(gpuIds[0]));

            Array<NvSciBufAttrKeyValuePair, 1> gpuBufferAttributes{
                {{NvSciBufGeneralAttrKey_GpuId, &gpuIds, sizeof(gpuIds)}}};

            FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                            gpuBufferAttributes.data(),
                                                            gpuBufferAttributes.size()));
        }
        else
        {
            throw ExceptionWithStatus(DW_NOT_SUPPORTED, "TensorNvSciPacketBase: unsupported dwDNNTensorType ", m_props.tensorType);
        }
    }

    void initializeFromNvSciBufObjs(dw::core::span<NvSciBufObj> bufs) final
    {
        if (bufs.size() != NUM_BUFFERS)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorNvSciPacketBase: invalid number of buffers provided.");
        }

        m_headerBuffer->bindNvSciBufObj(bufs.front());

        m_header       = static_cast<decltype(m_header)>(m_headerBuffer->getCpuPtr(ZERO_U));
        m_tensorHeader = static_cast<decltype(m_tensorHeader)>(m_headerBuffer->getCpuPtr(m_headerSize));

        // the reference buffer stores everything in its original state
        // the dispatch buffer is a copy of the reference buffer but it can be
        // corrupted by client, so it is separate
        TensorPacketAccessor<T>::setTensorHandle(m_reference, createAndBindNvSciBufForTensor(bufs[ONE_U]));
        m_dispatch = m_reference;
    }

    uint64_t getTensorTotalSize(NvSciBufObj buf) const
    {
        NvSciBufAttrList bufAttrList{};
        FRWK_CHECK_NVSCI_ERROR(NvSciBufObjGetAttrList(buf, &bufAttrList));

        NvSciBufAttrKeyValuePair queryAttrs[] = {
            {NvSciBufTensorAttrKey_Size, nullptr, 0}};
        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListGetAttrs(bufAttrList, queryAttrs, 1U));

        const uint64_t totalSize = *(static_cast<const uint64_t*>(queryAttrs[0].value));
        return totalSize;
    }

    size_t getPitch() const
    {
        size_t pitch = dwSizeOf(m_props.dataType);
        // calculate pitch
        if (m_props.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW || m_props.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC)
        {
            for (uint32_t idx = 0; idx < getHeightDimIdx(m_props.tensorLayout); ++idx)
            {
                auto result = dw::core::safeMul(pitch, m_props.dimensionSize[idx]);
                if (result.isSafe())
                {
                    pitch = result.value();
                }
                else
                {
                    throw ExceptionWithStatus(DW_FAILURE, "TensorNvSciPacketBase: getPitch: 'pitch*m_props.dimensionSize[idx]' will wraparound.");
                }
            }
        }
        else
        {
            throw ExceptionWithStatus(DW_NOT_SUPPORTED, "TensorNvSciPacketBase: unsupported tensorLayout ", m_props.tensorLayout);
        }

        return pitch;
    }

    dwDNNTensorHandle_t createAndBindNvSciBufForTensor(NvSciBufObj buf)
    {
        // extract data pointer from nvscibuf with class Buffer help
        const uint64_t totalSize    = getTensorTotalSize(buf);
        dwMemoryType memType        = (m_props.tensorType == DW_DNN_TENSOR_TYPE_CPU) ? DW_MEMORY_TYPE_CPU : DW_MEMORY_TYPE_CUDA;
        BufferProperties bufferProp = {GetBufferFlagsFromMemoryType(memType), totalSize};
        m_tensorBuffer              = std::make_unique<Buffer>(bufferProp);
        m_tensorBuffer->bindNvSciBufObj(buf);

        // create tensor with external memory from nvscibuf
        const size_t pitch = getPitch();
        uint8_t* extMemPtr = static_cast<uint8_t*>(GetBufferPointerWithMemoryType(*m_tensorBuffer, memType, ZERO_U));
        dwDNNTensorHandle_t tensorHandle{};
        FRWK_CHECK_DW_ERROR(dwDNNTensor_createWithExtMem(&tensorHandle, &m_props, extMemPtr, pitch));
        return tensorHandle;
    }

    void pack() final
    {
        // check that the dispatch buffer hasn't been corrupted from its original state
        // by verifying that the pointers haven't been moved.
        // If pointers were moved, then they likely no longer point to the shared allocation in the nvscibufobj
        // therefore, throw an exception.
        if (TensorPacketAccessor<T>::getTensorHandle(m_dispatch) != TensorPacketAccessor<T>::getTensorHandle(m_reference))
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "TensorNvSciPacketBase: reassigned pointers cannot be packed.");
        }
        // copy the dispatch buffer to the header buffer.
        *m_header = m_dispatch;

        dwDNNTensor tensorHeader{};
        FRWK_CHECK_DW_ERROR(dwDNNTensor_getTensor(&tensorHeader, TensorPacketAccessor<T>::getTensorHandle(m_dispatch)));
        *m_tensorHeader = tensorHeader;
    }

    void unpack() final
    {
        // unpack the header to the dispatch buffer,
        // set the pointers to the appropriate values (as the addresses in the header will be different
        // since the base address of the allocation buffer may be for a different engine or process)
        m_dispatch = *m_header;
        TensorPacketAccessor<T>::setTensorHandle(m_dispatch, TensorPacketAccessor<T>::getTensorHandle(m_reference));
        FRWK_CHECK_DW_ERROR(dwDNNTensor_setTimestamp(m_tensorHeader->timestamp_us, TensorPacketAccessor<T>::getTensorHandle(m_dispatch)));
    }

    GenericData getGenericData() final
    {
        return GenericData(&m_dispatch);
    }

protected:
    /*! Default Constructor
     *  For some packet type in which the specimen is not dwDNNTensorProperties (e.g. dwParkingEgmHeightMapTensor),
     *  the derived TensorNvSciPacket class should set the "m_props" in its own constructor.
     */
    TensorNvSciPacketBase()
    {
        m_headerSize       = sizeof(T);
        m_tensorHeaderSize = sizeof(dwDNNTensor);
        m_headerBufferSize = m_headerSize + m_tensorHeaderSize;

        // use two nvsci buffers for this packet
        // buffer 0 is the cpu header, i.e. the "m_headerBuffer"
        // buffer 1 is the tensor allocation, i.e. the "m_tensorBuffer"
        BufferProperties bufferProp = {GetBufferFlagsFromMemoryType(DW_MEMORY_TYPE_CPU), m_headerBufferSize};
        m_headerBuffer              = std::make_unique<Buffer>(bufferProp);
    }

    dwDNNTensorProperties m_props{};

private:
    size_t m_headerSize{0U};
    size_t m_tensorHeaderSize{0U};
    size_t m_headerBufferSize{0U};
    std::unique_ptr<Buffer> m_headerBuffer{};
    std::unique_ptr<Buffer> m_tensorBuffer{};

    T* m_header{};
    dwDNNTensor* m_tensorHeader{};
    T m_dispatch{};
    T m_reference{};
};

template <typename T>
constexpr uint32_t TensorNvSciPacketBase<T>::NUM_BUFFERS;

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_TENSORBASE_HPP_
