/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
///////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_BUFFER_HPP_
#define DW_FRAMEWORK_BUFFER_HPP_

#include <cstddef>
#include <dwcgf/Types.hpp>
#include <nvscibuf.h>
#include <dw/core/language/Optional.hpp>
#include <dw/cuda/misc/DevicePtr.hpp>

namespace dw
{
namespace framework
{

enum class BufferBackendType : uint32_t
{
    CPU  = 0,
    CUDA = 1,
    // Add others as appropriate
};

using BufferFlags = uint32_t;

inline bool BufferFlagsBackendEnabled(BufferFlags flags, BufferBackendType type)
{
    return (static_cast<uint32_t>(flags) & (1U << static_cast<uint32_t>(type))) != 0U;
}

inline void BufferFlagsEnableBackend(BufferFlags& flags, BufferBackendType type)
{
    flags |= (1U << static_cast<uint32_t>(type));
}

struct BufferProperties
{
    BufferFlags enabledBackends;
    size_t byteSize;
};

class BufferBase
{
public:
    static constexpr char LOG_TAG[] = "Buffer";

    BufferBase(BufferProperties properties)
        : m_properties(std::move(properties))
    {
    }

    const BufferProperties& getProperties() const
    {
        return m_properties;
    }

    virtual void bindNvSciBufObj(NvSciBufObj bufObj)
    {
        if (m_bufObj)
        {
            throw Exception(DW_CALL_NOT_ALLOWED, "BufferBase: Cannot bind NvSciBufObj twice");
        }
        m_bufObj = bufObj;
    }

    void fillNvSciBufAttrs(NvSciBufAttrList attrList) const
    {
        const NvSciBufType bufferType               = NvSciBufType_RawBuffer;
        const NvSciBufAttrValAccessPerm permissions = NvSciBufAccessPerm_ReadWrite;
        const uint64_t rawAlign                     = 4;

        Array<NvSciBufAttrKeyValuePair, 3> rawBufferAttributes =
            {{{NvSciBufGeneralAttrKey_Types, &bufferType, sizeof(bufferType)},
              {NvSciBufGeneralAttrKey_RequiredPerm, &permissions, sizeof(permissions)},
              {NvSciBufRawBufferAttrKey_Align, &rawAlign, sizeof(rawAlign)}}};

        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                        rawBufferAttributes.data(),
                                                        rawBufferAttributes.size()));

        if (m_properties.byteSize > 0U)
        {
            const uint64_t rawSize = m_properties.byteSize;
            Array<NvSciBufAttrKeyValuePair, 1> rawBufferSizeAttributes =
                {{{NvSciBufRawBufferAttrKey_Size, &rawSize, sizeof(rawSize)}}};
            FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                            rawBufferSizeAttributes.data(),
                                                            rawBufferSizeAttributes.size()));
        }
    }

    NvSciBufObj getNvSci()
    {
        return m_bufObj;
    }

protected:
    BufferProperties m_properties{};
    NvSciBufObj m_bufObj{};
};

class BufferCPU : public BufferBase
{
public:
    explicit BufferCPU(BufferProperties properties)
        : BufferBase(properties)
    {
    }

    void fillSpecificAttributes(NvSciBufAttrList attrList) const
    {
        const bool cpuAccessFlag = true;

        Array<NvSciBufAttrKeyValuePair, 1> rawBufferAttributes =
            {{{NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag, sizeof(cpuAccessFlag)}}};

        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                        rawBufferAttributes.data(),
                                                        rawBufferAttributes.size()));
    }

    void bindNvSciBufObj(NvSciBufObj bufObj) override
    {
        BufferBase::bindNvSciBufObj(bufObj);
        FRWK_CHECK_NVSCI_ERROR(NvSciBufObjGetCpuPtr(m_bufObj, &m_ptr));
    }

    void* getCpuPtr(size_t offset)
    {
        return &(static_cast<uint8_t*>(m_ptr)[offset]);
    }

private:
    void* m_ptr;
};

// BufferCUDA class encapsulates API mapping operations on single nvscibuf for cuda.
// @note - If a buffer needs to be mapped for multiple cuda devices, multiple instances of this class
// should be used and the appropriate device must be set as current before calling into each instance.
class BufferCUDA : public BufferBase
{
public:
    explicit BufferCUDA(BufferProperties properties)
        : BufferBase(properties)
    {
    }

    ~BufferCUDA()
    {
        if (m_cudaHandle)
        {
            FRWK_CHECK_CUDA_ERROR_NOTHROW(cudaFree(m_ptr));
            FRWK_CHECK_CUDA_ERROR_NOTHROW(cudaDestroyExternalMemory(m_cudaHandle));
        }
    }

    void fillSpecificAttributes(NvSciBufAttrList attrList) const
    {
        NvSciRmGpuId gpuIds[] = {{0}};
        int32_t cudaDevice;
        FRWK_CHECK_CUDA_ERROR(cudaGetDevice(&cudaDevice));
#ifndef DW_IS_SAFETY // TODO(dwsafety): DRIV-8345
        cudaDeviceProp deviceProps{};
        FRWK_CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProps, cudaDevice));
        static_assert(sizeof(deviceProps.uuid) == sizeof(gpuIds[0]), "BufferCUDA: cuda uuid size does not match size of NvSciRmGpuId");
        memcpy(static_cast<void*>(&gpuIds[0]), static_cast<void*>(&deviceProps.uuid), sizeof(gpuIds[0]));

        Array<NvSciBufAttrKeyValuePair, 1> rawBufferAttributes =
            {{{NvSciBufGeneralAttrKey_GpuId, &gpuIds, sizeof(gpuIds)}}};

        FRWK_CHECK_NVSCI_ERROR(NvSciBufAttrListSetAttrs(attrList,
                                                        rawBufferAttributes.data(),
                                                        rawBufferAttributes.size()));
#endif
    }

    void bindNvSciBufObj(NvSciBufObj bufObj) override
    {
        BufferBase::bindNvSciBufObj(bufObj);
        cudaExternalMemoryHandleDesc cudaMemHandleDesc = {};
        cudaMemHandleDesc.type                         = cudaExternalMemoryHandleTypeNvSciBuf;
        cudaMemHandleDesc.handle.nvSciBufObject        = bufObj;
        cudaMemHandleDesc.size                         = m_properties.byteSize;
        cudaMemHandleDesc.flags                        = 0;
        FRWK_CHECK_CUDA_ERROR(cudaImportExternalMemory(&m_cudaHandle, &cudaMemHandleDesc));

        cudaExternalMemoryBufferDesc cudaBufferDesc = {};
        memset(&cudaBufferDesc, 0, sizeof(cudaBufferDesc));
        cudaBufferDesc.size   = m_properties.byteSize;
        cudaBufferDesc.offset = 0;
        FRWK_CHECK_CUDA_ERROR(cudaExternalMemoryGetMappedBuffer(&m_ptr, m_cudaHandle, &cudaBufferDesc));
    }

    core::DevicePtr<void> getCudaPtr(size_t offset)
    {
        void* ptr = &(static_cast<uint8_t*>(m_ptr)[offset]);
        return core::MakeDevicePtr(ptr);
    }

private:
    cudaExternalMemory_t m_cudaHandle{};
    void* m_ptr{};
};

// Buffer class encapsulates API mapping operations on single nvscibuf.
// @note - If a buffer needs to be mapped for multiple cuda devices, multiple instances of this class
// should be used and the appropriate device must be set as current before calling into each instance.
class Buffer : public BufferBase
{

public:
    explicit Buffer(BufferProperties properties)
        : BufferBase(properties)
    {
        auto enabledBackends = m_properties.enabledBackends;
        if (BufferFlagsBackendEnabled(enabledBackends, BufferBackendType::CPU))
        {
            m_bufferCpu.emplace(properties);
        }
        if (BufferFlagsBackendEnabled(enabledBackends, BufferBackendType::CUDA))
        {
            m_bufferCuda.emplace(properties);
        }
    }

    void bindNvSciBufObj(NvSciBufObj bufObj) override
    {
        BufferBase::bindNvSciBufObj(bufObj);
        if (m_bufferCpu)
        {
            m_bufferCpu.value().bindNvSciBufObj(bufObj);
        }
        if (m_bufferCuda)
        {
            m_bufferCuda.value().bindNvSciBufObj(bufObj);
        }
    }

    void fillNvSciBufAttrs(NvSciBufAttrList attrList) const
    {
        BufferBase::fillNvSciBufAttrs(attrList);
        if (m_bufferCpu)
        {
            m_bufferCpu.value().fillSpecificAttributes(attrList);
        }
        if (m_bufferCuda)
        {
            m_bufferCuda.value().fillSpecificAttributes(attrList);
        }
    }

    void* getCpuPtr(size_t offset)
    {
        return m_bufferCpu.value().getCpuPtr(offset);
    }

    core::DevicePtr<void> getCudaPtr(size_t offset)
    {
        return m_bufferCuda.value().getCudaPtr(offset);
    }

private:
    dw::core::Optional<BufferCPU> m_bufferCpu{};
    dw::core::Optional<BufferCUDA> m_bufferCuda{};
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_BUFFER_HPP_
