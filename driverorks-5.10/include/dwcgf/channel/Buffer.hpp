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

    BufferBase(BufferProperties properties);

    virtual ~BufferBase() = default;

    const BufferProperties& getProperties() const;

    virtual void bindNvSciBufObj(NvSciBufObj bufObj);

    void fillNvSciBufAttrs(NvSciBufAttrList attrList) const;

    NvSciBufObj getNvSci();

protected:
    BufferProperties m_properties{};
    NvSciBufObj m_bufObj{};
};

class BufferCPU final : public BufferBase
{
public:
    explicit BufferCPU(BufferProperties properties);

    void fillSpecificAttributes(NvSciBufAttrList attrList) const;

    void bindNvSciBufObj(NvSciBufObj bufObj) override;

    void* getCpuPtr(size_t offset);

private:
    void* m_ptr;
};

// BufferCUDA class encapsulates API mapping operations on single nvscibuf for cuda.
// @note - If a buffer needs to be mapped for multiple cuda devices, multiple instances of this class
// should be used and the appropriate device must be set as current before calling into each instance.
class BufferCUDA final : public BufferBase
{
public:
    explicit BufferCUDA(BufferProperties properties);

    ~BufferCUDA() final;

    void fillSpecificAttributes(NvSciBufAttrList attrList) const;

    void bindNvSciBufObj(NvSciBufObj bufObj) override;

    core::DevicePtr<void> getCudaPtr(size_t offset);

private:
    cudaExternalMemory_t m_cudaHandle{};
    void* m_ptr{};
};

// Buffer class encapsulates API mapping operations on single nvscibuf.
// @note - If a buffer needs to be mapped for multiple cuda devices, multiple instances of this class
// should be used and the appropriate device must be set as current before calling into each instance.
class Buffer final : public BufferBase
{

public:
    explicit Buffer(BufferProperties properties);

    void bindNvSciBufObj(NvSciBufObj bufObj) override;

    void fillNvSciBufAttrs(NvSciBufAttrList attrList) const;

    void* getCpuPtr(size_t offset);

    core::DevicePtr<void> getCudaPtr(size_t offset);

private:
    dw::core::Optional<BufferCPU> m_bufferCpu{};
    dw::core::Optional<BufferCUDA> m_bufferCuda{};
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_BUFFER_HPP_
