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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/pyramid/Pyramid.h>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwcgf/channel/Buffer.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwPyramidImage, dwPyramidImageProperties, DW_PYRAMID_IMAGE);

using dw::core::util::ONE_U;

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
class PyramidImagePacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of pyramid image
     *  @param[in] ctx driveworks context */
    PyramidImagePacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~PyramidImagePacket();
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of Image handle */
    inline GenericData getGenericData() override
    {
        return GenericData(&m_dispatchPyramid);
    }

protected:
    dwPyramidImage m_pyramidImage{};
    dwPyramidImage m_dispatchPyramid{};
    dwPyramidImageProperties m_props{};

private:
    dwContextHandle_t m_ctx{};
};

// coverity[autosar_cpp14_a10_1_1_violation] : TODO(trushton) AVRR-4261 fix a10_1_1 violations in dwframework
class ChannelPacketPyramidImage : public PyramidImagePacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Image handle
     *  @param[in] ctx driveworks context */
    ChannelPacketPyramidImage(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~ChannelPacketPyramidImage() override;
    /*! Serializes the frame before transmission */
    void serializeImpl() override;
    /*! Deserializes the frame before transmission */
    void deserialize(size_t) override;

private:
    size_t m_planeCount[DW_PYRAMID_LEVEL_MAX_COUNT]{0};
    dwImageHandle_t m_imageHandleCPU[DW_PYRAMID_LEVEL_MAX_COUNT]{};
    dwImageHandle_t m_imageCPUCopy[DW_PYRAMID_LEVEL_MAX_COUNT]{};

    cudaStream_t m_cudaStream{};

    dwContextHandle_t m_ctx = DW_NULL_HANDLE;
};

class PyramidImageNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    struct CPUData
    {
        dwImageMetaData metadata{};
        dwTime_t timestamp{};
    };

public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of pyramid image
     *  @param[in] ctx driveworks context */
    PyramidImageNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~PyramidImageNvSciPacket();
    /*! Fill NvSciBuffAttrList at given index with Image properties used in constructing packet
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
    /*! Getter for number of Nvsci buffers
     *  @return number of nvsci buffers */
    inline uint32_t getNumBuffers() const final
    {
        if (m_props.levelCount == std::numeric_limits<decltype(m_props.levelCount)>::max())
        {
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "`m_props.levelCount + 1` will wrap.");
        }
        return dw::core::safeAdd(m_props.levelCount, ONE_U).value();
    }

    /*! Getter for generic data in channel packet
     *  @return GenericData representation of pyramid image */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_dispatch);
    }

private:
    dwContextHandle_t m_ctx{};
    dwPyramidImageProperties m_props{};
    std::unique_ptr<Buffer> m_headerDataBuffer{};

    dwPyramidImage m_orig{};
    dwPyramidImage m_dispatch{};

    uint8_t* m_origTopLineData[DW_PYRAMID_LEVEL_MAX_COUNT]{};
    uint8_t* m_origBottomLineData[DW_PYRAMID_LEVEL_MAX_COUNT]{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_
