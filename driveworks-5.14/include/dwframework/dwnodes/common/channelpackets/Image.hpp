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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_IMAGE_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_IMAGE_HPP_

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dwcgf/channel/Buffer.hpp>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include <dwshared/dwfoundation/dw/core/matrix/BaseMatrix.hpp>
#include <nvscibuf.h>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwImageHandle_t, dwImageProperties, DW_IMAGE_HANDLE);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwImageHandle_t*);

///////////////////////////////////////////////////////////////////////////////////////

namespace dw
{
namespace framework
{
class ImageHandlePacket : public IChannelPacket
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Image handle
     *  @param[in] ctx driveworks context */
    ImageHandlePacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~ImageHandlePacket();
    /*! Getter for generic data in channel packet
     *  @return GenericData representation of Image handle */
    inline GenericData getGenericData() override
    {
        return GenericData(&m_dispatchImage);
    }

protected:
    dwImageHandle_t m_imageHandle   = DW_NULL_HANDLE;
    dwImageHandle_t m_dispatchImage = DW_NULL_HANDLE;
    dwImageProperties m_prop{};
};

class ChannelPacketImageHandle : public ImageHandlePacket, public ChannelSocketPacketBase
{
public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Image handle
     *  @param[in] ctx driveworks context */
    ChannelPacketImageHandle(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~ChannelPacketImageHandle() override;
    /*! Image serialization static functions to allow re-use in dwBlindnessDetectionOutput serialization.
     *  @param[in] cpuImage image to copy data from
     *  @param[out] buffer_start location to copy data to
     *  @param[in] bufferSize maximum size of buffer copied
     *  @param[in] planeCount number of planes copied
     *  @param[in] elementSize size in bytes of pixel type
     *  @param[in] planeChannelCount number of channels of each plane
     *  @param[in] planeSize 2D representation of each plane size
     *  @return bytes serialized */
    static size_t serializeImage(dwImageCPU* cpuImage, unsigned char* buffer_start, size_t bufferSize,
                                 size_t planeCount, size_t elementSize, const uint32_t planeChannelCount[],
                                 dwVector2ui planeSize[]);
    /*! Image deserialization static functions to allow re-use in dwBlindnessDetectionOutput serialization.
     *  @param[out] copyToImage image handle to copy data to
     *  @param[in] buffer_start
     *  @param[in] bufferSize maximum size of buffer copied
     *  @param[in] planeCount number of planes copied
     *  @param[in] elementSize size in bytes of pixel type
     *  @param[in] planeChannelCount number of channels of each plane
     *  @param[in] planeSize 2D representation of each plane size */
    static void deserializeImage(dwImageHandle_t copyToImage, unsigned char* buffer_start, size_t bufferSize,
                                 size_t planeCount, size_t elementSize, const uint32_t planeChannelCount[],
                                 dwVector2ui planeSize[]);
    /*! Serializes the frame before transmission */
    void serializeImpl() final;
    /*! Deserializes the frame after transmission
     *  @param[in] unused */
    void deserialize(size_t) final;

private:
    dwImageHandle_t m_imageHandleCPU = DW_NULL_HANDLE;

    size_t m_elementSize = 0;
    size_t m_planeCount  = 0;
    uint32_t m_planeChannelCount[DW_MAX_IMAGE_PLANES]{};
    dwVector2ui m_planeSize[DW_MAX_IMAGE_PLANES]{};

    dwImageStreamerHandle_t m_streamerToCPU   = DW_NULL_HANDLE;
    dwImageStreamerHandle_t m_streamerFromCPU = DW_NULL_HANDLE;

    dwContextHandle_t m_ctx = DW_NULL_HANDLE;
};

class ImageHandleNvSciPacket : public IChannelPacket, public IChannelPacket::NvSciCallbacks
{
    static constexpr uint32_t NUM_BUFFERS{2U};

    struct CPUData
    {
        dwImageMetaData metadata{};
        dwTime_t timestamp{};
    };

public:
    /*! Constructor
     *  @param[in] specimen GenericData represention of Image handle
     *  @param[in] ctx driveworks context */
    ImageHandleNvSciPacket(const GenericData& specimen, dwContextHandle_t ctx);
    /*! Destructor */
    ~ImageHandleNvSciPacket();
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
        return NUM_BUFFERS;
    }

    /*! Getter for generic data in channel packet
     *  @return GenericData representation of Image handle */
    inline GenericData getGenericData() final
    {
        return GenericData(&m_dispatch);
    }

private:
    dwContextHandle_t m_ctx{};
    std::unique_ptr<Buffer> m_cpuDataBuffer{};
    CPUData* m_cpuData{};
    dwImageProperties m_reference{};
    dwImageHandle_t m_orig{};
    uint8_t* m_origTopLineData{};
    uint8_t* m_origBottomLineData{};
    dwImageHandle_t m_dispatch{};
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_IMAGE_HPP_
