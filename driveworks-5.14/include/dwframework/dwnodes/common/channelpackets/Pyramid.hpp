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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/pyramid/Pyramid.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dwcgf/channel/ChannelPacketImpl.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/channel/IChannelPacket.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwPyramidImage, dwPyramidImageProperties, DW_PYRAMID_IMAGE);

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
};

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
    dwImageStreamerHandle_t m_streamerToCPU[DW_PYRAMID_LEVEL_MAX_COUNT]{};
    dwImageStreamerHandle_t m_streamerFromCPU[DW_PYRAMID_LEVEL_MAX_COUNT]{};
    dwImageHandle_t m_imageHandleCPU[DW_PYRAMID_LEVEL_MAX_COUNT]{};
    dwContextHandle_t m_ctx = DW_NULL_HANDLE;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_PYRAMID_HPP_
