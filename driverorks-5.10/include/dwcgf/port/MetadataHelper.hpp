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
// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
///////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_METADATAHELPER_HPP_
#define DW_FRAMEWORK_METADATAHELPER_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>

#include <dw/core/language/Optional.hpp>

namespace dw
{
namespace framework
{

bool checkHeaderValidFlag(ChannelMetadata const& header, MetadataFlags flag);
MetadataPayload* extractMetadata(GenericData packet);

// Validity helper functions
void setValidityStatus(ChannelMetadata& header, dwValidityStatus const& status);
dw::core::Optional<dwValidityStatus> getValidityStatus(ChannelMetadata const& header);

// Timestamp helper functions
void setTimestamp(ChannelMetadata& header, dwTime_t const& timestamp);
dw::core::Optional<dwTime_t> getTimestamp(ChannelMetadata const& header);

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_METADATAHELPER_HPP_
