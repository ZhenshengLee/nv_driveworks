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
// Copyright (c) 2019-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
///////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_CHANNEL_NVSCIHELPER_HPP_
#define DW_FRAMEWORK_CHANNEL_NVSCIHELPER_HPP_

#include <cstdint>
#include <nvscierror.h>

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

namespace dw
{
namespace framework
{

const char* nvSciGetEventName(uint32_t event);
const char* nvSciGetErrorName(uint32_t error);

} // namespace framework
} // namespace dw

#define FRWK_CHECK_NVSCI_ERROR(e)                                                                                            \
    {                                                                                                                        \
        NvSciError FRWK_CHECK_NVSCI_ERROR_ret{(e)};                                                                          \
        if (FRWK_CHECK_NVSCI_ERROR_ret != NvSciError_Success)                                                                \
        {                                                                                                                    \
            DW_LOGE << "Failed with " << dw::framework::nvSciGetErrorName(static_cast<uint32_t>(FRWK_CHECK_NVSCI_ERROR_ret)) \
                    << "(" << static_cast<uint32_t>(FRWK_CHECK_NVSCI_ERROR_ret) << ")"                                       \
                    << " in " << __FILE__                                                                                    \
                    << ":" << __LINE__ << Logger::State::endl;                                                               \
            if (FRWK_CHECK_NVSCI_ERROR_ret == NvSciError_Timeout)                                                            \
                throw dw::core::ExceptionWithStatus(DW_TIME_OUT, "NvSci API Timeout");                                       \
            else                                                                                                             \
                throw dw::core::ExceptionWithStatus(DW_INTERNAL_ERROR, "NvSci internal error occured");                      \
        }                                                                                                                    \
    }

#endif // DW_FRAMEWORK_CHANNEL_NVSCIHELPER_HPP_
