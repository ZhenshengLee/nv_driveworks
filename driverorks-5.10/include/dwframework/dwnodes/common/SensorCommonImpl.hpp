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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SENSOR_COMMON_IMPL_HPP_
#define SENSOR_COMMON_IMPL_HPP_

#include <dw/core/base/Types.h>
#include <dwcgf/Exception.hpp>
#include <dw/rig/Rig.h>
#include <dw/core/logger/Logger.hpp>

namespace dw
{
namespace framework
{

class SensorCommonImpl
{

    using FixedString1024 = dw::core::FixedString<1024>;

public:
    static FixedString1024 adjustPath(const char* paramString, uint64_t sensorId, const char* toFind, dwConstRigHandle_t& rig)
    {
        constexpr char LOG_TAG[] = "SensorCommonImpl";
        FixedString1024 parameters(paramString);
        const char* path = nullptr;
        FRWK_CHECK_DW_ERROR(dwRig_getSensorDataPath(&path, static_cast<uint32_t>(sensorId), rig));
        if (path == nullptr)
        {
            throw Exception(DW_INVALID_ARGUMENT, "Driveworks sensor adjustPath: no sensor data found for sensorId ", sensorId,
                            " with param: \"", paramString, "\", please check whether the data path in param is accessible");
        }

        FixedString1024 p(path);
        FixedString1024 param(paramString);
        auto start = param.find(toFind);
        if (start == FixedString1024::NPOS)
        {
            return parameters;
        }

        FixedString1024 find(toFind);
        // String from start to end of pattern
        auto first = param.substr(0, start + find.size());
        // String from end of pattern to end of string
        auto second = param.substr(start + find.size(), param.size());

        // Extract file name from rest of arguments
        start = second.find(",");
        if (start == FixedString1024::NPOS)
        {
            start = second.size();
        }

        // Save original video path for later use
        auto origVideoPath = second.substr(0, start);

        // Append absolute path (actually 'p' might be not absolute, just completed with relative rig path dir)
        parameters = first.c_str();
        parameters += p;
        second = second.substr(start, second.size());

        // Append the rest of the param, if there's any
        parameters += second.c_str();

        if (find == "video=")
        {
            start = parameters.find("timestamp=");
            if (start != FixedString1024::NPOS)
            {
                // Extract the common prefix (i.e. rig dir) from video absolute path 'p'
                auto commonPrefix = p.rfind(origVideoPath);
                if (commonPrefix == FixedString1024::NPOS)
                {
                    // Absolute path returned from dwRig is supposed to have "original path" as suffix
                    // If that's not true, we cannot adjust timestamp path, just keep it unchanged
                    DW_LOGW << "adjustPath: [sensorId: " << sensorId << " with param:" << paramString << "] Failed to extract rig directory path, so that cannot adjust timestamp path" << Logger::State::endl;
                    commonPrefix = 0;
                }
                p = p.substr(0, commonPrefix);
                // Add common prefix to timestamp path
                auto patternSize = FixedString1024("timestamp=").size();
                first            = parameters.substr(0, start + patternSize);
                second           = parameters.substr(start + patternSize, parameters.size());
                parameters       = first;
                parameters += p;
                parameters += second;
            }
        }
        return parameters;
    }

    static dwSensorHandle_t createSensor(std::string sensorName, dwConstRigHandle_t rig, dwSALHandle_t sal)
    {
        using FixedString1024 = dw::core::FixedString<1024>;

        dwSensorParams sensorParams{};
        uint32_t sensorId             = 0;
        dwSensorHandle_t sensorHandle = DW_NULL_HANDLE;
        FRWK_CHECK_DW_ERROR(dwRig_findSensorByName(&sensorId, sensorName.c_str(), rig));
        FRWK_CHECK_DW_ERROR(dwRig_getSensorParameter(&sensorParams.parameters, sensorId, rig));
        FRWK_CHECK_DW_ERROR(dwRig_getSensorProtocol(&sensorParams.protocol, sensorId, rig));

        // Fix virtual sensor paths to be absolute rather than relative so sensor
        // initialize gets the right file path.
        FixedString1024 parameters("");
        if (FixedString1024(sensorParams.protocol).find(".virtual") != FixedString1024::NPOS)
        {
            parameters = dw::framework::SensorCommonImpl::adjustPath(sensorParams.parameters, sensorId, "video=", rig);
            parameters = dw::framework::SensorCommonImpl::adjustPath(parameters.c_str(), sensorId, "file=", rig);

            sensorParams.parameters = parameters.c_str();
        }

        FRWK_CHECK_DW_ERROR_MSG(dwSAL_createSensor(&sensorHandle, sensorParams, sal),
                                "Cannot create sensor");
        return sensorHandle;
    }
};
} // namespace framework
} // namespace dw

#endif // SENSOR_COMMON_IMPL_HPP_
