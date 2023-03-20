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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <signal.h>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <memory>

#ifdef LINUX
#include <execinfo.h>
#include <unistd.h>
#endif

#include <cstring>
#include <functional>
#include <list>
#include <iomanip>

#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/Log.hpp>
#include <framework/Checks.hpp>

// CORE
#include <dw/core/logger/Logger.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Version.h>

// HAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/gps/GPS.h>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
static bool gRun = true;

//------------------------------------------------------------------------------
void sig_int_handler(int)
{
    gRun = false;
}

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
#ifndef WINDOWS
    struct sigaction action = {};
    action.sa_handler       = sig_int_handler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command
#endif

    gRun = true;

    ProgramArguments arguments(
        {ProgramArguments::Option_t("driver", "gps.virtual"),
         ProgramArguments::Option_t("params", (std::string("file=") + dw_samples::SamplesDataPath::get() + "/samples/sensors/gps/1.gps").c_str())});

    if (!arguments.parse(argc, argv) || (!arguments.has("driver") && !arguments.has("params")))
    {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--driver=gps.virtual \t\t\t: one of the available GPS drivers "
                  << "(see sample_sensors_info)\n";
        std::cout << "\t--params=file=file.gps,arg2=value \t: comma separated "
                  << "key=value parameters for the sensor "
                  << "(see sample_sensor_info for a set of supported parameters)\n";

        return -1;
    }

    dwContextHandle_t sdk = DW_NULL_HANDLE;
    dwSALHandle_t hal     = DW_NULL_HANDLE;

    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams = {};

    CHECK_DW_ERROR(dwInitialize(&sdk, DW_VERSION, &sdkParams));

    // create HAL module of the SDK
    dwSAL_initialize(&hal, sdk);

    // open same GPS sensor twice, to demonstrate capability of sensor data sharing
    dwSensorHandle_t gpsSensor[2] = {DW_NULL_HANDLE, DW_NULL_HANDLE};
    for (int32_t i = 0; i < 2; i++)
    {
        dwSensorParams params{};
        std::string parameterString = arguments.get("params");
        params.parameters           = parameterString.c_str();
        params.protocol             = arguments.get("driver").c_str();
        if (dwSAL_createSensor(&gpsSensor[i], params, hal) != DW_SUCCESS)
        {
            std::cout << "Cannot create sensor " << params.protocol
                      << " with " << params.parameters << std::endl;

            dwSAL_release(hal);
            dwRelease(sdk);
            dwLogger_release();

            return -1;
        }
    }

    gRun = gRun && dwSensor_start(gpsSensor[0]) == DW_SUCCESS;
    gRun = gRun && dwSensor_start(gpsSensor[1]) == DW_SUCCESS;

    // Message msg;
    bool sensorRun[2] = {gRun, gRun};
    while (gRun)
    {

        if (!sensorRun[0] && !sensorRun[1])
            break;

        for (int i = 0; i < 2; i++)
        {
            if (!sensorRun[i])
                continue;

            dwGPSFrame frame;
            dwStatus status = DW_FAILURE;

            status = dwSensorGPS_readFrame(&frame, 50000, gpsSensor[i]);

            if (status == DW_END_OF_STREAM)
            {
                std::cout << "GPS[" << i << "] end of stream reached" << std::endl;
                sensorRun[i] = false;
                break;
            }
            else if (status == DW_TIME_OUT)
                continue;

            // log message
            std::cout << "GPS[" << i << "] - " << frame.timestamp_us;
            if (status != DW_SUCCESS) // msg.is_error)
            {
                std::cout << " ERROR " << dwGetStatusName(status); // msg.frame.id;
            }
            else
            {
                auto isSignalValid = [](const dwSignalValidity& validity) -> bool {
                    dwSignalStatus status;
                    dwSignalTimeoutStatus timeoutStatus;
                    dwSignalE2EStatus e2eStatus;
                    dwStatus retVal = dwSignal_decodeSignalValidity(&status, &timeoutStatus, &e2eStatus, validity);
                    return (retVal == DW_SUCCESS && status == DW_SIGNAL_STATUS_LAST_VALID);
                };

                std::cout << std::setprecision(10);

                if (isSignalValid(frame.validityInfo.latitude))
                    std::cout << " lat: " << frame.latitude;

                if (isSignalValid(frame.validityInfo.longitude))
                    std::cout << " lon: " << frame.longitude;

                if (isSignalValid(frame.validityInfo.altitude))
                    std::cout << " alt: " << frame.altitude;

                if (isSignalValid(frame.validityInfo.course))
                    std::cout << " course: " << frame.course;

                if (isSignalValid(frame.validityInfo.speed))
                    std::cout << " speed: " << frame.speed;

                if (isSignalValid(frame.validityInfo.climb))
                    std::cout << " climb: " << frame.climb;

                if (isSignalValid(frame.validityInfo.hdop))
                    std::cout << " hdop: " << frame.hdop;

                if (isSignalValid(frame.validityInfo.vdop))
                    std::cout << " vdop: " << frame.vdop;

                if (isSignalValid(frame.validityInfo.hacc))
                    std::cout << " hacc: " << frame.hacc;

                if (isSignalValid(frame.validityInfo.vacc))
                    std::cout << " vacc: " << frame.vacc;

                if (isSignalValid(frame.validityInfo.mode))
                    std::cout << " gps mode: " << frame.mode;
            }
            std::cout << std::endl;
        }
    }

    dwSensor_stop(gpsSensor[0]);
    dwSensor_stop(gpsSensor[1]);
    dwSAL_releaseSensor(gpsSensor[0]);
    dwSAL_releaseSensor(gpsSensor[1]);

    // release used objects in correct order
    dwSAL_release(hal);
    dwRelease(sdk);
    dwLogger_release();

    return 0;
}
