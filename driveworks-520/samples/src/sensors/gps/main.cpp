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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/signal/SignalStatus.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/gps/GPS.h>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
static bool gRun          = true;
static bool gDualMode     = false;
static bool enableDumpAll = false;

static std::unordered_map<dwSensorErrorID, std::string> gErrorStrings = {
    {DW_SENSORS_ERROR_CODE_GPS_MODE, "GPS sensor working in wrong modes"},
    {DW_SENSORS_ERROR_CODE_GPS_ACCURACY, "GPS sensor is not working in most accurate mode"},
};

//------------------------------------------------------------------------------
void sig_int_handler(int)
{
    gRun = false;
}

void dumpHeader()
{
    std::cout << "dump_dwGPSFrame,"
              << "timestamp_us,"
              << "latitude,"
              << "longitude,"
              << "altitude,"
              << "course,"
              << "speed,"
              << "climb,"
              << "hdop,"
              << "vdop,"
              << "pdop,"
              << "hacc,"
              << "vacc,"
              << "utcTimeUs,"
              << "satelliteCount,"
              << "fixStatus,"
              << "timestampQuality,"
              << "mode,"
              << std::endl;
}

void dumpAll(const dwGPSFrame gpsFrame)
{
    std::cout << "dump_dwGPSFrame,"
              << std::fixed << std::setprecision(8)
              << gpsFrame.timestamp_us << ","
              << gpsFrame.latitude << ","
              << gpsFrame.longitude << ","
              << gpsFrame.altitude << ","
              << gpsFrame.course << ","
              << gpsFrame.speed << ","
              << gpsFrame.climb << ","
              << gpsFrame.hdop << ","
              << gpsFrame.vdop << ","
              << gpsFrame.pdop << ","
              << gpsFrame.hacc << ","
              << gpsFrame.vacc << ","
              << gpsFrame.utcTimeUs << ","
              << static_cast<uint32_t>(gpsFrame.satelliteCount) << ","
              << gpsFrame.fixStatus << ","
              << gpsFrame.timestampQuality << ","
              << gpsFrame.mode << ","
              << std::endl;
}

//------------------------------------------------------------------------------
void process_seeking(ProgramArguments const& arguments, dwSensorHandle_t gpsSensor)
{
    dwStatus seekStatus = DW_SUCCESS;

    if (arguments.has("seek-to-event"))
    {
        size_t event = 0;
        try
        {
            event      = std::stoi(arguments.get("seek-to-event"), nullptr);
            seekStatus = dwSensor_seekToEvent(event, gpsSensor);
        }
        catch (std::exception&)
        {
            std::cout << "Invalid seek-to-event value provided. Will not perform seek" << std::endl;
        }
    }
    else if (arguments.has("seek-to-timestamp"))
    {
        dwTime_t timestamp = 0;
        try
        {
            timestamp  = std::stoll(arguments.get("seek-to-timestamp"), nullptr);
            seekStatus = dwSensor_seekToTime(timestamp, gpsSensor);
        }
        catch (std::exception&)
        {
            std::cout << "Invalid seek-to-timestamp value provided. Will not perform seek" << std::endl;
        }
    }

    if (seekStatus != DW_SUCCESS)
    {
        switch (seekStatus)
        {
        case DW_INVALID_ARGUMENT:
            std::cout << "Cannot seek to out-of-range timestamps or events" << std::endl;
            break;
        case DW_NOT_SUPPORTED:
            std::cout << "Specified GPS sensor does not implement seek functionality" << std::endl;
            break;
        case DW_NOT_AVAILABLE:
            std::cout << "No seek table found for recording file. Please specify create_seek=1 and try again" << std::endl;
            break;
        default:
            std::cout << "Seek error: " << seekStatus << std::endl;
            break;
        }
    }
}

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    struct sigaction action = {};
    action.sa_handler       = sig_int_handler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command

    gRun = true;

    ProgramArguments arguments(
        {ProgramArguments::Option_t("driver", "gps.virtual"),
         ProgramArguments::Option_t("dual-mode", "false"),
         ProgramArguments::Option_t("params", (std::string("file=") + dw_samples::SamplesDataPath::get() + "/samples/sensors/gps/1.gps").c_str()),
         ProgramArguments::Option_t("seek-to-event", false),
         ProgramArguments::Option_t("enable-dump-all", false),
         ProgramArguments::Option_t("seek-to-timestamp", false)});

    if (!arguments.parse(argc, argv) || (!arguments.has("driver") && !arguments.has("params")))
    {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--driver=gps.virtual \t\t\t: one of the available GPS drivers "
                  << "(see sample_sensors_info)\n";
        std::cout << "\t--params=file=file.gps,arg2=value \t: comma separated "
                  << "key=value parameters for the sensor "
                  << "(see sample_sensor_info for a set of supported parameters)\n";
        std::cout << "\t--seek-to-event=<N> \t\t\t: Start reading GPS frames from Nth event\n";
        std::cout << "\t--seek-to-timestamp=<N> \t\t: Start reading GPS frames from specified timestamp "
                  << "(will output frames with timestamps equal or greater than specified timestamp)\n";

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

    // Disable dual mode since there will be conflict for some specific sensor if the resource is exclusive
    gDualMode = arguments.get("dual-mode") == "true";
    // open same GPS sensor twice, to demonstrate capability of sensor data sharing
    dwSensorHandle_t gpsSensor[2] = {DW_NULL_HANDLE, DW_NULL_HANDLE};
    int numSensors                = gDualMode ? 2 : 1;
    for (int32_t i = 0; i < numSensors; i++)
    {
        dwSensorParams params{};
        std::string parameterString = arguments.get("params");
        params.parameters           = parameterString.c_str();
        params.protocol             = arguments.get("driver").c_str();
        enableDumpAll               = arguments.get("enable-dump-all") == "true";
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

    for (int32_t i = 0; i < numSensors; i++)
    {
        gRun = gRun && dwSensor_start(gpsSensor[i]) == DW_SUCCESS;
    }

    process_seeking(arguments, gpsSensor[0]);
    if (enableDumpAll)
    {
        dumpHeader();
    }

    // Message msg;
    bool sensorRun[2] = {gRun, gRun};
    while (gRun)
    {
        if (!sensorRun[0] && !sensorRun[numSensors - 1])
            break;

        for (int i = 0; i < numSensors; i++)
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
                std::cout << std::setprecision(10);

                if (dwSignal_checkSignalValidity(frame.validityInfo.latitude) == DW_SUCCESS)
                    std::cout << " lat: " << frame.latitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.longitude) == DW_SUCCESS)
                    std::cout << " lon: " << frame.longitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.altitude) == DW_SUCCESS)
                    std::cout << " alt: " << frame.altitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.course) == DW_SUCCESS)
                    std::cout << " course: " << frame.course;

                if (dwSignal_checkSignalValidity(frame.validityInfo.speed) == DW_SUCCESS)
                    std::cout << " speed: " << frame.speed;

                if (dwSignal_checkSignalValidity(frame.validityInfo.climb) == DW_SUCCESS)
                    std::cout << " climb: " << frame.climb;

                if (dwSignal_checkSignalValidity(frame.validityInfo.hdop) == DW_SUCCESS)
                    std::cout << " hdop: " << frame.hdop;

                if (dwSignal_checkSignalValidity(frame.validityInfo.vdop) == DW_SUCCESS)
                    std::cout << " vdop: " << frame.vdop;

                if (dwSignal_checkSignalValidity(frame.validityInfo.hacc) == DW_SUCCESS)
                    std::cout << " hacc: " << frame.hacc;

                if (dwSignal_checkSignalValidity(frame.validityInfo.vacc) == DW_SUCCESS)
                    std::cout << " vacc: " << frame.vacc;

                if (dwSignal_checkSignalValidity(frame.validityInfo.mode) == DW_SUCCESS)
                    std::cout << " gps mode: " << frame.mode;

                if (dwSignal_checkSignalValidity(frame.validityInfo.utcTimeUs) == DW_SUCCESS)
                    std::cout << " gps utc time: " << frame.utcTimeUs;

                if (dwSignal_checkSignalValidity(frame.validityInfo.timestampQuality) == DW_SUCCESS)
                    std::cout << " gps timestamp status: " << frame.timestampQuality;

                std::cout << " errorID(" << frame.errors << ")";

                if (frame.errors)
                {
                    std::cout << " error messages: [";
                    for (auto& error : gErrorStrings)
                    {
                        if (error.first & frame.errors)
                        {
                            std::cout << error.second << ",";
                        }
                    }
                    std::cout << "]";
                }
            }
            std::cout << std::endl;
            if (enableDumpAll)
            {
                dumpAll(frame);
            }
        }
    }

    for (int32_t i = 0; i < numSensors; i++)
    {
        dwSensor_stop(gpsSensor[i]);
        dwSAL_releaseSensor(gpsSensor[i]);
    }

    // release used objects in correct order
    dwSAL_release(hal);
    dwRelease(sdk);
    dwLogger_release();

    return 0;
}
