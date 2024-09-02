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
// SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cmath>
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
#include <dw/sensors/imu/IMU.h>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
static bool gRun              = true;
static bool timestampTrace    = false;
static bool enableDumpAll     = false;
static dwTime_t lastTimestamp = 0;

static std::unordered_map<dwSensorErrorID, std::string> gErrorStrings = {
    {DW_SENSORS_ERROR_CODE_IMU_ALIGNMENT_STATUS, "IMU Alignment status error"},
};

//------------------------------------------------------------------------------
void sig_int_handler(int)
{
    gRun = false;
}
void dumpHeader()
{
    std::cout << "dump_dwIMUFrame,"
              << "hostTimestamp,"
              << "orientation[0],"
              << "orientation[1],"
              << "orientation[2],"
              << "orientationQuaternion.x,"
              << "orientationQuaternion.y,"
              << "orientationQuaternion.z,"
              << "orientationQuaternion.w,"
              << "turnrate[0],"
              << "turnrate[1],"
              << "turnrate[2],"
              << "acceleration[0],"
              << "acceleration[1],"
              << "acceleration[2],"
              << "magnetometer[0],"
              << "magnetometer[1],"
              << "magnetometer[2],"
              << "heading,"
              << "temperature,"
              << "accelerationOffset[0],"
              << "accelerationOffset[1],"
              << "accelerationOffset[2],"
              << "turnrateOffset[0],"
              << "turnrateOffset[1],"
              << "turnrateOffset[2],"
              << "turnrateAccel[0],"
              << "turnrateAccel[1],"
              << "turnrateAccel[2],"
              << "imuTempQuality,"
              << "imuAccelerationQuality[0],"
              << "imuAccelerationQuality[1],"
              << "imuAccelerationQuality[2],"
              << "imuTurnrateQuality[0],"
              << "imuTurnrateQuality[1],"
              << "imuTurnrateQuality[2],"
              << "imuTurnrateOffsetQuality[0],"
              << "imuTurnrateOffsetQuality[1],"
              << "imuTurnrateOffsetQuality[2],"
              << "imuTurnrateAccelQuality[0],"
              << "imuTurnrateAccelQuality[1],"
              << "imuTurnrateAccelQuality[2],"
              << "imuTimestampQuality,"
              << "imuStatus,"
              << "timestamp_us,"
              << "sensorTimestamp,"
              << "sqc"
              << "\n";
}

void dumpAll(const dwIMUFrame& frame)
{
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic push
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    std::cout << "dump_dwIMUFrame,"
              << std::fixed << std::setprecision(8)
              << frame.hostTimestamp << ","
              << frame.orientation[0] << ","
              << frame.orientation[1] << ","
              << frame.orientation[2] << ","
              << frame.orientationQuaternion.x << ","
              << frame.orientationQuaternion.y << ","
              << frame.orientationQuaternion.z << ","
              << frame.orientationQuaternion.w << ","
              << frame.turnrate[0] << ","
              << frame.turnrate[1] << ","
              << frame.turnrate[2] << ","
              << frame.acceleration[0] << ","
              << frame.acceleration[1] << ","
              << frame.acceleration[2] << ","
              << frame.magnetometer[0] << ","
              << frame.magnetometer[1] << ","
              << frame.magnetometer[2] << ","
              << frame.heading << ","
              << frame.temperature << ","
              << frame.accelerationOffset[0] << ","
              << frame.accelerationOffset[1] << ","
              << frame.accelerationOffset[2] << ","
              << frame.turnrateOffset[0] << ","
              << frame.turnrateOffset[1] << ","
              << frame.turnrateOffset[2] << ","
              << frame.turnrateAccel[0] << ","
              << frame.turnrateAccel[1] << ","
              << frame.turnrateAccel[2] << ","
              << frame.imuTempQuality << ","
              << frame.imuAccelerationQuality[0] << ","
              << frame.imuAccelerationQuality[1] << ","
              << frame.imuAccelerationQuality[2] << ","
              << frame.imuTurnrateQuality[0] << ","
              << frame.imuTurnrateQuality[1] << ","
              << frame.imuTurnrateQuality[2] << ","
              << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[0]) << ","
              << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[1]) << ","
              << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[2]) << ","
              << frame.imuTurnrateAccelQuality[0] << ","
              << frame.imuTurnrateAccelQuality[1] << ","
              << frame.imuTurnrateAccelQuality[2] << ","
              << frame.imuTimestampQuality << ","
              << frame.imuStatus << ","
              << frame.timestamp_us << ","
              << frame.sensorTimestamp << ","
              << static_cast<uint32_t>(frame.sequenceCounter) << "\n";
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic pop
}

//------------------------------------------------------------------------------
void printAll(const dwIMUFrame& frame)
{
    bool containIMUInfo = false;

    if (timestampTrace)
    {
        std::cout << "[" << frame.hostTimestamp << "] ";
        std::cout << "lastTimeStamp: " << lastTimestamp << " ";
        std::cout << "delta: " << frame.hostTimestamp - lastTimestamp;
        std::cout << std::endl;

        lastTimestamp = frame.hostTimestamp;
    }
    else
    {
        std::cout << "[" << frame.hostTimestamp << "] ";
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic push
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        std::cout << "[" << frame.timestamp_us << "] ";
// coverity[autosar_cpp14_a16_7_1_violation] RFD Pending: TID-2023
#pragma GCC diagnostic pop

        // orientation
        if (dwSignal_checkSignalValidity(frame.validityInfo.orientation[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.orientation[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.orientation[2]) == DW_SUCCESS)
        {
            std::cout << "Orientation(";

            if (!std::isnan(frame.orientation[0]))
                std::cout << "R:" << frame.orientation[0] << " ";
            if (!std::isnan(frame.orientation[1]))
                std::cout << "P:" << frame.orientation[1] << " ";
            if (!std::isnan(frame.orientation[2]))
                std::cout << "Y:" << frame.orientation[2] << " ";

            std::cout << ") ";

            containIMUInfo = true;
        }

        // orientationQuaternion
        if (dwSignal_checkSignalValidity(frame.validityInfo.orientationQuaternion) == DW_SUCCESS)
        {
            std::cout << "OrientationQuaternion(";

            if (!std::isnan(frame.orientationQuaternion.x))
                std::cout << "X:" << frame.orientationQuaternion.x << " ";
            if (!std::isnan(frame.orientationQuaternion.y))
                std::cout << "Y:" << frame.orientationQuaternion.y << " ";
            if (!std::isnan(frame.orientationQuaternion.z))
                std::cout << "Z:" << frame.orientationQuaternion.z << " ";
            if (!std::isnan(frame.orientationQuaternion.w))
                std::cout << "W:" << frame.orientationQuaternion.w << " ";

            std::cout << ") ";

            containIMUInfo = true;
        }

        // gyroscope
        if (dwSignal_checkSignalValidity(frame.validityInfo.turnrate[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrate[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrate[2]) == DW_SUCCESS)
        {
            std::cout << "Gyro(";

            if (!std::isnan(frame.turnrate[0]))
                std::cout << "X:" << frame.turnrate[0] << " ";
            if (!std::isnan(frame.turnrate[1]))
                std::cout << "Y:" << frame.turnrate[1] << " ";
            if (!std::isnan(frame.turnrate[2]))
                std::cout << "Z:" << frame.turnrate[2] << " ";

            std::cout << ") ";
            containIMUInfo = true;
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.turnrateAccel[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrateAccel[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrateAccel[2]) == DW_SUCCESS)
        {
            std::cout << "turnrateAccel(";

            if (!std::isnan(frame.turnrateAccel[0]))
                std::cout << "X:" << frame.turnrateAccel[0] << " ";
            if (!std::isnan(frame.turnrateAccel[1]))
                std::cout << "Y:" << frame.turnrateAccel[1] << " ";
            if (!std::isnan(frame.turnrateAccel[2]))
                std::cout << "Z:" << frame.turnrateAccel[2] << " ";

            std::cout << ") ";
            containIMUInfo = true;
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.turnrateOffset[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrateOffset[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.turnrateOffset[2]) == DW_SUCCESS)
        {
            std::cout << "turnrateOffset(";

            if (!std::isnan(frame.turnrateOffset[0]))
                std::cout << "X:" << frame.turnrateOffset[0] << " ";
            if (!std::isnan(frame.turnrateOffset[1]))
                std::cout << "Y:" << frame.turnrateOffset[1] << " ";
            if (!std::isnan(frame.turnrateOffset[2]))
                std::cout << "Z:" << frame.turnrateOffset[2] << " ";

            std::cout << ") ";
            containIMUInfo = true;
        }

        // heading (i.e. compass)
        if (dwSignal_checkSignalValidity(frame.validityInfo.heading) == DW_SUCCESS)
        {
            std::cout << "Heading(" << frame.heading << ") ";
            containIMUInfo = true;
        }

        // Acceleration
        if (dwSignal_checkSignalValidity(frame.validityInfo.acceleration[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.acceleration[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.acceleration[2]) == DW_SUCCESS)
        {
            std::cout << "Acceleration(";

            if (!std::isnan(frame.acceleration[0]))
                std::cout << "X:" << frame.acceleration[0] << " ";
            if (!std::isnan(frame.acceleration[1]))
                std::cout << "Y:" << frame.acceleration[1] << " ";
            if (!std::isnan(frame.acceleration[2]))
                std::cout << "Z:" << frame.acceleration[2] << " ";

            std::cout << ") ";
            containIMUInfo = true;
        }

        // magnetometer
        if (dwSignal_checkSignalValidity(frame.validityInfo.magnetometer[0]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.magnetometer[1]) == DW_SUCCESS ||
            dwSignal_checkSignalValidity(frame.validityInfo.magnetometer[2]) == DW_SUCCESS)
        {

            std::cout << "Magnetometer(";

            if (!std::isnan(frame.magnetometer[0]))
                std::cout << "X:" << frame.magnetometer[0] << " ";
            if (!std::isnan(frame.magnetometer[1]))
                std::cout << "Y:" << frame.magnetometer[1] << " ";
            if (!std::isnan(frame.magnetometer[2]))
                std::cout << "Z:" << frame.magnetometer[2] << " ";

            std::cout << ") ";
            containIMUInfo = true;
        }

        // Alignment status
        if (dwSignal_checkSignalValidity(frame.validityInfo.alignmentStatus) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "INS status " << frame.alignmentStatus;
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.sensorTimestamp) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "Sensor timestamp " << frame.sensorTimestamp;
        }

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

        if (dwSignal_checkSignalValidity(frame.validityInfo.sequenceCounter) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << " sqc " << static_cast<uint32_t>(frame.sequenceCounter);
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.imuTurnrateQuality[0]) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "  imuTurnrateQuality " << static_cast<uint32_t>(frame.imuTurnrateQuality[0]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateQuality[1]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateQuality[2]);
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.imuTurnrateAccelQuality[0]) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "  imuTurnrateAccelQuality " << static_cast<uint32_t>(frame.imuTurnrateAccelQuality[0]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateAccelQuality[1]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateAccelQuality[2]);
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.imuTurnrateOffsetQuality[0]) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "  imuTurnrateOffsetQuality " << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[0]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[1]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuTurnrateOffsetQuality[2]);
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.imuAccelerationQuality[0]) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << "  imuAccelerationQuality " << static_cast<uint32_t>(frame.imuAccelerationQuality[0]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuAccelerationQuality[1]);
            std::cout << "  " << static_cast<uint32_t>(frame.imuAccelerationQuality[2]);
        }

        if (dwSignal_checkSignalValidity(frame.validityInfo.temperature) == DW_SUCCESS)
        {
            containIMUInfo = true;
            std::cout << " temp " << frame.temperature;
        }

        if (containIMUInfo == false)
            std::cout << "No IMU related info";

        std::cout << std::endl;
    }
}

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
            std::cout << "Specified IMU sensor does not implement seek functionality" << std::endl;
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
        {ProgramArguments::Option_t("driver", "imu.virtual"),
         ProgramArguments::Option_t("params", (std::string("file=") + dw_samples::SamplesDataPath::get() + "/samples/sensors/imu/imu.bin").c_str()),
         ProgramArguments::Option_t("timestamp-trace", "false"),
         ProgramArguments::Option_t("seek-to-event", false),
         ProgramArguments::Option_t("enable-dump-all", false),
         ProgramArguments::Option_t("seek-to-timestamp", false)});

    if (!arguments.parse(argc, argv) || (!arguments.has("driver") && !arguments.has("params")))
    {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--driver=imu.virtual \t\t\t: one of the available IMU drivers "
                  << "(see sample_sensors_info)\n";
        std::cout << "\t--params=file=file.txt,arg2=value \t: comma separated "
                  << "key=value parameters for the sensor "
                  << "(see sample_sensor_info for a set of supported parameters)";
        std::cout << "\t--timestamp-trace=<true/false>\t\t: enables timestamp tracing only\n";
        std::cout << "\t--seek-to-event=<N> \t\t\t: Start reading IMU frames from Nth event\n";
        std::cout << "\t--seek-to-timestamp=<N> \t\t: Start reading IMU frames from specified timestamp "
                  << "(will output frames with timestamps equal or greater than specified timestamp)\n";

        return -1;
    }

    dwContextHandle_t sdk = DW_NULL_HANDLE;
    dwSALHandle_t hal     = DW_NULL_HANDLE;

    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

#if (VIBRANTE_PDK_DECIMAL == 6000400)
    // Some sensors rely on FSI Coms which requires that SIGRTMIN is blocked by client application
    // threads as the signal is used by the NvFsiCom daemon.
    sigset_t lSigSet;
    (void)sigemptyset(&lSigSet);
    sigaddset(&lSigSet, SIGRTMIN);
    if (pthread_sigmask(SIG_BLOCK, &lSigSet, NULL) != 0)
        std::cout << "pthread_sigmask failed" << std::endl;
#endif

    // instantiate Driveworks SDK context
    CHECK_DW_ERROR(dwInitialize(&sdk, DW_VERSION, nullptr));

    // create HAL module of the SDK
    dwSAL_initialize(&hal, sdk);

    // create IMUM bus interface
    dwSensorHandle_t imuSensor = DW_NULL_HANDLE;
    {
        dwSensorParams params{};
        std::string parameterString = arguments.get("params");
        params.parameters           = parameterString.c_str();
        params.protocol             = arguments.get("driver").c_str();
        timestampTrace              = arguments.get("timestamp-trace") == "true";
        enableDumpAll               = arguments.get("enable-dump-all") == "true";
        if (dwSAL_createSensor(&imuSensor, params, hal) != DW_SUCCESS)
        {
            std::cout << "Cannot create sensor " << params.protocol
                      << " with " << params.parameters << std::endl;

            dwSAL_release(hal);
            dwRelease(sdk);
            dwLogger_release();

            return -1;
        }
    }

    gRun = dwSensor_start(imuSensor) == DW_SUCCESS;

    process_seeking(arguments, imuSensor);
    if (enableDumpAll)
    {
        dumpHeader();
    }

    while (gRun)
    {

        dwIMUFrame frame;
        dwStatus status = dwSensorIMU_readFrame(&frame, 10000, imuSensor);

        if (status == DW_END_OF_STREAM)
        {
            std::cout << "IMU end of stream reached" << std::endl;
            break;
        }
        else if (status == DW_TIME_OUT || status == DW_NOT_READY)
            continue;
        else if (status == DW_SUCCESS)
        {
            if (enableDumpAll)
            {
                dumpAll(frame);
            }
            printAll(frame); // Print all IMU outputs
        }
        else
        {
            std::cerr << " ERROR when reading IMU frame" << dwGetStatusName(status);
            break;
        }
    }

    dwSensor_stop(imuSensor);
    dwSAL_releaseSensor(imuSensor);

    // release used objects in correct order
    dwSAL_release(hal);
    dwRelease(sdk);
    dwLogger_release();

    return 0;
}
