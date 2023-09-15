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
// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_VEHICLEIO_TYPES_HPP_
#define DW_FRAMEWORK_VEHICLEIO_TYPES_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>
#include <dw/core/base/Types.h>
#include <dw/control/vehicleio/VehicleIOLegacyStructures.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>
#include <dw/control/vehicleio/VehicleIOExtra.h>

namespace dw
{
namespace framework
{

template <>
struct EnumDescription<dwVehicleIOLatMode>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOLatMode;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_L2_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_PARK),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LAT_MODE_LSS));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VEHICLEIO_LAT_MODE_LSS - DW_VEHICLEIO_LAT_MODE_UNKNOWN + 1), "dwVehicleIOLatMode enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVehicleIOLonMode>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOLonMode;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_PARK),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_CA),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_DRIVE_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_DRIVE_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_SPEED_LIMITING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_LON_MODE_DBS));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VEHICLEIO_LON_MODE_DBS - DW_VEHICLEIO_LON_MODE_UNKNOWN + 1), "dwVehicleIOLonMode enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLonCtrlSafetyLimRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLonCtrlSafetyLimRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_MANEUVERING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_PARKING_CONTROL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_REMOTE_OR_L4_PARKING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_DRIVER_BRAKE_SUPPORT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_COLLISION_AVOIDANCE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_COLLISION_AVOIDANCE_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_CRUISE_CONTROL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_L3_DRIVING));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_L3_DRIVING - DW_VIO_LON_CTRL_SAFETY_LIM_REQUEST_NONE + 1),
                      "dwVioLonCtrlSafetyLimRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlReferenceInputRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlReferenceInputRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_ACCEL_INPUT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_SPEED_INPUT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_DIST_INPUT));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_DIST_INPUT - DW_VIO_LONG_CTRL_REFERENCE_INPUT_REQUEST_IDLE + 1),
                      "dwVioLongCtrlReferenceInputRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlInteractionModeRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlInteractionModeRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_REGENERATION_MODE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_MIN_MODE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_MAX_MODE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_DIRECT_MODE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_DECOUPLING_MODE));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_DECOUPLING_MODE - DW_VIO_LONG_CTRL_INTERACTION_MODE_REQUEST_IDLE + 1),
                      "dwVioLongCtrlInteractionModeRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlAccelPerfRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlAccelPerfRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_COMFORT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_DYNAMIC),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_HIGH_ACCURACY),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_EMERGENCY));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_EMERGENCY - DW_VIO_LONG_CTRL_ACCEL_PERF_REQUEST_NONE + 1),
                      "dwVioLongCtrlAccelPerfRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlSecureRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlSecureRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_SECURE_REQUEST_NOMON_FLWUP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_SECURE_REQUEST_SSCMON_STNDSTILL_SEC),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_SECURE_REQUEST_SSC_SEC_RQ));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_SECURE_REQUEST_SSC_SEC_RQ - DW_VIO_LONG_CTRL_SECURE_REQUEST_NOMON_FLWUP + 1),
                      "dwVioLongCtrlSecureRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlHapticWarningTargetType>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlHapticWarningTargetType;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_HAPTIC_WARNING_TARGET_TYPE_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_HAPTIC_WARNING_TARGET_TYPE_SOFT_TARGET),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_HAPTIC_WARNING_TARGET_TYPE_HARD_TARGET));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_HAPTIC_WARNING_TARGET_TYPE_HARD_TARGET - DW_VIO_LONG_CTRL_HAPTIC_WARNING_TARGET_TYPE_NONE + 1),
                      "dwVioLongCtrlHapticWarningTargetType enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_INIT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_RUN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_TERM),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_TERMINATED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_STATUS_ERROR));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_STATUS_ERROR - DW_VIO_LONG_CTRL_STATUS_UNKNOWN + 1),
                      "dwVioLongCtrlStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlComAvailable>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlComAvailable;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_NO_PATH_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_MAIN_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_SAT_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_MAIN_SAT_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_BMRM_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_MAIN_BMRM_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_SAT_BMRM_VALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_COM_AVAILABLE_MAIN_SAT_BMRM_VALID));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_COM_AVAILABLE_MAIN_SAT_BMRM_VALID - DW_VIO_LONG_CTRL_COM_AVAILABLE_NO_PATH_VALID + 1),
                      "dwVioLongCtrlComAvailable enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlBrakeTorqueAvailable>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlBrakeTorqueAvailable;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_NO_BRKTRQ_AVL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_RED_BRKTRQ_AVL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_RED_BRKTRQ_PT_AVL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_FULL_BRKTRQ_AVL));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_FULL_BRKTRQ_AVL - DW_VIO_LONG_CTRL_BRAKE_TORQUE_AVAILABLE_NO_BRKTRQ_AVL + 1),
                      "dwVioLongCtrlBrakeTorqueAvailable enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlActiveSystem>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlActiveSystem;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_MANEUVER_CTRL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_PARK_CTRL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_REMOTE_CTRL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_DBS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_CA),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_CA_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_CRUISE_CTRL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_L3));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_ASU_L3 - DW_VIO_LONG_CTRL_ACTIVE_SYSTEM_NONE + 1), "dwVioLongCtrlActiveSystem enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

#pragma GCC diagnostic push // PARK_RECORD is deprecated
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <>
struct EnumDescription<dwVioLatCtrlModeStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlModeStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L2_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L2_HFE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_AES),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L3_EMG),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_LSS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_PARK_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_PARK_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_PARK_RECORD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_ESS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_ARP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_L2_PLUS_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_STATUS_EESF));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LAT_CTRL_MODE_STATUS_EESF - DW_VIO_LAT_CTRL_MODE_STATUS_IDLE + 1), "dwVioLongCtrlActiveSystem enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};
#pragma GCC diagnostic pop

template <>
struct EnumDescription<dwVioLatCtrlReferenceInputSelect>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlReferenceInputSelect;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_REFERENCE_INPUT_SELECT_CURVATURE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_REFERENCE_INPUT_SELECT_STEERINGANGLE));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LAT_CTRL_REFERENCE_INPUT_SELECT_STEERINGANGLE - DW_VIO_LAT_CTRL_REFERENCE_INPUT_SELECT_CURVATURE + 1),
                      "dwVioLatCtrlReferenceInputSelect enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioDrivePositionStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioDrivePositionStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVE_POSITION_STATUS_D),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVE_POSITION_STATUS_N),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVE_POSITION_STATUS_R),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVE_POSITION_STATUS_P));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_DRIVE_POSITION_STATUS_P - DW_VIO_DRIVE_POSITION_STATUS_D + 1),
                      "dwVioDrivePositionStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioTurnSignalStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioTurnSignalStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_TURN_SIGNAL_STATUS_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_TURN_SIGNAL_STATUS_OFF),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_TURN_SIGNAL_STATUS_LEFT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_TURN_SIGNAL_STATUS_RIGHT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_TURN_SIGNAL_STATUS_EMERGENCY));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_TURN_SIGNAL_STATUS_EMERGENCY - DW_VIO_TURN_SIGNAL_STATUS_UNKNOWN + 1),
                      "dwVioTurnSignalStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioDriverOverrideThrottle>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioDriverOverrideThrottle;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVER_OVERRIDE_THROTTLE_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVER_OVERRIDE_THROTTLE_NDEF1),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_DRIVER_OVERRIDE_THROTTLE_DRV_OVERRIDE));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_DRIVER_OVERRIDE_THROTTLE_DRV_OVERRIDE - DW_VIO_DRIVER_OVERRIDE_THROTTLE_IDLE + 1),
                      "dwVioDriverOverrideThrottle enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioBrakePedalStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioBrakePedalStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_PEDAL_STATUS_UPSTOP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_PEDAL_STATUS_PSD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_PEDAL_STATUS_NDEF2));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_BRAKE_PEDAL_STATUS_NDEF2 - DW_VIO_BRAKE_PEDAL_STATUS_UPSTOP + 1),
                      "dwVioBrakePedalStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioBrakeStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioBrakeStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_STATUS_NO_BRAKING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_STATUS_BRAKING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_BRAKE_STATUS_UNKNOWN));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_BRAKE_STATUS_UNKNOWN - DW_VIO_BRAKE_STATUS_NO_BRAKING + 1),
                      "dwVioBrakeStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlDriverInterventionStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlDriverInterventionStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_NOHOWDETECTION),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINLOOP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINLOOP_TOUCH),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINLOOP_GRASP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINLOOP_DOUBLEGRABBED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVPARKINTERRUPT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVL3INTERRUPT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINERRHOSWD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_DRVINLOOP_DEGRADEDHOSWD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_NOHOWDETECTION_DEGRADEDHOSWD));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_NOHOWDETECTION_DEGRADEDHOSWD - DW_VIO_LAT_CTRL_DRIVER_INTERVENTION_STATUS_NOHOWDETECTION + 1),
                      "dwVioLatCtrlDriverInterventionStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlFaultStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlFaultStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_INIT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_HYDRAULIC_CONTROL_UNIT_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_EPB_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_LTI_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_COM_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_REDUNDANCY_CHECK_FAILED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_BRAKE_OVERHEATED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_BRAKE_DEGRADATION),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_SAFETY_LIMITS_VIOLATED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_ASU_SGNL_INPUT_PLAUSI_CHCK_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_REDUNDANCY_UNIT_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_DRIVER_ABSENT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_SSC_FLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_FAULT_STATUS_ESP_OR_ASR_CTRL_ACTV));

        static_assert(ENUMERATOR_COLLECTION.size() == 15U, "dwVioLongCtrlFaultStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlErrorStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlErrorStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_PARKERROR_ENGMNT_RQ_SPD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_PARKERROR_MAX_SPD_LIM_MODE_MAX_SPD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_PARK_ERROR_PT_OFF),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_PARK_ERROR_MISC),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L2ERROR_EPS_REV_GR),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L2ERROR_EPS_TMP_OFF_SSA),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L2ERROR_OTHER),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_PLAUSIERROR),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_COMMERROR),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L3_ERROR_1),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L3_ERROR_2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_TIMEOUTDTCTD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_ENGAGEMENTPREVENTIONCOND),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_SAFETYDRIVERCNDTNSFLT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_RAS_ERROR_DERATING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_RAS_ERROR_MD_MAN_CANCEL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_ABSM_ERROR_IFCPERMCLOSED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L2Error_EPS_TMP_OFF_REJECT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_ERROR_STATUS_L2Error_EPS_DERATE));

        static_assert(ENUMERATOR_COLLECTION.size() == 20U, "dwVioLatCtrlErrorStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioHeadlightState>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioHeadlightState;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_HEADLIGHT_STATE_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_HEADLIGHT_STATE_OFF),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_HEADLIGHT_STATE_LOW_BEAM),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_HEADLIGHT_STATE_HIGH_BEAM),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_HEADLIGHT_STATE_DRL));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_HEADLIGHT_STATE_DRL - DW_VIO_HEADLIGHT_STATE_UNKNOWN + 1),
                      "dwVioHeadlightState enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlStatus>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlStatus;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_STATUS_INIT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_STATUS_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_STATUS_CTRL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_STATUS_TERMINATED),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_STATUS_ERROR));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LAT_CTRL_STATUS_ERROR - DW_VIO_LAT_CTRL_STATUS_INIT + 1),
                      "dwVioLatCtrlStatus enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVehicleIOGear>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOGear;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_PARK),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_REVERSE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_NEUTRAL),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_DRIVE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_GEAR_LOW));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VEHICLEIO_GEAR_LOW - DW_VEHICLEIO_GEAR_UNKNOWN + 1), "dwVehicleIOGear enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLongCtrlDrivePositionCommand>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLongCtrlDrivePositionCommand;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_D),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_R),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_P));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_P - DW_VIO_LONG_CTRL_DRIVE_POSITION_COMMAND_IDLE + 1),
                      "dwVioLongCtrlDrivePositionCommand enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVehicleIOTurnSignal>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOTurnSignal;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_TURNSIGNAL_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_TURNSIGNAL_OFF),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_TURNSIGNAL_LEFT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_TURNSIGNAL_RIGHT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_TURNSIGNAL_EMERGENCY));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VEHICLEIO_TURNSIGNAL_EMERGENCY - DW_VEHICLEIO_TURNSIGNAL_UNKNOWN + 1), "dwVehicleIOTurnSignal enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVehicleIOOverrides>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOOverrides;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_BRAKE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_STEERING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_THROTTLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_GEAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_OVERRIDE_MAX));

        static_assert(ENUMERATOR_COLLECTION.size() == 6U, "dwVehicleIOOverrides enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVehicleIOFaults>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVehicleIOFaults;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_NONE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_BRAKE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_STEERING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_THROTTLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_GEAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_SAFETY),
            DW_DESCRIBE_C_ENUMERATOR(DW_VEHICLEIO_FAULT_MAX));

        static_assert(ENUMERATOR_COLLECTION.size() == 7U, "dwVehicleIOFaults enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

#pragma GCC diagnostic push // PARK_RECORD is deprecated
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <>
struct EnumDescription<dwVioLatCtrlModeRequest>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVioLatCtrlModeRequest;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_IDLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L2_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L2_HFE),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_AES),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L3_EMG),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_LSS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_PARK_L2),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_PARK_L3),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_PARK_RECORD),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_ESS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_ARP),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_L2_PLUS_PLUS),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_MODE_REQUEST_EESF));
        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_LAT_CTRL_MODE_REQUEST_EESF - DW_VIO_LAT_CTRL_MODE_REQUEST_IDLE + 1), "dwVioLatCtrlModeRequest enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};
#pragma GCC diagnostic pop

template <>
struct EnumDescription<dwVIOAEBState>
{
    static constexpr auto get()
    {
        using EnumT                          = dwVIOAEBState;
        constexpr auto ENUMERATOR_COLLECTION = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_AEB_STATE_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_AEB_STATE_OFF),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_AEB_STATE_READY));

        static_assert(ENUMERATOR_COLLECTION.size() == (DW_VIO_AEB_STATE_READY - DW_VIO_AEB_STATE_UNKNOWN + 1),
                      "dwVIOAEBState enumerators need update");
        return ENUMERATOR_COLLECTION;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlLoopStatus>
{
    static constexpr auto get()
    {
        using EnumT                         = dwVioLatCtrlLoopStatus;
        constexpr auto enumeratorCollection = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_LOOP_STATUS_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_LOOP_STATUS_OPEN),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_LOOP_STATUS_BLENDING),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_LOOP_STATUS_CLOSED));

        static_assert(enumeratorCollection.size() == (DW_VIO_LAT_CTRL_LOOP_STATUS_CLOSED - DW_VIO_LAT_CTRL_LOOP_STATUS_UNKNOWN + 1), "dwVioLatCtrlLoopStatus enumerators need update");
        return enumeratorCollection;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlASILStatus>
{
    static constexpr auto get()
    {
        using EnumT                         = dwVioLatCtrlASILStatus;
        constexpr auto enumeratorCollection = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_A_S_I_L_STATUS_QM),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_A_S_I_L_STATUS_ASIL_A),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_A_S_I_L_STATUS_ASIL_B),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_A_S_I_L_STATUS_ASIL_C),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_A_S_I_L_STATUS_ASIL_D));

        static_assert(enumeratorCollection.size() == (DW_VIO_LAT_CTRL_A_S_I_L_STATUS_ASIL_D - DW_VIO_LAT_CTRL_A_S_I_L_STATUS_QM + 1), "dwVioLatCtrlASILStatus enumerators need update");
        return enumeratorCollection;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlInterventionDirectionELK>
{
    static constexpr auto get()
    {
        using EnumT                         = dwVioLatCtrlInterventionDirectionELK;
        constexpr auto enumeratorCollection = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_INVALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_FRONT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_LEFT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_RIGHT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_REAR));

        static_assert(enumeratorCollection.size() == (DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_REAR - DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_INVALID + 1), "dwVioLatCtrlInterventionDirectionELK enumerators need update");
        return enumeratorCollection;
    }
};

template <>
struct EnumDescription<dwVioLatCtrlInterventionDirectionLKA>
{
    static constexpr auto get()
    {
        using EnumT                         = dwVioLatCtrlInterventionDirectionLKA;
        constexpr auto enumeratorCollection = describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_INVALID),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_FRONT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_LEFT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_RIGHT),
            DW_DESCRIBE_C_ENUMERATOR(DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_REAR));

        static_assert(enumeratorCollection.size() == (DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_REAR - DW_VIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_INVALID + 1), "dwVioLatCtrlInterventionDirectionLKA enumerators need update");
        return enumeratorCollection;
    }
};
} // framework
} // dw
#endif // DW_FRAMEWORK_VEHICLEIO_TYPES_HPP_
