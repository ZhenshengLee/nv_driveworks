////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY,
// OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY
// IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of
// such information or for any infringement of patents or other rights of third
// parties that may result from its use. No license is granted by implication or
// otherwise under any patent or patent rights of NVIDIA Corporation. No third
// party distribution is allowed unless expressly authorized by NVIDIA.  Details
// are subject to change without notice. This code supersedes and replaces all
// information previously supplied. NVIDIA Corporation products are not
// authorized for use as critical components in life support devices or systems
// without express written approval of NVIDIA Corporation.
//
// Copyright (c) 2022-2024 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef DW_SENSORS_RADAR_RADARSSI_H_
#define DW_SENSORS_RADAR_RADARSSI_H_
// Generated by dwProto from radar_ssi.proto DO NOT EDIT BY HAND!
// See //3rdparty/shared/dwproto/README.md for more information

/////////////////////////////////////////////////////////////////////////////////////////

#include <dw/core/base/Types.h>

#include <dw/sensors/radar/RadarTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Defines radar calibration base stat
typedef enum dwRadarCalibrationBaseStat {
    /// stat success
    DW_RADAR_CALIBRATION_BASE_STAT_SUCCESS = 0,

    /// stat fail
    DW_RADAR_CALIBRATION_BASE_STAT_FAIL = 1,

    /// stat no calibration
    DW_RADAR_CALIBRATION_BASE_STAT_NO_CALIB = 2,

    /// stat no base calibration performed
    DW_RADAR_CALIBRATION_BASE_STAT_NO_BASE_CALIB_PERFORMED = 3,

    /// stat count
    DW_RADAR_CALIBRATION_BASE_COUNT = 4,
} dwRadarCalibrationBaseStat;

/// @brief Radar SSI calibration func stat
typedef enum dwRadarCalibrationFuncStat {
    /// stat AIC success
    DW_RADAR_CALIBRATION_FUNC_STAT_AIC_SUCCESS = 0,

    /// stat AIC failed
    DW_RADAR_CALIBRATION_FUNC_STAT_AIC_FAIL = 1,

    /// stat AIC active
    DW_RADAR_CALIBRATION_FUNC_STAT_AIC_ACTIVE = 2,

    /// stat End Of line Calibration success
    DW_RADAR_CALIBRATION_FUNC_STAT_EOC_SUCCESS = 3,

    /// stat End Of line Calibration failed
    DW_RADAR_CALIBRATION_FUNC_STAT_EOC_FAIL = 4,

    /// stat End Of line Calibration active
    DW_RADAR_CALIBRATION_FUNC_STAT_EOC_ACTIVE = 5,

    /// stat Service Drive Calibration success
    DW_RADAR_CALIBRATION_FUNC_STAT_SDC_SUCCESS = 6,

    /// stat Service Drive Calibration failed
    DW_RADAR_CALIBRATION_FUNC_STAT_SDC_FAIL = 7,

    /// stat Service Drive Calibration active
    DW_RADAR_CALIBRATION_FUNC_STAT_SDC_ACTIVE = 8,

    /// stat no calibration
    DW_RADAR_CALIBRATION_FUNC_STAT_NO_CALIB = 9,

    /// stat count
    DW_RADAR_CALIBRATION_FUNC_STAT_NO_COUNT = 10,
} dwRadarCalibrationFuncStat;

/// @brief Radar SSI calibration SOC(Sustained Online Calibration) stat
typedef enum dwRadarCalibrationSOCStat {
    /// stat Calibration Sustained Online Calibration active
    DW_RADAR_CALIBRATION_SOC_STAT_ACTIVE = 0,

    /// stat Calibration Sustained Online Calibration failed
    DW_RADAR_CALIBRATION_SOC_STAT_FAIL = 1,

    /// stat Calibration Sustained Online Calibration paused
    DW_RADAR_CALIBRATION_SOC_STAT_PAUSE = 2,

    /// stat Calibration Sustained Online Calibration deactive
    DW_RADAR_CALIBRATION_SOC_STAT_DEACTIVE = 3,

    /// stat Calibration Sustained Online Calibration count
    DW_RADAR_CALIBRATION_SOC_STAT_COUNT = 4,
} dwRadarCalibrationSOCStat;

/// Defines radar Orientation stat
typedef enum dwRadarOrientationStat {
    /// stat Orientation not coded
    DW_RADAR_ORIENT_STAT_DEFAULT = 0,

    /// stat Orientation normal
    DW_RADAR_ORIENT_STAT_NORMAL = 1,

    /// stat Orientation rotated(upside-down)
    DW_RADAR_ORIENT_STAT_ROTATED = 2,

    /// stat Orientation count
    DW_RADAR_ORIENT_STAT_COUNT = 3,
} dwRadarOrientationStat;

/// Defines radar health overall state
typedef enum dwRadarHealthOprtnStat {
    /// overall state of sensor health normal
    DW_RADAR_HEALTH_OPRTN_STAT_NORMAL = 0,

    /// overall state of sensor health off
    DW_RADAR_HEALTH_OPRTN_STAT_OFF = 1,

    /// overall state of sensor health error
    DW_RADAR_HEALTH_OPRTN_STAT_ERROR = 2,

    /// overall state of sensor health count
    DW_RADAR_HEALTH_OPRTN_STAT_COUNT = 3,
} dwRadarHealthOprtnStat;

/// Defines radar health measuring state
typedef enum dwRadarHealthOprtnMd {
    /// measuring state of sensor active
    DW_RADAR_HEALTH_MD_ACTIVE = 0,

    /// measuring state of sensor disabled
    DW_RADAR_HEALTH_MD_DISABLED = 1,

    /// measuring state of sensor test mode
    DW_RADAR_HEALTH_MD_TEST_MODE = 2,

    /// measuring state of sensor count
    DW_RADAR_HEALTH_MD_COUNT = 3,
} dwRadarHealthOprtnMd;

/// Defines radar health defect state
typedef enum dwRadarHealthDefectDtct {
    /// Defect state of sensor full func
    DW_RADAR_HEALTH_DEFECT_DTCT_FULL_FUNC = 0,

    /// Defect state of sensor not full func
    DW_RADAR_HEALTH_DEFECT_DTCT_NOT_FULL_FUNC = 1,

    /// Defect state of sensor out of order
    DW_RADAR_HEALTH_DEFECT_DTCT_OUT_OF_ORDER = 2,

    /// Defect state of sensor count
    DW_RADAR_HEALTH_DEFECT_DTCT_COUNT = 3,
} dwRadarHealthDefectDtct;

/// Defines reasons of sensor defect
typedef enum dwRadarHealthDefectRSN {
    /// Reasons of sensor defect no defect
    DW_RADAR_HEALTH_DEFECT_RSN_NO_DEFECT = 0,

    /// Reasons of sensor defect wakeup line malfunc
    DW_RADAR_HEALTH_DEFECT_RSN_WAKEUP_LINE_MALFUNC = 1,

    /// Reasons of sensor defect ethernet malfunc
    DW_RADAR_HEALTH_DEFECT_RSN_ETHERNET_MALFUNC = 2,

    /// Reasons of sensor defect sensor idnetification issue
    DW_RADAR_HEALTH_DEFECT_RSN_SENSOR_IDENTIFICATION_ISSUE = 3,

    /// Reasons of sensor defect radar failure
    DW_RADAR_HEALTH_DEFECT_RSN_RADAR_FAILURE = 4,

    /// Reasons of sensor defect hardware failure
    DW_RADAR_HEALTH_DEFECT_RSN_HARDWARE_FAILURE = 5,

    /// Reasons of sensor defect software failure
    DW_RADAR_HEALTH_DEFECT_RSN_SOFTWARE_FAILURE = 6,

    /// Reasons of sensor defect input signal malfunc
    DW_RADAR_HEALTH_DEFECT_RSN_INPUT_SIGNAL_MALFUNC = 7,

    /// Reasons of sensor defect count
    DW_RADAR_HEALTH_DEFECT_RSN_COUNT = 8,
} dwRadarHealthDefectRSN;

/// Defines state of sensor diagnostic
typedef enum dwRadarHealthdiagMD {
    /// State of sensor diagnostic interface idle
    DW_RADAR_HEALTH_DIAG_MD_IDLE = 0,

    /// State of sensor diagnostic interface eol mode
    DW_RADAR_HEALTH_DIAG_MD_EOL = 1,

    /// State of sensor diagnostic interface diag mode
    DW_RADAR_HEALTH_DIAG_MD_DIAG = 2,

    /// State of sensor diagnostic interface extend diag mode
    DW_RADAR_HEALTH_DIAG_MD_EXTEND_DIAG = 3,

    /// State of sensor diagnostic count
    DW_RADAR_HEALTH_DIAG_MD_COUNT = 4,
} dwRadarHealthdiagMD;

/// Defines state of supply voltage
typedef enum dwRadarHealthSuppVoltStat {
    /// State of supply voltage within limits
    DW_RADAR_HEALTH_SUPP_VOLT_STAT_WITHIN_LIMITS = 0,

    /// State of supply voltage voltage low
    DW_RADAR_HEALTH_SUPP_VOLT_STAT_VOLTAGE_LOW = 1,

    /// State of supply voltage voltage high
    DW_RADAR_HEALTH_SUPP_VOLT_STAT_VOLTAGE_HIGH = 2,

    /// State of supply voltage count
    DW_RADAR_HEALTH_SUPP_VOLT_STAT_COUNT = 3,
} dwRadarHealthSuppVoltStat;

/// Defines state of internal sensor ECU temperature
typedef enum dwRadarHealthTempStat {
    /// State of internal sensor ECU temperature normal temp
    DW_RADAR_HEALTH_TEMP_STAT_NORMAL_TEMP = 0,

    /// State of internal sensor ECU temperature under temp
    DW_RADAR_HEALTH_TEMP_STAT_UNDER_TEMP = 1,

    /// State of internal sensor ECU temperature over temp
    DW_RADAR_HEALTH_TEMP_STAT_OVER_TEMP = 2,

    /// State of internal sensor ECU temperature count
    DW_RADAR_HEALTH_TEMP_STAT_COUNT = 3,
} dwRadarHealthTempStat;

/// Defines status of disturbance by electromagentic phenomenon
typedef enum dwRadarHealthExternDisturb {
    /// Status of disturbance by electromagentic phenomenon (interference) normal
    DW_RADAR_HEALTH_EXTERN_DISTURB_NORMAL = 0,

    /// Status of disturbance by electromagentic phenomenon (interference) disturbed
    DW_RADAR_HEALTH_EXTERN_DISTURB_DISTURBED = 1,

    /// Status of disturbance by electromagentic phenomenon (interference) count
    DW_RADAR_HEALTH_EXTERN_DISTURB_COUNT = 2,
} dwRadarHealthExternDisturb;

/// Defines status of RF transmission
typedef enum dwRadarHealthTxPwrRedctnStat {
    /// Status of RF transmission normal
    DW_RADAR_HEALTH_TX_PWR_REDCTN_STAT_NORMAL = 0,

    /// Status of RF transmission pwr limited
    DW_RADAR_HEALTH_TX_PWR_REDCTN_STAT_PWR_LIMITED = 1,

    /// Status of RF transmission count
    DW_RADAR_HEALTH_TX_PWR_REDCTN_STAT_COUNT = 2,
} dwRadarHealthTxPwrRedctnStat;

/// Defines status of radom heating driver heating
typedef enum dwRadarHealthRadomHeatStat {
    /// Status of radom heating driver heating off
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_OFF = 0,

    /// Status of radom heating driver heating LVL1
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_LVL1 = 1,

    /// Status of radom heating driver heating LVL2
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_LVL2 = 2,

    /// Status of radom heating driver heating not enabled
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_NOT_ENABLED = 3,

    /// Status of radom heating driver heating RXINV
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_RXINV = 4,

    /// Status of radom heating driver heating fault
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_FAULT = 5,

    /// Status of radom heating driver heating control fault
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_CONTROL_FAULT = 6,

    /// Status of radom heating driver heating count
    DW_RADAR_HEALTH_RADOM_HEAT_STAT_HEATING_COUNT = 7,
} dwRadarHealthRadomHeatStat;

/// Defines status of time sync
typedef enum dwRadarHealthTmSyncStat {
    /// Status no error
    DW_RADAR_HEALTH_TM_SYNC_STAT_NO_ERROR = 0,

    /// Status of time sync timeout
    DW_RADAR_HEALTH_TM_SYNC_STAT_TIMEOUT = 1 << 0,

    /// Status of time sync sync to gateway
    DW_RADAR_HEALTH_TM_SYNC_STAT_SYNC_TO_GATEWAY = 1 << 2,

    /// Status of time sync global time base
    DW_RADAR_HEALTH_TM_SYNC_STAT_GLOBAL_TIME_BASE = 1 << 3,

    /// Status of time sync timeleap future
    DW_RADAR_HEALTH_TM_SYNC_STAT_TIMELEAP_FUTURE = 1 << 4,

    /// Status of time sync timeleap past
    DW_RADAR_HEALTH_TM_SYNC_STAT_TIMELEAP_PAST = 1 << 5,

    /// Status of time sync local time base
    DW_RADAR_HEALTH_TM_SYNC_STAT_LOCAL_TIME_BASE = 1 << 6,

    /// Status of time sync invalid
    DW_RADAR_HEALTH_TM_SYNC_STAT_INVALID = 1 << 7,
} dwRadarHealthTmSyncStat;

/// Defines whether replicates the security master synchronisation status sync
typedef enum dwRadarHealthVSMSyncStat {
    /// Replicates the security master synchronisation status not sync
    DW_RADAR_HEALTH_VSM_SYNC_STAT_NOT_SYNC = 0,

    /// Replicates the security master synchronisation status sync
    DW_RADAR_HEALTH_VSM_SYNC_STAT_SYNC = 1,

    /// Replicates the security master synchronisation status count
    DW_RADAR_HEALTH_VSM_SYNC_STAT_COUNT = 2,
} dwRadarHealthVSMSyncStat;

/// Defines state of rangegate management ranggate
typedef enum dwRadarHealthRangeGateStat {
    /// state of rangegate management ranggate fixed
    DW_RADAR_HEALTH_RANGE_GATE_STAT_FIXED = 0,

    /// state of rangegate management ranggate adaptive
    DW_RADAR_HEALTH_RANGE_GATE_STAT_ADAPTIVE = 1,

    /// state of rangegate management ranggate count
    DW_RADAR_HEALTH_RANGE_GATE_STAT_COUNT = 2,
} dwRadarHealthRangeGateStat;

/// Defines FuSa Error detected
typedef enum dwRadarHealthFuSaStat {
    /// FuSa Error detected default
    DW_RADAR_HEALTH_FUSA_STAT_DEFAULT = 0,

    /// FuSa Error detected norm
    DW_RADAR_HEALTH_FUSA_STAT_NORM = 1,

    /// FuSa Error detected err
    DW_RADAR_HEALTH_FUSA_STAT_ERR = 2,

    /// FuSa Error detected count
    DW_RADAR_HEALTH_FUSA_STAT_COUNT = 3,
} dwRadarHealthFuSaStat;

/// Defines park mode status
typedef enum dwRadarHealthParkModeStat {
    /// Park mode stat deactive
    DW_RADAR_HEALTH_PARK_MODE_STAT_DEACTIVE = 0,

    /// Park mode stat active
    DW_RADAR_HEALTH_PARK_MODE_STAT_ACTIVE = 1,

    /// Park mode stat count
    DW_RADAR_HEALTH_PARK_MODE_STAT_COUNT = 2,
} dwRadarHealthParkModeStat;

/// Defines radar fault status
typedef enum dwRadarFaultStatus {
    /// radar fault status operational
    DW_RADAR_FAULT_STATUS_OPERATIONAL = 0,

    /// radar fault status initialization
    DW_RADAR_FAULT_STATUS_INITIALIZATION = 1,

    /// radar fault status hardware error
    DW_RADAR_FAULT_STATUS_HW_ERROR = 2,

    /// radar fault status DSP error
    DW_RADAR_FAULT_STATUS_DSP_ERROR = 3,

    /// radar fault status communication error
    DW_RADAR_FAULT_STATUS_COMMUNICATION_ERROR = 4,

    /// radar fault status temporary error
    DW_RADAR_FAULT_STATUS_TEMPORARY_ERROR = 5,

    /// radar fault status other
    DW_RADAR_FAULT_STATUS_OTHERS = 6,

    /// radar fault status disabled
    DW_RADAR_FAULT_STATUS_RADAR_DISABLED = 7,
} dwRadarFaultStatus;

/// Defines radar blind status
typedef enum dwRadarBlindStatus {
    /// radar blind status operational
    DW_RADAR_BLIND_STATUS_OPERATIONAL = 0,

    /// radar blind status temporary blindness
    DW_RADAR_BLIND_STATUS_TEMPORARY_BLINDNESS = 1,

    /// radar blind status permanant blindness
    DW_RADAR_BLIND_STATUS_PERMANANT_BLINDNESS = 2,

    /// radar blind status error
    DW_RADAR_BLIND_STATUS_ERROR = 3,
} dwRadarBlindStatus;

/// Defines radar windows action request
typedef enum dwRadarWindowACTRequest {
    /// radar window action no request
    DW_RADAR_WINDOW_ACT_REQ_NO_REQUSTE = 0,

    /// radar window action open all
    DW_RADAR_WINDOW_ACT_REQ_OPEN_ALL = 1,
} dwRadarWindowACTRequest;

/// Defines blockage detected by radar
typedef enum dwRadarHealthBlockageStat {
    /// Health blockage information no blockage
    DW_RADAR_HEALTH_BLOCKAGE_STAT_NO_BLOCKAGE = 0,

    /// Health blockage information full blockage
    DW_RADAR_HEALTH_BLOCKAGE_STAT_FULL_BLOCKAGE = 1,

    /// Health blockage information part blockage high
    DW_RADAR_HEALTH_BLOCKAGE_STAT_PART_BLOCKAGE_HIGH = 2,

    /// Health blockage information part blockage medium
    DW_RADAR_HEALTH_BLOCKAGE_STAT_PART_BLOCKAGE_MEDIUM = 3,

    /// Health blockage information part blockage low
    DW_RADAR_HEALTH_BLOCKAGE_STAT_PART_BLOCKAGE_LOW = 4,

    /// Health blockage information part blockage defect
    DW_RADAR_HEALTH_BLOCKAGE_STAT_BLOCKAGE_DEFECT = 5,

    /// Health blockage information part blockage visibiity check after startup active
    DW_RADAR_HEALTH_BLOCKAGE_STAT_BLOCKAGE_CHECK_ACTIVE = 6,

    /// Health blockage information part blockage count
    DW_RADAR_HEALTH_BLOCKAGE_STAT_COUNT = 7,
} dwRadarHealthBlockageStat;

// Define segment blockage information
typedef enum dwRadarPerfBlockageStat {
    /// Segment blockage information no blockage
    DW_RADAR_PERF_BLOCKAGE_STAT_NO_BLOCKAGE = 0,

    /// Segment blockage information full blockage
    DW_RADAR_PERF_BLOCKAGE_STAT_FULL_BLOCKAGE = 1,

    /// Segment blockage information part blockage high
    DW_RADAR_PERF_BLOCKAGE_STAT_PART_BLOCKAGE_HIGH = 2,

    /// Segment blockage information part blockage medium
    DW_RADAR_PERF_BLOCKAGE_STAT_PART_BLOCKAGE_MEDIUM = 3,

    /// Segment blockage information part blockage low
    DW_RADAR_PERF_BLOCKAGE_STAT_PART_BLOCKAGE_LOW = 4,

    /// Segment blockage information part blockage low
    DW_RADAR_PERF_BLOCKAGE_STAT_PART_BLOCKAGE_DEFECT = 5,

    /// Segment blockage information part blockage count
    DW_RADAR_PERF_BLOCKAGE_STAT_PART_BLOCKAGE_COUNT = 6,
} dwRadarPerfBlockageStat;

// Define segment reason
typedef enum dwRadarPerfRedctnRsn {
    /// Segment reason for blockage unknown
    DW_RADAR_PERF_REDCTN_RSN_UNKNOWN = 0,

    /// Segment reason for blockage absorptive
    DW_RADAR_PERF_REDCTN_RSN_ABSORPTIVE = 1,

    /// Segment reason for blockage distortive
    DW_RADAR_PERF_REDCTN_RSN_DISTORTIVE = 2,

    /// Segment reason for blockage interference
    DW_RADAR_PERF_REDCTN_RSN_INTERFERENCE = 3,

    /// Segment reason for blockage snow on fascia
    DW_RADAR_PERF_REDCTN_RSN_SNOW_ON_FASCIA = 4,

    /// Segment reason for blockage water on fascia
    DW_RADAR_PERF_REDCTN_RSN_WATER_ON_FASCIA = 5,

    /// Segment reason for blockage soll on fascia
    DW_RADAR_PERF_REDCTN_RSN_SOIL_ON_FASCIA = 6,

    /// Segment reason for blockage scratch on fascia
    DW_RADAR_PERF_REDCTN_RSN_SCRATCH_ON_FASCIA = 7,

    /// Segment reason for blockage count
    DW_RADAR_PERF_REDCTN_RSN_COUNT = 8,
} dwRadarPerfRedctnRsn;

/// Defines the bitmasks of errors detected by diagnostic function
typedef enum dwRadarHealthError {
    /// Has unavailable data in the scan
    DW_RADAR_HEALTH_ERROR_SNA = 1 << 0,

    /// Has frame drop
    DW_RADAR_HEALTH_ERROR_FRAME_DROP = 1 << 1,

    /// CRC error of sensor data output
    DW_RADAR_HEALTH_ERROR_SODA_CRC = 1 << 2,

    /// CRC error of sensor health info output
    DW_RADAR_HEALTH_ERROR_SSI_CRC = 1 << 3,

    /// Has frame overrrun
    DW_RADAR_HEALTH_ERROR_FRAME_OVERRRUN = 1 << 4,

    /// No data have been received from the publisher at all
    DW_RADAR_HEALTH_ERROR_SPLITPDU_NO_DATA_FROM_PUBLISHER = 1 << 5,

    /// Not enough data where the E2E check yielded OK from the publisher is
    /// available since the initialization, sample(s) cannot be used.
    DW_RADAR_HEALTH_ERROR_SPLITPDU_NOT_ENOUGH_DATA = 1 << 6,

    /// Too few data where the E2E check yielded OK or too many data
    /// where the e2e check yielded ERROR were received within the E2E time
    /// window – communication of the sample of this event not functioning
    /// properly, sample(s) cannot be used.
    DW_RADAR_HEALTH_ERROR_SPLITPDU_E2E_FAILURE = 1 << 7,

    /// No E2E state machine available. Return value of function GetSMState if
    /// EndToEndTransformationComSpecProps.disableEndToEndStateMachine is set
    /// to TRUE (not available yet in AMSR implementation)
    DW_RADAR_HEALTH_ERROR_SPLITPDU_STATE_MACHINE_DISABLED = 1 << 8,

    /// Has azimuth out of range
    DW_RADAR_HEALTH_ERROR_AZIMUTH_OUT_OF_RANGE = 1 << 9,

    /// Has return out of range
    DW_RADAR_HEALTH_ERROR_NUM_RETURNS_OUT_OF_RANGE = 1 << 10,

    /// Has zero detections
    DW_RADAR_HEALTH_ERROR_ZERO_RETURNS = 1 << 11,
} dwRadarHealthError;

/// Defines radar Perf Segment validity
typedef struct dwRadarPerfSegmentValidity
{
    /// validity info for dwRadarHealthInfo.ID
    bool IDValidity;

    /// validity info for dwRadarHealthInfo.aziAnglBegin
    bool aziAnglBeginValidity;

    /// validity info for dwRadarHealthInfo.aziAnglEnd
    bool aziAnglEndValidity;

    /// validity info for dwRadarHealthInfo.rangeGain
    bool rangeGainValidity;

    /// validity info for dwRadarHealthInfo.rangeGainRCS
    bool rangeGainRCSValidity;

    /// validity info for dwRadarHealthInfo.detctnRangeMin
    bool detctnRangeMinValidity;

    /// validity info for dwRadarHealthInfo.detctnRangeMax
    bool detctnRangeMaxValidity;

    /// validity info for dwRadarHealthInfo.blockageStat
    bool blockageStatValidity;

    /// validity info for dwRadarHealthInfo.redctnRsn
    bool redctnRsnValidity;

    /// validity info for dwRadarHealthInfo.percElems
    bool percElemsValidity[8];

    /// validity info for dwRadarHealthInfo.snrRefTarget
    bool snrRefTargetValidity;
} dwRadarPerfSegmentValidity;

/// Defines radar Perf Info validity
typedef struct dwRadarPerfInfoValidity
{
    /// validity info for dwRadarPerfInfo.orientRoll
    bool orientRollValidity;

    /// validity info for dwRadarPerfInfo.cosOffsetX
    bool cosOffsetXValidity;

    /// validity info for dwRadarPerfInfo.cosOffsetY
    bool cosOffsetYValidity;

    /// validity info for dwRadarPerfInfo.cosOffsetZ
    bool cosOffsetZValidity;

    /// validity info for dwRadarPerfInfo.validSegAzi
    bool validSegAziValidity;

    /// validity info for dwRadarPerfInfo.validSegElev
    bool validSegElevValidity;

    /// validity info for dwRadarPerfInfo.totalSegAzi
    bool totalSegAziValidity;

    /// validity info for dwRadarPerfInfo.totalSegElev
    bool totalSegElevValidity;

    /// validity info for dwRadarPerfInfo.size
    bool sizeValidity;
    dwRadarPerfSegmentValidity perfSegmentsValidity[32];
} dwRadarPerfInfoValidity;

/// Defines radar Health Info validity
typedef struct dwRadarHealthInfoValidity
{
    /// validity info for dwRadarHealthInfo.oprtnStat
    bool oprtnStatValidity;

    /// validity info for dwRadarHealthInfo.oprtnMd
    bool oprtnMdValidity;

    /// validity info for dwRadarHealthInfo.defectDtct
    bool defectDtctValidity;

    /// validity info for dwRadarHealthInfo.defectRsn
    bool defectRsnValidity;

    /// validity info for dwRadarHealthInfo.diagMd
    bool diagMdValidity;

    /// validity info for dwRadarHealthInfo.suppVoltStat
    bool suppVoltStatValidity;

    /// validity info for dwRadarHealthInfo.tempStat
    bool tempStatValidity;

    /// validity info for dwRadarHealthInfo.validInputSignalsCurr
    bool validInputSignalsCurrValidity;

    /// validity info for dwRadarHealthInfo.validInputSignalsTrgt
    bool validInputSignalsTrgtValidity;

    /// validity info for dwRadarHealthInfo.ExternDisturb
    bool externDisturbValidity;

    /// validity info for dwRadarHealthInfo.txPwrRedctnStat
    bool txPwrRedctnStatValidity;

    /// validity info for dwRadarHealthInfo.radomHeatStat
    bool radomHeatStatValidity;

    /// validity info for dwRadarHealthInfo.tmSyncStat
    bool tmSyncStatValidity;

    /// validity info for dwRadarHealthInfo.vsmSyncStat
    bool vsmSyncStatValidity;

    /// validity info for dwRadarHealthInfo.rangeGateStat
    bool rangeGateStatValidity;

    /// validity info for dwRadarHealthInfo.fuSaStat
    bool fuSaStatValidity;

    /// validity info for dwRadarHealthInfo.parkModeStat
    bool parkModeStatValidity;

    /// validity info for dwRadarHealthInfo.faultStatus
    bool faultStatusValidity;

    /// validity info for dwRadarHealthInfo.blindStatus
    bool blindStatusValidity;

    /// validity info for dwRadarHealthInfo.windowACTRequest
    bool windowACTRequestValidity;

    /// validity info for dwRadarHealthInfo.blockageStat
    bool blockageStatValidity;
} dwRadarHealthInfoValidity;

/// Defines radar Calibration Info validity
typedef struct dwRadarCalibrationInfoValidity
{
    /// validity info for dwRadarCalibrationInfo.baseStat
    bool baseStatValidity;

    /// validity info for dwRadarCalibrationInfo.funcStat
    bool funcStatValidity;

    /// validity info for dwRadarCalibrationInfo.socStat
    bool socStatValidity;

    /// validity info for dwRadarCalibrationInfo.orientationStat
    bool orientationStatValidity;

    /// validity info for dwRadarCalibrationInfo.orientAzi
    bool orientAziValidity;

    /// validity info for dwRadarCalibrationInfo.orientElev
    bool orientElevValidity;

    /// validity info for dwRadarCalibrationInfo.orientErrAzi
    bool orientErrAziValidity;

    /// validity info for dwRadarCalibrationInfo.orientErrElev
    bool orientErrElevValidity;

    /// validity info for dwRadarCalibrationInfo.originPosX
    bool originPosXValidity;

    /// validity info for dwRadarCalibrationInfo.originPosY
    bool originPosYValidity;

    /// validity info for dwRadarCalibrationInfo.originPosZ
    bool originPosZValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnLowLmtAzi
    bool corrctnLowLmtAziValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnLowLmtElev
    bool corrctnLowLmtElevValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnUpLmtAzi
    bool corrctnUpLmtAziValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnUpLmtElev
    bool corrctnUpLmtElevValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnOrientAzi
    bool corrctnOrientAziValidity;

    /// validity info for dwRadarCalibrationInfo.corrctnOrientElev
    bool corrctnOrientElevValidity;

    /// validity info for dwRadarCalibrationInfo.progressPercent
    bool progressPercentValidity;
} dwRadarCalibrationInfoValidity;

/// Defines radar SSI validity
typedef struct dwRadarSSIValidity
{
    /// validity info for dwRadarCalibrationInfo
    dwRadarCalibrationInfoValidity calibrationInfoValidity;

    /// validity info for dwRadarHealthInfoValidity
    dwRadarHealthInfoValidity healthInfoValidity;

    /// validity info for dwRadarPerfInfoValidity
    dwRadarPerfInfoValidity perfInfoValidity;
} dwRadarSSIValidity;

/// Defines radar calibration Info
typedef struct dwRadarCalibrationInfo
{
    /// Radar calibration base stat
    dwRadarCalibrationBaseStat baseStat;

    /// Radar calibration func stat
    dwRadarCalibrationFuncStat funcStat;

    /// Radar calibration soc stat
    dwRadarCalibrationSOCStat socStat;

    /// Radar calibration orientation stat
    dwRadarOrientationStat orientationStat;

    /// Sensor mounting orientation in azimuth Rad, nominal value
    float32_t orientAzi;

    /// Sensor mounting orientation in elevation Rad, nominal value
    float32_t orientElev;

    /// Sensor mounting orientation error in azimuth Rad, statistical calibration measurement accuracy
    float32_t orientErrAzi;

    /// Sensor mounting orientation error in elevation Rad, statistical calibration measurement accuracy
    float32_t orientErrElev;

    /// Sensor mounting position in coordinate system defined by radar supplier x mm
    int16_t originPosX;

    /// Sensor mounting position in coordinate system is defined by radar supplier y mm
    int16_t originPosY;

    /// Sensor mounting position in coordinate system is defined by radar supplier z mm
    int16_t originPosZ;

    /// Lower  limit of correction range in azimuth Rad
    float32_t corrctnLowLmtAzi;

    /// Lower  limit of correction range in elevation Rad
    float32_t corrctnLowLmtElev;

    /// Upper limit of correction range in azimuth Rad
    float32_t corrctnUpLmtAzi;

    /// Upper limit of correction range in elevation Rad
    float32_t corrctnUpLmtElev;

    /// the calculated correction angle deviation of the mechanical sensor alignment/nominal process in azimuth Rad
    float32_t corrctnOrientAzi;

    /// the calculated correction angle deviation of the mechanical sensor alignment/nominal process in elevation Rad
    float32_t corrctnOrientElev;

    /// Calibration progress for SDC in percent
    uint8_t progressPercent;
} dwRadarCalibrationInfo;

// Define struct for sensor health info
typedef struct dwRadarHealthInfo
{
    /// Overall state of sensor health
    dwRadarHealthOprtnStat oprtnStat;

    /// Measuring state of sensor
    dwRadarHealthOprtnMd oprtnMd;

    /// Defect state of sensor
    dwRadarHealthDefectDtct defectDtct;

    /// Reasons of sensor defect
    dwRadarHealthDefectRSN defectRsn;

    /// State of sensor diagnostic interface
    dwRadarHealthdiagMD diagMd;

    /// State of supply voltage
    dwRadarHealthSuppVoltStat suppVoltStat;

    /// State of internal sensor ECU temperature
    dwRadarHealthTempStat tempStat;

    /// Number of currently valid input signals
    // None
    uint8_t validInputSignalsCurr;

    /// Number of target  valid input signals
    // None
    uint8_t validInputSignalsTrgt;

    /// Status of disturbance by electromagentic phenomenon (interference)
    dwRadarHealthExternDisturb externDisturb;

    /// Status of RF transmission
    dwRadarHealthTxPwrRedctnStat txPwrRedctnStat;

    /// Status of radom heating driver
    dwRadarHealthRadomHeatStat radomHeatStat;

    /// Status of time sync
    dwRadarHealthTmSyncStat tmSyncStat;

    /// Replicates the security master synchronisation status
    dwRadarHealthVSMSyncStat vsmSyncStat;

    /// state of rangegate management
    dwRadarHealthRangeGateStat rangeGateStat;

    /// FuSa Error detected
    dwRadarHealthFuSaStat fuSaStat;

    /// Park mode stat
    dwRadarHealthParkModeStat parkModeStat;

    /// Radar fault status
    dwRadarFaultStatus faultStatus;

    /// Radar blind status
    dwRadarBlindStatus blindStatus;

    /// Window action request
    dwRadarWindowACTRequest windowACTRequest;

    /// Radar blockage stat
    dwRadarHealthBlockageStat blockageStat;
} dwRadarHealthInfo;

/// @brief Define radar perf segment.
typedef struct dwRadarPerfSegment
{
    /// Unique segment ID NoUnit
    /// Invalid:SNA:[63]
    uint8_t ID;

    /// Begin of segment in azimuth rad
    /// Invalid:SNA:[1023]
    float32_t aziAnglBegin;

    /// End of segment in azimuth rad
    /// Invalid:SNA:[1023]
    float32_t aziAnglEnd;

    /// Range gain for segment based on reference target percent
    /// Invalid:SNA:[255]
    uint8_t rangeGain;

    /// Reference RCS for reference target dB
    /// Invalid:SNA:[1023]
    int16_t rangeGainRCS;

    /// Minimum real detection range based on reference target m
    /// Invalid:SNA:[1023]
    float32_t detctnRangeMin;

    /// Maximum real detection range based on reference target m
    /// Invalid:SNA:[1023]
    float32_t detctnRangeMax;

    /// Segment blockage information
    dwRadarPerfBlockageStat blockageStat;

    /// Segment reason for blockage
    dwRadarPerfRedctnRsn redctnRsn;

    /// Segment probability for reason for blockage
    /// ARRAY-SIZE-SEMANTICS : FIXED-SIZE
    uint8_t percElems[8];

    /// The reference targets signal to noise ratio dB
    /// Invalid:None
    float32_t snrRefTarget;
} dwRadarPerfSegment;

/// Define struct for sensor performance info
typedef struct dwRadarPerfInfo
{
    /// Sensor mounting orientation in roll
    /// Valid:SENSORIENT_DEFAULT:[0],SENSORIENT_NORM:[1],SENSORIENT_ROTATED:[2],SNA:[3]
    uint8_t orientRoll;

    /// Sensor coordinate system offset between mounting and antenna plane in x m
    /// Invalid:NACPT:[254],SNA:[255]
    float32_t cosOffsetX;

    /// Sensor coordinate system offset between mounting and antenna plane in y m
    /// Invalid:NACPT:[254],SNA:[255]
    float32_t cosOffsetY;

    /// Sensor coordinate system offset between mounting and antenna plane in z m
    /// Invalid:NACPT:[254],SNA:[255]
    float32_t cosOffsetZ;

    /// Valid number of segments in azimuth NoUnit
    /// Invalid:SNA:[63]
    uint8_t validSegAzi;

    /// Valid number of segments in elevation NoUnit
    /// Invalid:SNA:[63]
    uint8_t validSegElev;

    /// Total (valid + invalid) number of segments in azimuth NoUnit
    /// Invalid:SNA:[63]
    uint8_t totalSegAzi;

    /// Total (valid + invalid) number of segments in elevation NoUnit
    /// Invalid:SNA:[63]
    uint8_t totalSegElev;

    /// size of perfSegments
    uint32_t size;

    /// Array of dwRadarPerfSegment
    dwRadarPerfSegment perfSegments[32];
} dwRadarPerfInfo;

/// sensor supplement data structure, include calibration info, performance, blockage, health signal
typedef struct dwRadarSSI
{
    /// timestamp when the packet received
    dwTime_t timestamp;

    /// validity info for calibInfo, healthInfo and perfInfo
    dwRadarSSIValidity validityInfo;

    /// calibration info for such as calibration status, positions
    dwRadarCalibrationInfo calibInfo;

    /// health info
    dwRadarHealthInfo healthInfo;

    /// performance info
    dwRadarPerfInfo perfInfo;

    /// Bitmask of Health errors
    dwRadarHealthError healthError;
} dwRadarSSI;

// roadcast debug struct with sensor header info and radar SSI
typedef struct dwRadarSSIDebugInfo
{
    /// Sensor-provided scan index
    uint32_t scanIndex;

    /// Sensor timestamp at which the current measurement scan was started (us). Same time domain as hostTimestamp
    dwTime_t sensorTimestamp;

    /// Host timestamp at reception of first packet belonging to this scan (us)
    dwTime_t hostTimestamp;

    /// Type of scan
    dwRadarScanType scanType;

    /// Number of radar returns in this scan
    uint32_t numReturns;

    /// Attached RadarSSI message
    dwRadarSSI RadarSSI;
} dwRadarSSIDebugInfo;

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_RADAR_RADARSSI_H_