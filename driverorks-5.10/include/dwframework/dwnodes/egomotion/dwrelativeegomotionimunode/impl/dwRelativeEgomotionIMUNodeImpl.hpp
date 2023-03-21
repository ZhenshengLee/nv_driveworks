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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_IMPL_DWRELATIVEEGOMOTIONIMUNODEIMPL_HPP_
#define DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_IMPL_DWRELATIVEEGOMOTIONIMUNODEIMPL_HPP_

#include <dwcgf/node/SimpleNodeT.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/egomotion/dwrelativeegomotionimunode/dwRelativeEgomotionIMUNode.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/egomotion/EgomotionState.h>
#include <dw/rig/Rig.hpp>

namespace dw
{
namespace framework
{

class dwRelativeEgomotionIMUNodeImpl : public SimpleProcessNodeT<dwRelativeEgomotionIMUNode>, public IAsyncResetable, public IContainsPreShutdownAction
{
public:
    static constexpr char LOG_TAG[] = "dwRelativeEgomotionIMUNode";

    dwRelativeEgomotionIMUNodeImpl(const dwRelativeEgomotionIMUNodeInitParams& params,
                                   const dwContextHandle_t ctx);
    dwRelativeEgomotionIMUNodeImpl(const dwContextHandle_t ctx)
        : m_ctx(ctx){};

    ~dwRelativeEgomotionIMUNodeImpl() override;

    dwStatus reset() override;

    dwStatus setAsyncReset() override;
    dwStatus executeAsyncReset() override;
    dwStatus preShutdown() override;

    dwStatus setupImpl() override;
    dwStatus validate() override;

    dwStatus addIMU(ManagedPortInput<dwIMUFrame>& imuFramePort);
    dwStatus addVehicleState(ManagedPortInput<dwVehicleIOSafetyState>& vehicleSafetyStatePort,
                             ManagedPortInput<dwVehicleIONonSafetyState>& vehicleNonSafetyStatePort,
                             ManagedPortInput<dwVehicleIOActuationFeedback>& vehicleActuationFeedbackPort);
    dwStatus updateIMUExtrinsics(ManagedPortInput<dwSensorNodeProperties>& imuExtrinsicPort);
    dwStatus updateWheelRadii(ManagedPortInput<dwCalibratedWheelRadii>& wheelRadiiPort);

    dwStatus sendState(ManagedPortOutput<dwEgomotionStateHandle_t>& egomotionStatePort,
                       ManagedPortOutput<dwEgomotionResultPayload>& resultPayloadPort,
                       ManagedPortOutput<dwTransformation3fPayload>& transPayloadPort,
                       ManagedPortOutput<dwEgomotionPosePayload>& posePayloadPort,
                       ManagedPortOutput<dwCalibratedIMUIntrinsics>& IMUIntrinsicsPort);

    dwStatus sendState(ManagedPortOutput<dwEgomotionStateHandle_t>& egomotionStatePort,
                       ManagedPortOutput<dwEgomotionResultPayload>& resultPayloadPort,
                       ManagedPortOutput<dwTransformation3fPayload>& transPayloadPort,
                       ManagedPortOutput<dwEgomotionPosePayload>& posePayloadPort);

    inline const dwEgomotionParameters& getEgomotionParameters() { return m_egomotionParams; };
    inline dwEgomotionHandle_t& getEgomotionHandle() { return m_egomotion; };
    dwStatus sendGyroBias(ManagedPortOutput<dwCalibratedIMUIntrinsics>& IMUIntrinsicsPort);

    void initializeParams(const dwRelativeEgomotionIMUNodeInitParams& params);

protected:
    void sendRoadCast(ManagedPortOutput<dwEgomotionResultPayload>& resultPayloadPort, ManagedPortOutput<dwTransformation3fPayload>& transPayloadPort, ManagedPortOutput<dwEgomotionPosePayload>& posePayloadPort);
    void sendModuleHandle();

    void monitorSignals(dwTime_t const currentHostTime, dwTime_t const latestEgomotionTimestamp) const;

    // Vehicle copy
    dwVehicle m_vehicle{};

    // Time bookkeeping
    dwTime_t m_lastUpdate{};

    dwEgomotionHandle_t m_egomotion = DW_NULL_HANDLE;
    const dwContextHandle_t m_ctx   = DW_NULL_HANDLE;

    dwEgomotionParameters m_egomotionParams{};
    void** m_outputHandle{nullptr};

    uint32_t m_imuSensorID = 0U;

    static const dwTime_t INITIALIZATION_TIMEOUT = 5'000'000; // [us]
    static const dwTime_t MISSING_SIGNAL_TIMEOUT = 50'000;    // [us]
    static const uint64_t MINIMUM_INPUT_COUNT    = 100;       // [-]

    dwTime_t m_latestVioTimestamp = DW_TIME_INVALID;
    dwTime_t m_latestIMUTimestamp = DW_TIME_INVALID;
    dwTime_t m_initTimestamp      = DW_TIME_INVALID;
    mutable bool m_isDrained      = false; // marked mutable for usage in logging functions otherwise const

    uint64_t m_vioInputCounter = 0U;
    uint64_t m_imuInputCounter = 0U;

private:
    void initializePorts(const dwRelativeEgomotionIMUNodeInitParams& params);

    void monitorVIOTimeOffset(const dwTime_t& currentHostTime) const;
    void monitorIMUTimeOffset(const dwTime_t& currentHostTime) const;
    void monitorEgomotionTimeOffset(const dwTime_t& currentHostTime, const dwTime_t& latestEgomotionTimestamp) const;

    dwStatus processAddIMU();
    dwStatus processAddVehicleState();
    dwStatus processUpdateIMUExtrinsics();
    dwStatus processUpdateWheelRadii();
    dwStatus processSendState();

    bool m_shutdown = false;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_IMPL_DWRELATIVEEGOMOTIONIMUNODEIMPL_HPP_
