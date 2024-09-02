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
// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_EGOMOTION_UTILS_IMUBIASESTIMATORPARAMETERS_H_
#define DW_EGOMOTION_UTILS_IMUBIASESTIMATORPARAMETERS_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief IMU accelerometer bias estimator parameters.
 */
typedef struct dwEgomotionIMUAccBiasEstimatorParameters
{
    //! Maximum possible bias as specified by manufacturer in m/s^2.
    float32_t biasRange;

    //! Maximum possible bias drift speed as specified by manufacturer in m/s^2/s.
    float32_t maxDriftSpeed;

    //! Maximum possible bias drift span over a 3 minute period as specified by manufacturer in m/s^2.
    float32_t maxDriftSpan;
} dwEgomotionIMUAccBiasEstimatorParameters;

/**
 * @brief IMU gyroscope bias estimator parameters.
 */
typedef struct dwEgomotionIMUGyroBiasEstimatorParameters
{
    //! Maximum possible bias as specified by manufacturer in rad/s.
    float32_t biasRange;

    //! Maximum possible bias drift speed as specified by manufacturer in rad/s/s.
    float32_t maxDriftSpeed;

    //! Maximum possible bias drift span over a 3 minute period as specified by manufacturer in rad/s.
    float32_t maxDriftSpan;
} dwEgomotionIMUGyroBiasEstimatorParameters;

#ifdef __cplusplus
}
#endif

#endif // DW_EGOMOTION_UTILS_IMUBIASESTIMATORPARAMETERS_H_
