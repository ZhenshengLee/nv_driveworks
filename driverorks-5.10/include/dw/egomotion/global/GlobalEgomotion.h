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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Global Egomotion Methods</b>
 *
 * @b Description: Set of APIs to estimate global egomotion.
 *
 */

/**
 * @defgroup global_egomotion_group Global Egomotion Interface
 * @ingroup egomotion_group
 *
 * @brief Provides global location and orientation estimation functionality.
 *
 * @{
 */

#ifndef DW_EGOMOTION_GLOBAL_GLOBALEGOMOTION_H_
#define DW_EGOMOTION_GLOBAL_GLOBALEGOMOTION_H_

#include <dw/egomotion/Egomotion.h>
#include <dw/sensors/gps/GPS.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwGlobalEgomotionObject* dwGlobalEgomotionHandle_t;
typedef struct dwGlobalEgomotionObject const* dwGlobalEgomotionConstHandle_t;

/**
 * @brief GNSS Sensor characteristics.
 **/
typedef struct
{
    //! GNSS antenna position in the rig coordinate system [m]
    dwVector3f antennaPosition;

    //! Expected horizontal position noise (CEP) of the GNSS sensor [m]
    //! A default value of 2.5 [m] will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t horizontalNoiseMeter;

    //! Expected vertical position noise (CEP) of the GNSS sensor [m]
    //! A default value of 5 [m] will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t verticalNoiseMeter;

} dwGNSSCharacteristics;

/**
 * @brief Holds initialization parameters for the global egomotion module.
 */
typedef struct
{
    //! Sensor characteristics.
    //! If this struct is zero initialized, default assumptions are made.
    dwGNSSCharacteristics sensorCharacteristics;

    //! Expected magnitude of relative egomotion rotational drift [deg/s]
    //! A default value of 10 [deg/h] will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t rotationalDrift;

    //! Size of history array, in number of state estimates it holds.
    //! A default value is 1000 is used if this is left zero-initialized.
    size_t historySize;

} dwGlobalEgomotionParameters;

/**
 * @brief Holds global egomotion state estimate.
 **/
typedef struct
{
    dwGeoPointWGS84 position;  //!< Position in WGS-84 reference system.
    dwQuaternionf orientation; //!< Rotation from rig coordinate system to ENU coordinate system.

    bool validPosition;    //!< Indicates validity of `position` estimate.
    bool validOrientation; //!< Indicates validity of `orientation` estimate.

    dwTime_t timestamp; //!< Estimate timestamp.

} dwGlobalEgomotionResult;

/**
 * @brief Holds global egomotion uncertainty estimate.
 **/
typedef struct
{
    dwConfidence3f position;    //!< Position uncertainty (easting [m], northing [m], altitude [m]).
    dwConfidence3f orientation; //!< Orientation uncertainty (roll [rad], pitch [rad], yaw [rad]).

    bool validPosition;    //!< Indicates validity of `position` uncertainty estimate.
    bool validOrientation; //!< Indicates validity of `orientation` uncertainty estimate.

    dwTime_t timestamp; //!< Estimate timestamp.

} dwGlobalEgomotionUncertainty;

/**
 * Initialize global egomotion parameters from a provided RigConfiguration. This will read out relevant sensor
 * parameters and apply them on top of default parameters.
 *
 * @param[out] params Pointer to a parameter struct to be filled out with sensor parameters
 * @param[in] rigConfiguration Handle to a rig configuration to retrieve parameters from
 * @param[in] gpsSensorName name of the GPS sensor to be used
 *
 * @return DW_INVALID_ARGUMENT - if provided params pointer or rig handle are invalid<br>
 *         DW_FILE_INVALID - if provided sensor could not be found in the rig config<br>
 *         DW_SUCCESS - if initialization of parameters succeeded<br>
 *
 * @note Clears any existing parameters set in `params`.
 *
 * @note Following parameters are extracted from the rig configuration:
 *       GPS sensor:
 *            - Position of the sensor -> `dwGlobalEgomotionParameters.sensorCharacteristics.antennaPosition`
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_initParamsFromRig(dwGlobalEgomotionParameters* params,
                                             dwConstRigHandle_t rigConfiguration,
                                             const char* gpsSensorName);

/**
 * Initializes the global egomotion module.
 *
 * @param[out] handle A pointer to the handle for the created module.
 * @param[in] params A pointer to the configuration parameters of the module.
 * @param[in] ctx Specifies the handler to the context under which the module is created.
 *
 * @return DW_INVALID_ARGUMENT - if provided parameters are invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_SUCCESS - if initialization succeeded <br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_initialize(dwGlobalEgomotionHandle_t* handle,
                                      const dwGlobalEgomotionParameters* params,
                                      dwContextHandle_t ctx);

/**
 * Resets the state estimate and all history of the global egomotion module.
 *
 * @param[in] handle Global Egomotion handle to be reset.
 *
 * @return DW_INVALID_HANDLE - if the provided handle is invalid. <br>
 *         DW_SUCCESS - if reset succeeded <br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_reset(dwGlobalEgomotionHandle_t handle);

/**
 * Releases the global egomotion module.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] handle Global Egomotion handle to be released.
 *
 * @return DW_INVALID_HANDLE - if the provided egomotion handle is invalid. <br>
 *         DW_SUCCESS  - if release succeeded <br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_release(dwGlobalEgomotionHandle_t handle);

/**
 * Adds relative egomotion estimate to the global egomotion module.
 *
 * @param[in] egomotionResult State estimate provided by relative egomotion.
 * @param[in] egomotionUncertainty Uncertainty estimate provided by relative egomotion.
 * @param[in] handle Global Egomotion handle.
 *
 * @note the state and state uncertainty estimates are required to at least have the following flags
 *       set:
 *         - DW_EGOMOTION_ROTATION
 *         - DW_EGOMOTION_LIN_VEL_X
 *         - DW_EGOMOTION_ANG_VEL_Z
 *       If any of those flags isn't set, the provided estimate will be ignored and this method
 *       will return DW_NOT_AVAILABLE.
 *
 * @return DW_INVALID_ARGUMENT - if any input arguments are invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_NOT_AVAILABLE - if the relative egomotion estimates are missing required flags or the timestamp isn't more recent that previous <br>
 *         DW_NOT_SUPPORTED - if the relative egomotion rotation estimate is not relative to local frame <br>
 *         DW_NOT_READY - no new estimate was added to history; filter missing GPS data or estimate timestamp
 *                        isn't more recent than current estimate timestamp. <br>
 *         DW_SUCCESS - a new estimate has been added to the history <br>
 **/
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_addRelativeMotion(const dwEgomotionResult* egomotionResult,
                                             const dwEgomotionUncertainty* egomotionUncertainty,
                                             dwGlobalEgomotionHandle_t handle);

/**
 * Adds GPS measurement to the global egomotion module.
 *
 * @param[in] measurement GPS measurement.
 * @param[in] handle Global Egomotion handle.
 *
 * @note providing GPS measurements only isn't sufficient to generate a state estimate, call
 *       `dwGlobalEgomotion_addRelativeMotion()` to provide the relative egomotion state estimate
 *       and generate new estimates.
 *
 * @return DW_INVALID_HANDLE - if the provided handle is invalid <br>
*          DW_INVALID_ARGUMENT - if measurement timestamp isn't more recent than previous <br>
 *         DW_SUCCESS - GPS data was added to global egomotion <br>
 **/
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_addGPSMeasurement(const dwGPSFrame* measurement,
                                             dwGlobalEgomotionHandle_t handle);

/**
 * Get timestamp of current filter estimate.
 *
 * @param[out] timestamp Timestamp of current filter estimate.
 * @param[in] handle Global Egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if provided timestamp pointer is invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_NOT_AVAILABLE - no state estimate available yet <br>
 *         DW_SUCCESS - timestamp successfully returned<br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_getTimestamp(dwTime_t* timestamp,
                                        dwGlobalEgomotionConstHandle_t handle);

/**
 * Get current filter state estimate.
 *
 * @param[out] result Current filter state estimate.
 * @param[out] uncertainty Current global state uncertainty estimate (optional, can be null).
 * @param[in] handle Global Egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if provided pointers are invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_NOT_AVAILABLE - no state estimate available yet <br>
 *         DW_SUCCESS - estimates successfully returned<br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_getEstimate(dwGlobalEgomotionResult* result,
                                       dwGlobalEgomotionUncertainty* uncertainty,
                                       dwGlobalEgomotionConstHandle_t handle);

/**
 * Computes global state estimate at given timestamp, if necessary by linear interpolation between
 * available history entries.
 *
 * This API currently supports linear extrapolation, limited to 1 second. The uncertainty estimates
 * are not extrapolated and held constant.
 *
 * This method does not modify the filter state.
 *
 * @param[out] result Global state estimate.
 * @param[out] uncertainty Global state uncertainty estimate (optional, can be null).
 * @param[in] timestamp Timestamp for which to provide position estimate.
 * @param[in] handle Global Egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if provided pointers are invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_NOT_AVAILABLE - no state estimate available yet <br>
 *         DW_SUCCESS - estimates successfully returned<br>
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_computeEstimate(dwGlobalEgomotionResult* result,
                                           dwGlobalEgomotionUncertainty* uncertainty,
                                           dwTime_t timestamp,
                                           dwGlobalEgomotionConstHandle_t handle);

/**
 * Returns the number of estimates currently stored in the history.
 *
 * @param[out] num A pointer to the number of estimates in the history.
 * @param[in] handle Global Egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_SUCCESS - state history successfully returned <br>
 **/
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_getHistorySize(size_t* num, dwGlobalEgomotionConstHandle_t handle);

/**
 * Returns an entry from the history array.
 *
 * @param[out] result Global state estimate.
 * @param[out] uncertainty Global state uncertainty estimate (optional, can be null).
 * @param[in] index Index into the history, in the range [0, `dwGlobalEgomotion_getHistorySize`),
 *                  with 0 being most recent estimate and last element pointing to oldest estimate.
 * @param[in] handle Global Egomotion module handle.
 *
 * @return DW_NOT_AVAILABLE - no state estimate available yet, or the requested index is outside of
 *                            the available history range <br>
 *         DW_INVALID_ARGUMENT - if the provided pointer is invalid <br>
 *         DW_INVALID_HANDLE - if the provided handle is invalid <br>
 *         DW_SUCCESS - history entry successfully returned<br>
 **/
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_getHistoryEntry(dwGlobalEgomotionResult* result,
                                           dwGlobalEgomotionUncertainty* uncertainty,
                                           size_t index,
                                           dwGlobalEgomotionConstHandle_t handle);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EGOMOTION_GLOBAL_GLOBALEGOMOTION_H_
