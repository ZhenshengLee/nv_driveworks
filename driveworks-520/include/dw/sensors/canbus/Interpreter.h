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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Interpreter</b>
 *
 * @b Description: This file defines CAN message interpreter methods.
 */

/**
 * @defgroup interpreter_group Interpreter
 * @ingroup can_group
 * @brief Defines the CAN message interpreter methods.
 *
 * @{
 */

#ifndef DW_SENSORS_CANBUS_INTERPRETER_H_
#define DW_SENSORS_CANBUS_INTERPRETER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>

#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/VehicleData.h>

/// Maximal number of signals in a DBC message.
#define DW_SENSORS_CAN_INTERPRETER_MESSAGE_MAX_SIGNALS 128

/// Maximal length of a message name [characters].
#define DW_SENSORS_CAN_INTERPRETER_MAX_MESSAGE_NAME_LEN 32

/// Maximal length of a signal name [characters].
#define DW_SENSORS_CAN_INTERPRETER_MAX_SIGNAL_NAME_LEN 32

#ifdef __cplusplus
extern "C" {
#endif

/// \brief CAN message interpreter handle.
typedef struct dwCANInterpreterObject* dwCANInterpreterHandle_t;

/// \brief Pushes new messages to the interpreter.
/// For the user-function expected behavior see the API description dwCANInterpreter_consume() in the API section.
typedef void (*dwCANInterpreterAddMessageFunc_t)(const dwCANMessage* msg, void* userData);

/// \brief Gets the number of signals available to be retrieved. The same signal may be available multiple times.
/// For the user-function expected behavior see the API description dwCANInterpreter_getNumberSignals() in the API section.
typedef uint32_t (*dwCANInterpreterGetNumAvailableFunc_t)(void* userData);

/// \brief Gets the name and type of a signal; the callback must return false if the query fails, true otherwise.
/// For the user-function expected behavior see the API description dwCANInterpreter_getSignalName() and
/// dwCANInterpreter_getSignalType in the API section.
typedef bool (*dwCANInterpreterGetSignalInfoFunc_t)(const char8_t** name, dwTrivialDataType* type,
                                                    dwCANVehicleData* data,
                                                    uint32_t idx, void* userData);

/// \brief Returns false if no float32 data is available, true otherwise
/// For the user-function expected behavior see the API description dwCANInterpreter_getf32() in the API section.
typedef bool (*dwCANInterpreterGetDataf32Func_t)(float32_t* value, dwTime_t* timestamp_us, uint32_t idx, void* userData);

/// \brief Returns false if no float64 data is available, true otherwise
/// For the user-function expected behavior see the API description dwCANInterpreter_getf64() in the API section.
typedef bool (*dwCANInterpreterGetDataf64Func_t)(float64_t* value, dwTime_t* timestamp_us, uint32_t idx, void* userData);

/// \brief Returns false if no int32 data is available, true otherwise
/// For the user-function expected behavior see the API description dwCANInterpreter_geti32() in the API section.
typedef bool (*dwCANInterpreterGetDatai32Func_t)(int32_t* value, dwTime_t* timestamp_us, uint32_t idx, void* userData);

/// Interface for callback based CAN data interpreter.
typedef struct dwCANInterpreterInterface
{
    /// addMessage callback function.
    dwCANInterpreterAddMessageFunc_t addMessage;
    /// getNumAvailableSignals callback function.
    dwCANInterpreterGetNumAvailableFunc_t getNumAvailableSignals;
    /// getSignalInfo callback function.
    dwCANInterpreterGetSignalInfoFunc_t getSignalInfo;
    /// getDataf32 callback function.
    dwCANInterpreterGetDataf32Func_t getDataf32;
    /// getDataf64 callback function.
    dwCANInterpreterGetDataf64Func_t getDataf64;
    /// getDatai32 callback function.
    dwCANInterpreterGetDatai32Func_t getDatai32;

} dwCANInterpreterInterface;

/**
 * Creates a CAN data interpreter based on user provided callbacks. Such an interpreter can be
 * used same way as the DBC based one. Any non-provided method, i.e., set to NULL,
 * is indicated with DW_NOT_IMPLEMENTED error when calling the corresponding interpreter method.
 *
 * @note this interpreter does not support the creation of messages and encoding of data.
 *
 * @param[out] interpreter A pointer to the place to return a valid interpreter handle on success.
 * @param[in] callbacks Specifies a set of callback defining the interpreter interface.
 * @param[in] userData A pointer to the user data to be passed to callbacks.
 * @param[in] context Specifies the handler to the context under which the interpreted is to be created.
 *
 * @return DW_INVALID_ARGUMENT if all 'addMessage', 'getNumAvailableSignals' or 'getSignalInfo' methods are NULL <br>
 *         DW_INVALID_HANDLE if provided context is invalid <br>
 *         DW_SUCCESS if successfully completed.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwCANInterpreter_buildFromCallbacks(dwCANInterpreterHandle_t* interpreter, dwCANInterpreterInterface callbacks,
                                             void* userData, dwContextHandle_t context);

/**
 * Creates a CAN data interpreter based on DBC file format.
 *
 * @note For generating the interpreter from DBC file format, a parser: <br>
 * 1. prints out the summary of parsing in debug logger. <br>
 * 2. supports only Simple Signal Multiplexing (8-bit mux) (does not support Extended Signal Multiplexing). <br>
 * 3. supports up to DW_SENSORS_CAN_INTERPRETER_MESSAGE_MAX_SIGNALS signals in a message. <br>
 * 4. supports extended CAN format. <br>
 * 5. ignores messages with no signals defined. <br>
 * 6. ignores messages defined with data length as 0 bytes. <br>
 * 7. ignores Signal Type and Signal Group Definitions (normally not used). <br>
 * 8. does not support signal sizes larger than 32 bits. <br>
 *
 * @param[out] interpreter A pointer to the place to return a valid interpreter handle on success.
 * @param[in] inputDBC A pointer to the file path for the input DBC file.
 * @param[in] ctx Specifies the handler to the context under which the interpreter is to be created.
 *
 * @return DW_INVALID_ARGUMENT if some inputs are NULL. <br>
 *         DW_INVALID_HANDLE if provided context is invalid. <br>
 *         DW_SUCCESS if successfully completed.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 **/
DW_API_PUBLIC
dwStatus dwCANInterpreter_buildFromDBC(dwCANInterpreterHandle_t* interpreter,
                                       const char8_t* inputDBC,
                                       dwContextHandle_t ctx);

/**
 * Creates a CAN data interpreter based on DBC file format and initializes the interpreter from a string.
 * 
 * @param[out] interpreter A pointer to the place to return a valid interpreter handle on success.
 * @param[in] dbc String content of a DBC file.
 * @param[in] ctx Specifies the handler to the context under which the interpreter is to be created.
 *
 * @return DW_INVALID_ARGUMENT if some inputs are NULL. <br>
 *         DW_INVALID_HANDLE if provided context is invalid. <br>
 *         DW_SUCCESS if successfully completed.
 *
 * @note This method is same as `dwCANInterpreter_buildFromDBC()` with a difference that a string is accepted as input
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 **/
DW_API_PUBLIC
dwStatus dwCANInterpreter_buildFromDBCString(dwCANInterpreterHandle_t* interpreter,
                                             const char8_t* dbc,
                                             dwContextHandle_t ctx);

/**
 * Closes previously opened interpreter. This unloads the interpreter plugin
 * as well as deallocates previously allocated memory.
 *
 * @note The given interpreter handle is set to NULL on success.
 *
 * @param[in] interpreter The interpreter handle previously created or opened through a file.
 *
 * @return DW_INVALID_HANDLE if provided interpreter handle is invalid. <br>
 *         DW_SUCCESS if successfully completed.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwCANInterpreter_release(dwCANInterpreterHandle_t interpreter);

// ------------ encoding
/**
* Gets the number of signals a CAN interpreter can encode into the provided message.
*
* @note This API is currently only supported by the interpreter built from DBC file.
*
* @param[out] num A pointer to variable, which is filled with the number.
* @param[in] msg CAN message to be used for encoding.
* @param[in] interpreter Specifies the interpreter handle previously create or opened through a file.
*
* @return DW_INVALID_ARGUMENT if the given pointer is NULL. <br>
*         DW_NOT_SUPPORTED if the provided interpreter does not support this method. <br>
*         DW_SUCCESS if successfully completed.
*
* @note A valid CAN message id must be set in the provided message.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getNumberSignalsInMessage(uint32_t* num,
                                                    const dwCANMessage* msg,
                                                    dwCANInterpreterHandle_t interpreter);

/**
* Gets the name of a signal that the CAN interpreter can encode in the currently active encoded message.
* 
* @note This API is currently only supported by an interpreter built from DBC file.
*
* @param[out] name A pointer to a const char8_t array containing a null-terminated string with signal name.
* @param[in] idx Specifies the index of a signal that is in range
*                [0:dwCANInterpreter_getNumberSignalsInMessage()-1].
* @param[in] msg CAN message to be used for encoding.
* @param[in] interpreter Specifies the interpreter handle previously created or opened through a file.
*
* @return DW_INVALID_ARGUMENT if given pointer is NULL or the index is out of range <br>
*         DW_NOT_AVAILABLE if no encoding has been started<br>
*         DW_NOT_SUPPORTED if provided interpreter does not support this method. <br>
*         DW_SUCCESS if successfully completed.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getSignalNameInMessage(const char8_t** name, uint32_t idx,
                                                 const dwCANMessage* msg,
                                                 dwCANInterpreterHandle_t interpreter);

/**
* Initializes an empty CAN message an interpreter can encoded signals into.
*
*
* @note This API is currently only supported by the interpreter built from DBC file.
*
* @param[out] msg A pointer to a CAN message to be initialized.
* @param[in] id CAN ID of the message.
* @param[in] interpreter Specifies the interpreter handle previously create or opened through a file.
*
* @return DW_INVALID_ARGUMENT if the given pointer is NULL or the message id is unknown to the interpreter. <br>
*         DW_NOT_SUPPORTED if provided interpreter does not support this method. <br>
*         DW_SUCCESS if successfully completed.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_createMessage(dwCANMessage* msg, uint32_t id, dwCANInterpreterHandle_t interpreter);

/**
* Initializes an empty can message an interpreter can encoded signals into.
* @note This API is currently only supported by the interpreter built from DBC file.
*
* @param[out] msg A pointer to a CAN message to be initialized.
* @param[in] msgName Name of the CAN message.
* @param[in] interpreter Specifies the interpreter handle previously create or opened through a file.
*
* @return DW_INVALID_ARGUMENT if the given pointer is NULL or the message id is unknown to the interpreter. <br>
*         DW_NOT_SUPPORTED if provided interpreter does not support this method. <br>
*         DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_createMessageFromName(dwCANMessage* msg, const char8_t* msgName,
                                                dwCANInterpreterHandle_t interpreter);

/**
* Encodes a value for a signal into a given message. The signal must be defined within the message ID.
* The size of the message changes to be at least the size required to cover the encoded value.
*
* @note This API is currently only supported by an interpreter built from DBC file.
*
* @param[in] value The value for the signal to be encoded.
* @param[in] signal Specifies the name of the signal to set value.
* @param[in] msg CAN message to be used for encoding.
* @param[in] interpreter Specifies the interpreter handle previously created.
*
* @return DW_INVALID_ARGUMENT if given pointer is NULL. <br>
*         DW_NOT_AVAILABLE if given signal name is not available in this message. <br>
*         DW_NOT_SUPPORTED if provided interpreter does not support this method. <br>
*         DW_SUCCESS if successfully completed.
*
* @note Given CAN message must have a valid id set. The signal must be available in the given message.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_encodef32(float32_t value, const char8_t* signal,
                                    dwCANMessage* msg,
                                    dwCANInterpreterHandle_t interpreter);

/**
* Same as dwCANInterpreter_encodef32, but for float64 types. 
* 
* @see dwCANInterpreter_encodef32
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_encodef64(float64_t value, const char8_t* signal,
                                    dwCANMessage* msg,
                                    dwCANInterpreterHandle_t interpreter);

/**
* Same as dwCANInterpreter_encodef32, but for int32 types.
* 
* @see dwCANInterpreter_encodef32
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_encodei32(int32_t value, const char8_t* signal,
                                    dwCANMessage* msg,
                                    dwCANInterpreterHandle_t interpreter);

// ----------------- consumption
/**
* Pushes a new message to the interpreter. It is the responsibility of the interpreter to parse
* the message and fill out internal structures to be capable of delivering the current vehicle
* state to the application. The state can be retrieved using the 'dwCANInterpreter_getX()' methods.
*
* @param[in] msg A pointer to the CAN frame to be consumed by the interpreter.
* @param[in] interpreter Specifies the interpreter handle previously created or opened through a file.
*
* @return DW_INVALID_HANDLE if provided interpreter handle is invalid. <br>
*         DW_NOT_IMPLEMENTED if provided interpreter does not implement the necessary callbacks. <br>
*         DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_consume(const dwCANMessage* msg,
                                  dwCANInterpreterHandle_t interpreter);

/**
* Gets the number of signals decoded and available for consumption by the application side. In general the
* number of signals is the number of valid signals found in last consumed CAN message. However this does not
* have to be true in all cases, especially in callback based interpreters.
*
* @param[out] num A pointer to the variable to be filled with the number.
* @param[in] interpreter Specifies the interpreter handle previously created or opened through a file.
*
* @return DW_INVALID_HANDLE if provided interpreter handle is invalid. <br>
*         DW_INVALID_ARGUMENT if given pointer is NULL. <br>
*         DW_SUCCESS if successfully completed.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getNumberSignals(uint32_t* num, dwCANInterpreterHandle_t interpreter);

/**
* Gets the index of a signal of the last consumed message that corresponds to a certain predefined data type.
* Some SDK modules need to know how to access data from CAN bus, independent on the definition of the
* signal name.
* 
* @note This API is not supported by an interpreter built from DBC file.
*
* @param[out] idx An index is returned here. On succes the index is in range [0; dwCANInterpreter_getNumberSignals()-1].
* @param[in] data Specifies the ID of the data signal you are interested in.
* @param[in] interpreter Specifies the interpreter handle previously created.
*
* @return DW_SUCCESS if data signal is ready to be gathered. <br>
*         DW_NOT_AVAILABLE if the last consumed message does not provide a signal for the requested data signal. <br>
*         DW_INVALID_HANDLE if given handle is invalid. <br>
*         DW_INVALID_ARGUMENT if given arguments are invalid, i.e., NULL. <br>
*         DW_NOT_IMPLEMENTED if provided interpreter does not support querying. <br>
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getDataSignalIndex(uint32_t* idx, dwCANVehicleData data,
                                             dwCANInterpreterHandle_t interpreter);

/**
* Gets the name of a signal available for consumption by the application as a string.
*
* @param[out] name A pointer to a const char8_t array containing a null-terminated string with signal name.
* @param[in] idx Specifies the index of a signal that is in range [0; dwCANInterpreter_getNumberSignals()-1].
* @param[in] interpreter Specifies the interpreter handle previously created or opened through a file.
*
* @return DW_INVALID_HANDLE if provided interpreter handle is invalid. <br>
*         DW_INVALID_ARGUMENT if given index or pointer are invalid. <br>
*         DW_SUCCESS if successfully completed.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getSignalName(const char8_t** name, uint32_t idx,
                                        dwCANInterpreterHandle_t interpreter);

/**
* Returns the type of the value of a signal available for consumption by the application.
*
* @param[out] type The data type is returned here.
* @param[in] idx Specifies the index of a signal that is in range [0; dwCANInterpreter_getNumberSignals()-1].
* @param[in] interpreter Specifies the interpreter handle previously created.
*
* @return DW_SUCCESS if signal is supported. <br>
*         DW_INVALID_HANDLE if given handle is invalid. <br>
*         DW_INVALID_ARGUMENT if given arguments are invalid, i.e., NULL. <br>
*         DW_FAILURE on any other error.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getSignalType(dwTrivialDataType* type, uint32_t idx,
                                        dwCANInterpreterHandle_t interpreter);

/**
* Returns the range of validity of data for a signal available for consumption by the application.
*
* @param[out] minimum The minimum of the range of validity.
* @param[out] maximum The maximum of the range of validity.
* @param[in] idx Specifies the index of a signal that is in range [0; dwCANInterpreter_getNumberSignals()-1].
* @param[in] interpreter Specifies the interpreter handle previously created.
*
* @return DW_INVALID_HANDLE if provided interpreter handle is invalid. <br>
*         DW_INVALID_ARGUMENT if given index or pointer are invalid. <br>
*         DW_SUCCESS if successfully completed.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getSignalRange(float64_t* minimum, float64_t* maximum, uint32_t idx,
                                         dwCANInterpreterHandle_t interpreter);

/**
* Gets a 'float32_t' value from the available values.
* The timestamp represent the time when the signal value was valid, i.e., when signal was received.
*
* @note Each signal value can only be obtained once per consumed message. <br>
* @note In case of DBC based interpreter, if both maximum and minimum are zero for a given signal in DBC,
*       the range restriction is invalidated.
*
* @param[out] value A pointer to the data value is returned here.
* @param[out] timestamp_us Timestamp of the signal.
* @param[in] idx Specifies the index within the max event numbers from dwCANInterpreter_getNumberSignals API.
* @param[in] interpreter A pointer to the interpreter handle previously created or opened through a file.
*
* @return DW_INVALID_HANDLE if given interpreter handle is not valid. <br>
*         DW_INVALID_ARGUMENT if given pointers are invalid. <br>
*         DW_NOT_AVAILABLE if no data update happened, i.e., no data was retrieved. <br>
*         DW_SUCCESS if successfully completed.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getf32(float32_t* value, dwTime_t* timestamp_us, uint32_t idx, dwCANInterpreterHandle_t interpreter);

/**
* Same as dwCANInterpreter_getf32, but for float64 types. 
* 
* @see dwCANInterpreter_getf32
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_getf64(float64_t* value, dwTime_t* timestamp_us, uint32_t idx, dwCANInterpreterHandle_t interpreter);

/**
* Same as dwCANInterpreter_getf32, but for int32 types. 
* 
* @see dwCANInterpreter_getf32
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwCANInterpreter_geti32(int32_t* value, dwTime_t* timestamp_us, uint32_t idx, dwCANInterpreterHandle_t interpreter);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CANBUS_INTERPRETER_H_
