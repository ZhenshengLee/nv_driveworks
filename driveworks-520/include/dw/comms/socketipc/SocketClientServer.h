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
 * <b>NVIDIA DriveWorks API: Inter Process Communication</b>
 *
 * @b Description: This file defines methods for inter-process communication.
 */

/**
 * @defgroup ipc_group IPC Interface
 *
 * @brief Provides functionality for inter-process communication (IPC).
 *
 * @{
 */

#ifndef DW_IPC_SOCKETCLIENTSERVER_H_
#define DW_IPC_SOCKETCLIENTSERVER_H_

#include <dw/core/base/Config.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Status.h>
#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Handle representing the a network socket server.
typedef struct dwSocketServerObject* dwSocketServerHandle_t;

/// Handle representing the a network socket client.
typedef struct dwSocketClientObject* dwSocketClientHandle_t;

/// Handle representing the a bi-directional client-server network socket connection.
typedef struct dwSocketConnectionObject* dwSocketConnectionHandle_t;

/**
 * @brief Creates and initializes a socket server accepting incoming client connections. Create a connectionPool to
 *        save all client connections of the socket server.
 *
 * @param[out] server A pointer to the server handle will be returned here.
 * @param[in] port The network port the server is listening on.
 * @param[in] connectionPoolSize The maximal number of concurrently acceptable connections.
 * @param[in] context Specifies the handle to the context under which the socket server is created.
 *
 * @return DW_CANNOT_CREATE_OBJECT - if socket initialization failed. <br>
 *         DW_INVALID_HANDLE - if provided context handle is invalid,i.e. null or of wrong type  <br>
 *         DW_SUCCESS - if server is created successfully.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwSocketServer_initialize(dwSocketServerHandle_t* const server, uint16_t const port, size_t const connectionPoolSize,
                                   dwConstContextHandle_t const context);

/**
 * @brief Accepts an incoming connection at a socket server. It will wait for connections from clients in a timeout ms.
 *        If there are connections from clients, it will return directly.
 *
 * @param[out] connection A pointer to the socket connection handle will be returned here.
 * @param[in] timeoutUs Timeout to block this call.
 * @param[in] server A handle to the socket server.
 *
 * @return DW_BUFFER_FULL - if the server's connection pool is depleted. <br>
 *         DW_TIME_OUT - if no connection could be accepted before the timeout. <br>
 *         DW_INVALID_ARGUMENT - if the server handle is invalid. <br>
 *         DW_SUCCESS - if a socket connection was established.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketServer_accept(dwSocketConnectionHandle_t* const connection, dwTime_t const timeoutUs,
                               dwSocketServerHandle_t const server);

/**
 * @brief Broadcasts a message to all connected sockets of the pool. The pool is created in the initialization of the SocketServer.
 *        And the sockets which is sent message to are saved by dwSocketServer_accept.
 *
 * @param[in] buffer A pointer to the data to be send.
 * @param[in] bufferSize The number of bytes to send.
 * @param[in] server A handle to the socket server.
 *
 * @return DW_INVALID_HANDLE - if buffer or socket handle is invalid,i.e. null or of wrong type  <br>
 *         DW_FAILURE - if a single message was not send successfully to all connections. <br>
 *         DW_SUCCESS - if all messages to all connections were send successfully.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketServer_broadcast(uint8_t const* const buffer, size_t const bufferSize, dwSocketServerHandle_t const server);

/**
 * @brief Terminate a socket server. It will release the memory of the SocketServer object, close and shutdown the socket.
 *
 * @param[in] server A handle to the socket server.
 *
 * @return DW_FAILURE - if server was not terminated cleanly <br>
 *         DW_SUCCESS - if server was terminated cleanly.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketServer_release(dwSocketServerHandle_t const server);

/**
 * @brief Creates and initializes a socket client. Create a connectionPool to
 *        save all clients sockets.
 *
 * @param[out] client A pointer to the client handle will be returned here.
 * @param[in] connectionPoolSize The maximal number of concurrently connected connections.
 * @param[in] context Specifies the handle to the context under which the socket client is created.
 *
 * @return DW_INVALID_HANDLE - if provided context handle is invalid, i.e. null or of wrong type <br>
 *         DW_SUCCESS - when client was created successfully.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwSocketClient_initialize(dwSocketClientHandle_t* const client, size_t const connectionPoolSize,
                                   dwConstContextHandle_t const context);

/**
 * @brief Connects a socket connection to a listening socket server. And it will add the connection to the client connection pool
 *        if the operation is successful. If it can't connect to the server in the timeout value, it will return.
 *
 * @param[out] connection A pointer to the socket connection handle will be returned here.
 * @param[in] host A pointer to string representation of the the server's IP address or hostname.
 * @param[in] port The network port the server is listening on.
 * @param[in] timeoutUs Timeout to block this call.
 * @param[in] client A handle to the socket client.
 *
 * @return DW_BUFFER_FULL - if the clients's connection pool is depleted. <br>
 *         DW_CANNOT_CREATE_OBJECT - if the socket can't be created or the server's IP/hostname is invalid <br>
 *         DW_TIME_OUT - if no connection could be accepted before the timeout. <br>
 *         DW_INVALID_ARGUMENT - if the client handle is invalid. <br>
 *         DW_SUCCESS - if a socket connection was established.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwSocketClient_connect(dwSocketConnectionHandle_t* const connection, char8_t const* const host, uint16_t const port,
                                dwTime_t const timeoutUs, dwSocketClientHandle_t const client);

/**
 * @brief Broadcasts a message to all connected sockets of the pool.The pool is created in the initialization of the SocketClient.
 *        And the sockets which is sent message to are saved by dwSocketClient_connect.
 *
 * @param[in] buffer A pointer to the data to be send.
 * @param[in] bufferSize The number of bytes to send.
 * @param[in] client A handle to the socket client.
 *
 * @return DW_INVALID_HANDLE - if buffer or socket handle is invalid, i.e. null or of wrong type  <br>
 *         DW_FAILURE - if a single message was not send successfully to all connections. <br>
 *         DW_SUCCESS - if all messages to all connections were send successfully.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketClient_broadcast(uint8_t const* const buffer, size_t const bufferSize, dwSocketClientHandle_t const client);

/**
 * @brief Terminate a socket client. It will release the memory of the SocketClient object, close and shutdown the socket.
 *
 * @param[in] client A handle to the socket client.
 *
 * @return DW_FAILURE - if client was not terminated cleanly <br>
 *         DW_SUCCESS - if client was terminated cleanly.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketClient_release(dwSocketClientHandle_t const client);

/**
 * Send a message of a given length through the socket connection with a timeout.
 * While sending has not timedout, the buffer will be retried to be sent as long as possible.
 *
 * @param[in] buffer A pointer to the data to be send.
 * @param[in,out] bufferSize A pointer to the number of bytes to send, and the actual number of bytes sent.
 * @param[in] timeoutUs Amount of time to try to send the data
 * @param[in] connection A handle to the network socket connection.
 *
 * @return DW_INVALID_HANDLE - if buffer or socket handle is invalid,i.e. null or of wrong type <br>
 *         DW_END_OF_STREAM - if the connection ended. <br>
 *         DW_FAILURE - if data could not be transmitted. <br>
 *         DW_TIME_OUT - if no data could be transmitted within the requested timeout time. <br>
 *         DW_SUCCESS - if data could be send.
 *
 * @note Even if the method times out, at least partially the data could have been sent.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketConnection_write(const void* const buffer, size_t* const bufferSize, dwTime_t const timeoutUs,
                                  dwSocketConnectionHandle_t const connection);

/**
 * @brief Peek at a message of a given length from the network connection (blocking within timeout period).
 *        It will not delete the message content in the protocol stack.(difference from dwSocketConnection_read)
 *
 * @param[in,out] buffer A pointer to the memory location data is written to.
 * @param[in,out] bufferSize A pointer to the number of bytes to receive, and the actual number of bytes
 *                           received on success.
 * @param[in] timeoutUs Timeout to block this call.
 *                       Specify 0 for non-blocking behavior.
 *                       Specify DW_TIMEOUT_INFINITE for infinitely blocking behavior until data is available.
 * @param[in] connection A handle to the network socket connection.
 *
 * @return DW_INVALID_HANDLE - if buffer or socket handle is invalid,i.e. null or of wrong type  <br>
 *         DW_END_OF_STREAM - if the connection ended. <br>
 *         DW_TIME_OUT - if no data is available to be received before the timeout. <br>
 *         DW_FAILURE - if data could not be received or timeout could not be set. <br>
 *         DW_SUCCESS - if data could be received.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketConnection_peek(uint8_t* const buffer, size_t* const bufferSize,
                                 dwTime_t const timeoutUs,
                                 dwSocketConnectionHandle_t const connection);

/**
 * Receive a message of a given length from the network connection. The method blocks for the provided
 * amount of time to receive the data. It will delete the message content in the protocol stack.(difference from dwSocketConnection_peek)
 *
 * @param[in,out] buffer A pointer to the memory location data is written to.
 * @param[in,out] bufferSize A pointer to the number of bytes to receive, and the actual number of bytes received.
 * @param[in] timeoutUs Time to wait to receive the content. Can be 0 for non-blocking and DW_TIMEOUT_INFINITE for blocking mode
 * @param[in] connection A handle to the network socket connection.
 *
 * @return DW_INVALID_HANDLE - if buffer or socket handle is invalid,i.e. null or of wrong type  <br>
 *         DW_END_OF_STREAM - if the connection ended. <br>
 *         DW_TIME_OUT - if timed out before any data could be received <br>
 *         DW_FAILURE - if data could not be received. <br>
 *         DW_SUCCESS - if data could be received.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketConnection_read(void* const buffer, size_t* const bufferSize, dwTime_t const timeoutUs, dwSocketConnectionHandle_t const connection);

/**
 * @brief Terminate a socket connection. It will release the memory of the SocketConnection object, close and shutdown the socket in the connection.
 *
 * @param[in] connection A handle to the socket connection.
 *
 * @return DW_FAILURE - if connection was not terminated cleanly <br>
 *         DW_SUCCESS - if connection was terminated cleanly.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSocketConnection_release(dwSocketConnectionHandle_t const connection);

#ifdef __cplusplus
}
#endif

/** @} */

#endif // DW_IPC_SOCKETCLIENTSERVER_H_
