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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: DNN Methods</b>
 *
 * @b Description: This file defines Deep Neural Network methods.
 */

/**
 * @defgroup dnn_group DNN Interface
 *
 * @brief Defines Deep Neural Network (DNN) module for performing inference using NVIDIA<sup>&reg;</sup> TensorRT<sup>&tm;</sup> models.
 *
 * @{
 */

#ifndef DW_DNN_H_
#define DW_DNN_H_

#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/core/base/Config.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/dnn/plugin/DNNPlugin.h>
#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handles representing Deep Neural Network interface.
 */
typedef struct dwDNNObject* dwDNNHandle_t;
typedef struct dwDNNObject const* dwConstDNNHandle_t;

/** Specifies TensorRT model header.
 */
typedef struct
{
    dwDataConditionerParams dataConditionerParams; /**< DataConditioner parameters for running this network.*/
} dwDNNMetaData;

/**
 * Creates and initializes a TensorRT Network from file.
 *
 * @param[out] network A pointer to network handle that will be initialized from parameters.
 * @param[in] modelFilename A pointer to the name of the TensorRT model file.
 * @param[in] pluginConfiguration An optional pointer to plugin configuration for custom layers.
 * @param[in] processorType Processor that the inference should run on. Note that the model must be
 * generated for this processor type.
 * @param[in] context Specifies the handle to the context under which the DNN module is created.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the network handle or the model filename are NULL. <br>
 *         DW_DNN_INVALID_MODEL - if the provided model is invalid. <br>
 *         DW_CUDA_ERROR - if compute capability does not met the network type's requirements. <br>
 *         DW_FILE_NOT_FOUND - if given model file does not exist. <br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note The network file must be created by the TensorRT_optimization tool.
 *
 * @note DNN module will look for metadata file named \<modelFilename\>.json in the same folder.
 * If it is present, metadata will be loaded from that file. Otherwise, it will be filled with default
 * values. Example metadata:
 *
 *     {
 *         "dataConditionerParams" : {
 *             "meanValue" : [0.0, 0.0, 0.0],
 *             "splitPlanes" : true,
 *             "pixelScaleCoefficient": 1.0,
 *             "ignoreAspectRatio" : false,
 *             "doPerPlaneMeanNormalization" : false
 *         }
 *         "tonemapType" : "none",
 *         "__comment": "tonemapType can be one of {none, agtm}"
 *     }
 * \
 *
 */
DW_API_PUBLIC
dwStatus dwDNN_initializeTensorRTFromFile(dwDNNHandle_t* const network, const char8_t* const modelFilename,
                                          const dwDNNPluginConfiguration* const pluginConfiguration,
                                          dwProcessorType const processorType, dwContextHandle_t const context);

/**
 * Creates and initializes a TensorRT Network from file with DLA Engine ID.
 *
 * @param[out] network A pointer to network handle that will be initialized from parameters.
 * @param[in] modelFilename A pointer to the name of the TensorRT model file.
 * @param[in] pluginConfiguration An optional pointer to plugin configuration for custom layers.
 * @param[in] processorType Processor that the inference should run on. Note that the model must be
 * generated for this processor type.
 * @param[in] engineId Specifies the DLA engine id if the processorType is DW_PROCESSOR_TYPE_CUDLA.
 * @param[in] context Specifies the handle to the context under which the DNN module is created.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the network handle or the model filename are NULL. <br>
 *         DW_DNN_INVALID_MODEL - if the provided model is invalid. <br>
 *         DW_CUDA_ERROR - if compute capability does not met the network type's requirements. <br>
 *         DW_FILE_NOT_FOUND - if given model file does not exist. <br>
 *         DW_SUCCESS otherwise.<br>
*/
DW_API_PUBLIC
dwStatus dwDNN_initializeTensorRTFromFileWithEngineId(dwDNNHandle_t* const network, const char8_t* const modelFilename,
                                                      const dwDNNPluginConfiguration* const pluginConfiguration,
                                                      dwProcessorType const processorType, uint32_t engineId,
                                                      dwContextHandle_t const context);

/**
 * Creates and initializes a TensorRT Network from memory.
 *
 * @param[out] network A pointer to network handle that is initialized from parameters.
 * @param[in] modelContent A pointer to network content in the memory.
 * @param[in] modelContentSize Specifies the size of network content in memory, in bytes.
 * @param[in] pluginConfiguration An optional pointer to plugin configuration for custom layers.
 * @param[in] processorType Processor that the inference should run on. Note that the model must be
 * generated for this processor type.
 * @param[in] context Specifies a handle to the context under which the DNN module is created.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the network handle or the model content are NULL. <br>
 *         DW_DNN_INVALID_MODEL - if the provided model is invalid. <br>
 *         DW_CUDA_ERROR - if compute capability does not met the network type's requirements. <br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note The network file must be created by the TensorRT_optimization tool.
 *
 * @note DNN module will fill metadata with default values.
 */
DW_API_PUBLIC
dwStatus dwDNN_initializeTensorRTFromMemory(dwDNNHandle_t* const network,
                                            const char8_t* const modelContent,
                                            uint32_t const modelContentSize,
                                            const dwDNNPluginConfiguration* const pluginConfiguration,
                                            dwProcessorType const processorType, dwContextHandle_t const context);

/**
 * Creates and initializes a TensorRT Network from memory with DLA Engine ID.
 *
 * @param[out] network A pointer to network handle that is initialized from parameters.
 * @param[in] modelContent A pointer to network content in the memory.
 * @param[in] modelContentSize Specifies the size of network content in memory, in bytes.
 * @param[in] pluginConfiguration An optional pointer to plugin configuration for custom layers.
 * @param[in] processorType Processor that the inference should run on. Note that the model must be
 * generated for this processor type.
 * @param[in] engineId Specifies the DLA engine id if the processorType is DW_PROCESSOR_TYPE_CUDLA.
 * @param[in] context Specifies a handle to the context under which the DNN module is created.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the network handle or the model content are NULL. <br>
 *         DW_DNN_INVALID_MODEL - if the provided model is invalid. <br>
 *         DW_CUDA_ERROR - if compute capability does not met the network type's requirements. <br>
 *         DW_SUCCESS otherwise.<br>
 *
*/
DW_API_PUBLIC
dwStatus dwDNN_initializeTensorRTFromMemoryWithEngineId(dwDNNHandle_t* const network,
                                                        const char8_t* const modelContent,
                                                        uint32_t const modelContentSize,
                                                        const dwDNNPluginConfiguration* const pluginConfiguration,
                                                        dwProcessorType const processorType, uint32_t engineId,
                                                        dwContextHandle_t const context);
/**
 * Resets a given network.
 *
 * @param[in] network Network handle to reset.
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle is null. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_reset(dwDNNHandle_t const network);

/**
 * Releases a given network.
 *
 * @param[in] network The network handle to release.
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle is NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_release(dwDNNHandle_t const network);

/**
 * Forwards pass from the first input blob to the first output blob (a shortcut for a single input - single output
 * network).
 *
 * @note This method requires the network to be single input - single output.
 *
 * @note The size of the output blob and input blob must match the respective
 * sizes returned by dwDNN_getOutputSize() and dwDNN_getInputSize().
 *
 * @param[out] dOutput A pointer to the output blob in GPU memory.
 * @param[in] dInput A pointer to the input blob in GPU memory.
 * @param[in] batchsize Batch size for inference. Batch size must be equal or less than the one that was
 * given at the time of model generation.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle, output or input are NULL or
 *         if provided network is not a single input - single output network. <br>
 *         DW_INTERNAL_ERROR - if DNN engine cannot execute inference on the given network. <br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Inference performance might be suboptimal if the network has been generated for a larger batch size.
 * @note If the processor type is DLA, the inference will always perform on maximum batch size. The only
 * advantage of giving a lower batchsize as input is that the output to be copied will be smaller in size.
 */
DW_API_PUBLIC
dwStatus dwDNN_inferSIO(float32_t* const dOutput, const float32_t* const dInput, uint32_t const batchsize,
                        dwDNNHandle_t const network);

/**
 * Forwards pass from all input blobs to all output blobs.
 *
 * @note: The size of the output blob and input blob must match the respective sizes returned by
 * dwDNN_getOutputSize() and dwDNN_getInputSize().
 * @note: The umber of blobs must match the respective number of blobs returned by dwDNN_getInputBlobCount()
 * and dwDNN_getOutputBlobCount().
 *
 * @param[out] dOutput A pointer to an array of pointers to the input blobs in GPU Memory.
 * @param[in] dInput A pointer to an array of pointers to the output blobs in GPU Memory.
 * @param[in] batchsize Batch size for inference. Batch size must be equal or less than the one that was
 * given at the time of model generation.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle , output or input are NULL. <br>
 *         DW_INTERNAL_ERROR - if DNN engine cannot execute inference on the given network. <br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Inference performance might be suboptimal if the network has been generated for a larger batch size.
 * @note If the processor type is DLA, the inference will always perform on maximum batch size. The only
 * advantage of giving a lower batchsize as input is that the output to be copied will be smaller in size.
 */
DW_API_PUBLIC
dwStatus dwDNN_inferRaw(float32_t* const* const dOutput, const float32_t* const* const dInput,
                        uint32_t const batchsize, dwDNNHandle_t const network);

/**
 * Sets the CUDA stream for infer operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is the stream 0, resulting in synchronous operations.
 * @param[in] network A handle to the DNN module to set CUDA stream for.
 *
 * @return DW_INVALID_ARGUMENT if the given network handle is NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_setCUDAStream(cudaStream_t const stream, dwDNNHandle_t const network);

/**
 * Gets the CUDA stream used by the feature list.
 *
 * @param[out] stream The CUDA stream currently used.
 * @param[in] network A handle to the DNN module.
 *
 * @return DW_INVALID_HANDLE if the given network handle or the stream are NULL or of wrong type. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getCUDAStream(cudaStream_t* const stream, dwDNNHandle_t const network);

/**
 * Gets the input blob size at blobIndex.
 *
 * @param[out] blobSize A pointer to the where the input blob size is returned.
 * @param[in] blobIndex Specifies the blob index; must be in the range [0 dwDNN_getInputBlobCount()-1].
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle or the blob size are NULL or
 *         blobIndex is not in range [0 dwDNN_getInputBlobCount()-1]. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getInputSize(dwBlobSize* const blobSize, uint32_t const blobIndex, dwDNNHandle_t const network);

/**
 * Gets the output blob size at blobIndex.
 *
 * @param[out] blobSize A pointer to the where the output blob size is returned.
 * @param[in] blobIndex Specifies the blob index; must be in the range [0 dwDNN_getOutputBlobCount()-1].
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle is or blobSize are NULL or
 *         blobIndex is not in range [0 dwDNN_getOutputBlobCount()-1]. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getOutputSize(dwBlobSize* const blobSize, uint32_t const blobIndex, dwDNNHandle_t const network);

/**
 * Gets the input tensor properties at blobIndex.
 *
 * @param[out] tensorProps Tensor properties.
 * @param[in] blobIndex Specifies the blob index; must be in the range [0 dwDNN_getInputBlobCount()-1].
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle or the blob size are NULL or
 *         blobIndex is not in range [0 dwDNN_getInputBlobCount()-1]. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getInputTensorProperties(dwDNNTensorProperties* const tensorProps, uint32_t const blobIndex, dwDNNHandle_t const network);

/**
 * Gets the output tensor properties at blobIndex.
 *
 * @param[out] tensorProps Tensor properties.
 * @param[in] blobIndex Specifies the blob index; must be in the range [0 dwDNN_getOutputBlobCount()-1].
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle is or blobSize are NULL or
 *         blobIndex is not in range [0 dwDNN_getOutputBlobCount()-1]. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getOutputTensorProperties(dwDNNTensorProperties* const tensorProps, uint32_t const blobIndex, dwDNNHandle_t const network);

/**
 * Gets the input blob count.
 *
 * @param[out] count A pointer to the number of input blobs.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle or count are NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getInputBlobCount(uint32_t* const count, dwDNNHandle_t const network);

/**
 * Gets the output blob count.
 * @param[out] count A pointer to the number of output blobs.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle or count are NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getOutputBlobCount(uint32_t* const count, dwDNNHandle_t const network);

/**
 * Gets the index of an input blob with a given blob name.
 *
 * @param[out] blobIndex A pointer to the index of the blob with the given name.
 * @param[in] blobName A pointer to the name of an input blob.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle, blobIndex or blobName are null or
 *         an input blob with the given name is not found. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getInputIndex(uint32_t* const blobIndex, const char8_t* const blobName, dwDNNHandle_t const network);

/**
 * Gets the index of an output blob with a given blob name.
 *
 * @param[out] blobIndex A pointer to the index of the blob with the given name.
 * @param[in] blobName A pointer to the name of an output blob.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_HANDLE - if provided network handle, blobIndex or blobName are NULL or
 *         an output blob with the given name is not found or the network handle is of wrong type <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_getOutputIndex(uint32_t* const blobIndex, const char8_t* const blobName, dwDNNHandle_t const network);

/**
 * Returns the metadata for the associated network model.
 *
 * @param[out] metaData A pointer to a metadata structure.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle or metaData are NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 *
 */
DW_API_PUBLIC
dwStatus dwDNN_getMetaData(dwDNNMetaData* const metaData, dwDNNHandle_t const network);

/**
 * Runs inference pipeline on the given input.
 * @param[out] outputTensors Output tensors.
 * @param[in] outputTensorCount Number of output tensors.
 * @param[in] inputTensors Input tensors.
 * @param[in] inputTensorCount Number of input tensors.
 * @param[in] network Network handle created with dwDNN_initialize().
 *
 * @return DW_INVALID_ARGUMENT - if provided network handle, input or output are NULL. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwDNN_infer(dwDNNTensorHandle_t* const outputTensors, uint32_t const outputTensorCount,
                     dwConstDNNTensorHandle_t* const inputTensors, uint32_t const inputTensorCount, dwDNNHandle_t const network);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_DNN_H_
