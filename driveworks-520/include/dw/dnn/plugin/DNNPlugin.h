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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: DNN Plugin Interface</b>
 *
 * @b Description: This file defines the DNN custom layer plugin interface layer.
 */

#ifndef DW_DNN_PLUGIN_H_
#define DW_DNN_PLUGIN_H_

/**
 * @defgroup dnn_plugin_group DNN Plugin
 * Provides an interface for supporting non-standard DNN layers.
 *
 * @ingroup dnn_group
 * @{
 */

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Specifies plugin configuration
 */
typedef struct
{
    const char8_t* pluginLibraryPath; /**< Path to a plugin shared object. Path must be either absolute path or path relative to DW lib folder. */
    const char8_t* layerName;         /**< Name of the custom layer. */
} dwDNNCustomLayer;

/** Specified plugin configuration
 */
typedef struct
{
    const dwDNNCustomLayer* customLayers; /**< Array of custom layers. */
    size_t numCustomLayers;               /**< Number of custom layers */
} dwDNNPluginConfiguration;

/**
 * Represents different layouts for plugins
 */
typedef enum dwDNNPluginLayout {
    DW_DNN_PLUGIN_LAYOUT_LINEAR     = 0,                           //!< CHW
    DW_DNN_PLUGIN_LAYOUT_NCHW       = DW_DNN_PLUGIN_LAYOUT_LINEAR, //!< CHW
    DW_DNN_PLUGIN_LAYOUT_NC2HW2     = 1,                           //!< CHW with 2-element packed channels
    DW_DNN_PLUGIN_LAYOUT_CHW2       = DW_DNN_PLUGIN_LAYOUT_NC2HW2, //!< CHW with 2-element packed channels
    DW_DNN_PLUGIN_LAYOUT_NHWC8      = 2,                           //!< HWC with 8-element packed channels. (C must be a multiple of 8)
    DW_DNN_PLUGIN_LAYOUT_HWC8       = DW_DNN_PLUGIN_LAYOUT_NHWC8,  //!< HWC with 8-element packed channels. (C must be a multiple of 8)
    DW_DNN_PLUGIN_LAYOUT_CHW4       = 3,                           //!< CHW with 4-element packed channels
    DW_DNN_PLUGIN_LAYOUT_CHW16      = 4,                           //!< CHW with 16-element packed channels
    DW_DNN_PLUGIN_LAYOUT_CHW32      = 5,                           //!< CHW with 32-element packed channels
    DW_DNN_PLUGIN_LAYOUT_DHWC8      = 6,                           //!< DHWC with 8-element packed channels. (C must be a multiple of 8)
    DW_DNN_PLUGIN_LAYOUT_CDHW32     = 7,                           //!< CDHW with 32-element packed channels
    DW_DNN_PLUGIN_LAYOUT_HWC        = 8,                           //!< HWC Non-vectorized channel-last format
    DW_DNN_PLUGIN_LAYOUT_DLA_LINEAR = 9,                           //!< CHW DLA planar format
    DW_DNN_PLUGIN_LAYOUT_DLA_HWC4   = 10,                          //!< HWC DLA image format
} dwDNNPluginLayout;

/**
 * Stores DNN weights.
 */
typedef struct dwDNNPluginWeights
{
    dwPrecision precision; //!< data type of the weights
    const void* values;    //!< the weight values, in a contiguous array
    int64_t count;         //!< the number of weights in the array
} dwDNNPluginWeights;

/**
 * Plugin field type. Equivalent to PluginFieldType in TensorRT
 */
typedef enum dwDNNPluginFieldType {
    DW_DNN_PLUGIN_FIELD_TYPE_FLOAT16 = 0, //!< FP16 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_FLOAT32 = 1, //!< FP32 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_FLOAT64 = 2, //!< FP64 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_INT8    = 3, //!< INT8 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_INT16   = 4, //!< INT16 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_INT32   = 5, //!< INT32 field type.
    DW_DNN_PLUGIN_FIELD_TYPE_CHAR    = 6, //!< char field type.
    DW_DNN_PLUGIN_FIELD_TYPE_DIMS    = 7, //!< dwBlobSize field type.
    DW_DNN_PLUGIN_FIELD_TYPE_UNKNOWN = 8  //!< Unknown field type.
} dwDNNPluginFieldType;

/**
 * DNN plugin field. Equivalent to PluginField in TensorRT.
 */
typedef struct dwDNNPluginField
{
    /// Plugin field attribute name
    const char8_t* name;
    /// Plugin field attribute data
    const void* data;
    /// Plugin fild attribute type
    dwDNNPluginFieldType type;
    /// Number of data entries in the plugin attribute
    int32_t length;
} dwDNNPluginField;

/**
 * DNN plugin field colleciton. Equivalent to PluginFieldCollection in TensorRT.
 */
typedef struct dwDNNPluginFieldCollection
{
    int32_t numFields;              //!< Number of dwDNNPluginField entries
    const dwDNNPluginField* fields; //!< Pointer to dwDNNPluginField entries
} dwDNNPluginFieldCollection;

/**
 * DNN Plugin tensor descriptor
 */
typedef struct dwDNNPluginTensorDesc
{
    dwBlobSize dims;          //!< Tensor dimensions
    dwPrecision precision;    //!< Tensor precision
    dwDNNPluginLayout layout; //!< Tensor layout
    float32_t scale;          //!< Tensor scale
} dwDNNPluginTensorDesc;

/**
 * _dwDNNPluginHandle_t can be optionally used for storing and accessing variables among the functions
 * defined below.
 */
typedef void* _dwDNNPluginHandle_t;
typedef void const* _dwConstDNNPluginHandle_t;

/**
 * Creates a custom plugin.
 *
 * @param[out] handle Pointer to a DNN plugin object.
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_create(_dwDNNPluginHandle_t* handle);

/**
 * Clones the plugin. Note that shallow copy is sufficient.
 *
 * @param[out] out Pointer to the clone.
 * @param[in] handle Pointer to a DNN plugin object.
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_clone(_dwDNNPluginHandle_t* out, _dwDNNPluginHandle_t handle);

/**
 * Initializes the created plugin.
 *
 * @param[in] handle Pointer to a DNN plugin object.
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_setup(_dwDNNPluginHandle_t handle);

/**
 * Terminates the plugin.
 *
 * @param[in] handle Pointer to a DNN plugin object.
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_terminate(_dwDNNPluginHandle_t handle);

/**
 * Destroys the plugin.
 *
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_destroy(_dwDNNPluginHandle_t handle);

/**
 * Returns number of outputs
 * @param[out] numOutputs Number of outputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getNumOutputs(int32_t* numOutputs, _dwDNNPluginHandle_t handle);

/**
 * Returns output dimensions of an output at a given index based on inputDimensions
 * @param[out] outputDimensions Output dimensions
 * @param[in] outputIndex Output index
 * @param[in] inputDimensions Array of input dimensions
 * @param[in] numInputs Number of inputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getOutputDimensions(dwBlobSize* outputDimensions,
                                          int32_t outputIndex, const dwBlobSize* inputDimensions,
                                          int32_t numInputs, _dwDNNPluginHandle_t handle);

/**
 * Returns workspace size.
 * @param[out] workspaceSize Workspace size
 * @param[in] maxBatchSize Maximum batch size
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getWorkspaceSize(size_t* workspaceSize, int32_t maxBatchSize,
                                       _dwDNNPluginHandle_t handle);

/**
 * Returns a flag indicating whether the given format is supported
 *
 * @param[out] res Flag indicating whether the given format is supported
 * @param[in] index Index of the tensor descriptor in inOut.
 * @param[in] inOut List of input/output tensor descriptors.
 * @param[in] numInputs Number of inputs.
 * @param[in] numOutputs Number of outputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_supportsFormatCombination(bool* res, int32_t index, const dwDNNPluginTensorDesc* inOut,
                                                int32_t numInputs, int32_t numOutputs,
                                                _dwConstDNNPluginHandle_t handle);

/**
 * Configures the plugin with given format
 * @param[in] inputDescs Array of input tensor descriptors
 * @param[in] numInputs Number of inputs
 * @param[in] outputDescs Array of output tensor descriptors
 * @param[in] numOutputs Number of outputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_configurePlugin(const dwDNNPluginTensorDesc* inputDescs, int32_t numInputs,
                                      const dwDNNPluginTensorDesc* outputDescs, int32_t numOutputs,
                                      _dwDNNPluginHandle_t handle);

/**
 * Performs forward-pass.
 * @param[in] batchSize Batch size
 * @param[in] inputs Array of inputs
 * @param[in] outputs Array of outputs
 * @param[in] workspace Pointer to workspace
 * @param[in] stream CUDA stream
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_enqueue(int32_t batchSize, const void* const* inputs, void** outputs,
                              void* workspace, cudaStream_t stream, _dwDNNPluginHandle_t handle);

/**
 * Returns serialization size
 * @param[out] serializationSize Serialization size
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getSerializationSize(size_t* serializationSize, _dwDNNPluginHandle_t handle);

/**
 * Serializes the plugin to buffer. The size of the buffer is returned by ::_dwDNNPlugin_getSerializationSize.
 *
 * @param[out] buffer Buffer to store the layer to
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_serialize(void* buffer, _dwDNNPluginHandle_t handle);

/**
 * Deserializes plugin from buffer.
 *
 * @param[in] name Name of the plugin.
 * @param[in] buffer Buffer to deserialize plugin from.
 * @param[in] len Size of the buffer in bytes.
 * @param[in] handle Pointer to a DNN plugin object.
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_deserializeFromBuffer(const char8_t* name, const void* buffer,
                                            size_t len, _dwDNNPluginHandle_t handle);

/**
 * Deserializes plugin from field collection.
 *
 * @param[in] name Name of the plugin.
 * @param[in] fieldCollection Field collection
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_deserializeFromFieldCollection(const char8_t* name,
                                                     const dwDNNPluginFieldCollection* fieldCollection,
                                                     _dwDNNPluginHandle_t handle);

/**
 * Deserializes plugin from weights. This is only required if the raw model is CAFFE.
 *
 * @param[in] weights List of weights
 * @param[in] numWeights Number of weights
 * @param[in] handle Pointer to a DNN plugin object
 * @return dwStatus
 */
dwStatus _dwDNNPlugin_deserializeFromWeights(const dwDNNPluginWeights* weights, int32_t numWeights,
                                             _dwDNNPluginHandle_t handle);

/**
 * Returns the plugin type as string.
 *
 * @param[out] pluginType Plugin type
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getPluginType(const char8_t** pluginType, _dwConstDNNPluginHandle_t handle);

/**
 * Returns plugin version as string.
 *
 * @param[out] pluginVersion Plugin version
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getPluginVersion(const char8_t** pluginVersion, _dwConstDNNPluginHandle_t handle);

/**
 * Sets plugin namespace.
 *
 * @param[in] pluginNamespace Plugin namespace
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_setPluginNamespace(const char8_t* pluginNamespace, _dwDNNPluginHandle_t handle);

/**
 * Returns plugin namespace.
 *
 * @param[out] pluginNamespace Plugin namespace
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getPluginNamespace(const char8_t** pluginNamespace, _dwConstDNNPluginHandle_t handle);

/**
 * Returns whether output is broadcast across batch.
 *
 * @param[out] isOutputBroadcastAcrossBatch Flag indicating whether output at outputIndex is broadcast across
 * batch
 * @param[in] outputIndex Output index
 * @param[in] inputIsBroadcasted List of flags indicating whether inputs are broadcasted
 * @param[in] numInputs Number of inputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_isOutputBroadcastAcrossBatch(bool* isOutputBroadcastAcrossBatch,
                                                   int32_t outputIndex, const bool* inputIsBroadcasted,
                                                   int32_t numInputs, _dwConstDNNPluginHandle_t handle);

/**
 * Returns whether plugin can use input that is broadcast across batch without replication.
 *
 * @param[out] canBroadcastInputAcrossBatch Flag indicating whether plugin can use input that is broadcast
 * across batch without replication
 * @param[in] inputIndex Index of input that could be broadcast
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_canBroadcastInputAcrossBatch(bool* canBroadcastInputAcrossBatch, int32_t inputIndex,
                                                   _dwConstDNNPluginHandle_t handle);

/**
 * Returns output precision at given index given the input precisions.
 *
 * @param[out] outputPrecision Output precision
 * @param[in] outputIndex Output index
 * @param[in] inputPrecisions List of input precisions
 * @param[in] numInputs Number of inputs
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getOutputPrecision(dwPrecision* outputPrecision, int32_t outputIndex,
                                         const dwPrecision* inputPrecisions, int32_t numInputs,
                                         _dwConstDNNPluginHandle_t handle);

/**
 * Returns a list of fields that needs to be passed to plugin at creation.
 *
 * @param[out] fieldCollection Field collection
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
dwStatus _dwDNNPlugin_getPluginFieldCollection(dwDNNPluginFieldCollection* fieldCollection,
                                               _dwDNNPluginHandle_t handle);

// -----------------------------------------------------------------------------
// Deprecated Functions
// -----------------------------------------------------------------------------

/**
 * Initializes the custom plugin from weights.
 * @param[out] handle Pointer to a DNN Plugin object
 * @param[in] layerName Name of the custom layer
 * @param[in] weights Array of weights structure
 * @param[in] numWeights Number of weights structure in weights
 * @return DW_SUCCESS, DW_FAILURE
 */
DW_DEPRECATED("WARNING: will be removed in the next major release")
dwStatus _dwDNNPlugin_initializeFromWeights(_dwDNNPluginHandle_t* handle, const char8_t* layerName,
                                            const dwDNNPluginWeights* weights, int32_t numWeights);

/**
 * Initializes the custom plugin from serialized bytes
 * @param[out] handle Pointer to a DNN plugin object
 * @param[in] layerName Name of the custom layer
 * @param[in] data Serialized layer data
 * @param[in] length Length of the serialized data
 * @return DW_SUCCESS, DW_FAILURE
 */
DW_DEPRECATED("WARNING: will be removed in the next major release")
dwStatus _dwDNNPlugin_initialize(_dwDNNPluginHandle_t* handle, const char8_t* layerName,
                                 const void* data, size_t length);

/**
 * Releases the custom plugin
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
DW_DEPRECATED("WARNING: will be removed in the next major release")
dwStatus _dwDNNPlugin_release(_dwDNNPluginHandle_t handle);

/**
 * Configures the plugin with given format
 * @param[in] inputDimensions Array of input dimensions
 * @param[in] numInputs Number of inputs
 * @param[in] outputDimensions Array of output dimensions
 * @param[in] numOutputs Number of outputs
 * @param[in] precision Precision
 * @param[in] layout Layout
 * @param[in] maxBatchSize Maximum batch size
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
DW_DEPRECATED("WARNING: will be removed in the next major release")
dwStatus _dwDNNPlugin_configureWithFormat(const dwBlobSize* inputDimensions,
                                          int32_t numInputs, const dwBlobSize* outputDimensions,
                                          int32_t numOutputs, dwPrecision precision,
                                          dwDNNPluginLayout layout, int32_t maxBatchSize,
                                          _dwDNNPluginHandle_t handle);

/**
 * Returns a flag indicating whether the given format is supported
 * @param[out] res Flag indicating whether the given format is supported
 * @param[in] precision Precision
 * @param[in] pluginLayout Layout
 * @param[in] handle Pointer to a DNN plugin object
 * @return DW_SUCCESS, DW_FAILURE
 */
DW_DEPRECATED("WARNING: will be removed in the next major release")
dwStatus _dwDNNPlugin_supportsFormat(bool* res, dwPrecision precision,
                                     dwDNNPluginLayout pluginLayout, _dwDNNPluginHandle_t handle);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // DW_DNN_PLUGIN_H_
