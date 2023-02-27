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
 * <b>NVIDIA DriveWorks API: DNNTensor Structures and Methods</b>
 *
 * @b Description: This file defines DNNTensor structures and methods
 */

/**
 * @defgroup dnntensor_group DNNTensor Interface
 *
 * @brief Defines DNNTensor module for managing tensor content.
 *
 * @{
 */

#ifndef DW_DNN_TENSOR_H_
#define DW_DNN_TENSOR_H_

#include <dw/core/context/Context.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handles representing Deep Neural Network interface.
 */
typedef struct dwDNNTensorObject* dwDNNTensorHandle_t;
typedef struct dwDNNTensorObject const* dwConstDNNTensorHandle_t;

/// Maximum number of dimensions a tensor can have including batch dimension (N).
#define DW_DNN_TENSOR_MAX_DIMENSIONS 8U

/// Speficies the type of a tensor
typedef enum {
    /// CPU tensor
    DW_DNN_TENSOR_TYPE_CPU = 0,
    /// CUDA tensor
    DW_DNN_TENSOR_TYPE_CUDA = 1,
    /// NvMedia tensor
    DW_DNN_TENSOR_TYPE_NVMEDIA = 2,
} dwDNNTensorType;

/// Specifies the layout of a tensor
/// Here the letters in the suffix define:
///   - N: number of images in the batch
///   - H: height of the image
///   - W: width of the image
///   - C: number of channels of the image
typedef enum {
    /// Planar tensor. This is the most common tensor layout.
    DW_DNN_TENSOR_LAYOUT_NCHW = 0,
    /// Interleaved tensor
    DW_DNN_TENSOR_LAYOUT_NHWC = 1,
    /// Tensor with both interleaved and planar channels.
    /// The interleaved channels are fixed and the dimension size at index 0 is the
    /// number of actual interleaved channels on the last plane.
    DW_DNN_TENSOR_LAYOUT_NCHWx = 2
} dwDNNTensorLayout;

/// Represents the color space the data is represented in. If unknown, then its custom or non color data
typedef enum {
    DW_DNN_TENSOR_COLORSPACE_UNKNOWN = 0,
    DW_DNN_TENSOR_COLORSPACE_RGB     = 1,
    DW_DNN_TENSOR_COLORSPACE_YUV     = 2,
} dwDNNTensorColorSpace;

/// Specifies DNNTensor properties.
typedef struct
{
    /// Data type of elements of the tensor
    dwTrivialDataType dataType;
    /// Tensor type
    dwDNNTensorType tensorType;
    /// Tensor layout
    dwDNNTensorLayout tensorLayout;
    /// Indicates whether the memory allocation should be mapped to GPU.
    /// This allocates memory using cudaHostAlloc with cudaHostAllocMapped flag.
    /// This argument is only valid if tensor type is DW_DNN_TENSOR_TYPE_CPU or DW_DNN_TENSOR_TYPE_CUDA.
    bool isGPUMapped;
    /// Number of dimensions of the tensor
    uint32_t numDimensions;
    /// Dimensions of the tensor to match the selected layout type.
    /// The order of dimensions is defined by the `dwDNNTensorLayout` by reading the last suffix of
    /// DW_DNN_TENSOR_LAYOUT_... in reverse order. For example given `::DW_DNN_TENSOR_LAYOUT_NCHW`, `dimensionSize`
    /// should be set to: [0] = width, [1] = height, [2] = number of channels and [3] = batch size.
    ///
    /// @note Use `dwDNNTensor_getLayoutView()` to traverse the tensor in order to avoid
    /// having to compute the stride and offset for each dimension.
    uint32_t dimensionSize[DW_DNN_TENSOR_MAX_DIMENSIONS];
    /// Color space of the data in the tensor
    dwDNNTensorColorSpace colorSpace;
    /// Tensor scale value for reformatting fro m higher precision to lower precision. Values are min and max of dynamic range
    float32_t dynamicRange[2];
} dwDNNTensorProperties;

/// Exposes the content of a dwDNNTensorHandle_t.
typedef struct
{
    /// Defines the properties of the tensor
    dwDNNTensorProperties prop;
    /// Pointer to the tensor content on CPU/GPU or NvMedia
    /// @note NvMedia content cannot be traversed using this pointer. dwDNNTensorStreamer should be used
    /// to stream NvMedia to CPU or GPU to be able to access the content.
    const void* ptr;
} dwDNNTensor;

/**
 * Creates and allocates resources for a dwDNNTensorHandle_t based on the properties
 * @param[out] tensorHandle A tensor handle
 * @param[in] properties Pointer to the tensor properties
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the tensor was created successfully, <br>
 *         DW_INVALID_ARGUMENT if any of the given parameters is invalid, <br>
 *         DW_INVALID_HANDLE if the given conext handle is invalid. <br>
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_create(dwDNNTensorHandle_t* const tensorHandle,
                            dwDNNTensorProperties const* const properties,
                            dwContextHandle_t const ctx);

/**
 * Destroys the tensor handle and frees any memory created by dwDNNTensor_create().
 *
 * @param[in] tensorHandle A tensor handle
 *
 * @return DW_SUCCESS if the tensor was destroyed successfully, <br>
 *         DW_INVALID_HANDLE if the given tensor handle is invalid.  <br>
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_destroy(dwDNNTensorHandle_t const tensorHandle);

/**
 * Returns coefficients to facilitate traversing the given dimension.
 * The coefficients can be used to estimate the location in memory by:
 * tensorData[idx * stride + offset] where idx is in [0, numElements)
 *
 * @note This function is not supported by NvMedia tensors. In order to access the content of the tensors in
 * question, stream them to CPU or CUDA using dwDNNTensorStreamer.
 *
 * @param[out] offset Offset. Stored in CPU.
 * @param[out] stride Stride. Stored in CPU.
 * @param[out] numElements Number of elements in the given desired dimension. Stored in CPU.
 * @param[in] indices List of indices having the same size as dimensions of the tensor and indicating
 * the starting position of the traversal
 * @param[in] numIndices Number of given indices
 * @param[in] dimension Dimension to be traversed
 * @param[in] tensorHandle A tensor handle
 * @return DW_SUCCESS if the tensor was destroyed successfully, <br>
 *         DW_INVALID_HANDLE if the given tensor handle is invalid.  <br>
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_getLayoutView(size_t* const offset, size_t* const stride, size_t* const numElements,
                                   const uint32_t* const indices, uint32_t const numIndices,
                                   uint32_t const dimension, dwConstDNNTensorHandle_t const tensorHandle);

/**
 * Retrieves the properties of a dwDNNTensorHandle_t
 *
 * @param[out] properties A pointer to the properties
 * @param[in] tensorHandle A tensor handle
 * @return DW_SUCCESS if the tensor properties are retrieved successfully, <br>
 *         DW_INVALID_HANDLE if the given tensor handle is invalid. <br>
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_getProperties(dwDNNTensorProperties* const properties, dwConstDNNTensorHandle_t const tensorHandle);

/**
 * Retrieves the dwDNNTensor of a dwDNNTensorHandle_t.
 * @note Any modification to the data pointed by the tensor retrieved will modify the content of the
 * original handle
 *
 * @param[out] tensor A pointer to the dwDNNTensor pointer
 * @param[in] tensorHandle A tensor handle
 *
 * @return DW_SUCCESS if the dwDNNTensor is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given tensor pointer or tensor handle is invalid. <br>
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_getTensor(dwDNNTensor* const tensor, dwDNNTensorHandle_t const tensorHandle);

/**
 * Locks the tensor and retrieves pointer to the data with write access.
 * Read access can also be achieved by dwDNNTensor_getTensor().
 * Locking a tensor will block any operation that shall modify the content.
 *
 * @note This function will block if the tensor is locked by another thread.
 *
 * @param[out] data A pointer to the beginning of the tensor content.
 * @param[in] tensorHandle A tensor handle
 * @return DW_SUCCESS if the tensor is successfully locked and data is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given data pointer or tensor handle is invalid.
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_lock(void** const data, dwDNNTensorHandle_t const tensorHandle);

/**
 * Tries to lock the tensor. Returns immediately.
 * If the lock operation is successful, isLocked is set to true, and the data points to the beginning of the
 * content of the tensor.
 * Otherwise, isLocked is set to false and the data points to nullptr.
 *
 * @note If the tensor has already been locked by the same thread, the behavior is undefined.
 *
 * @param[out] isLocked A flag indicating if the lock operation is successful.
 * @param[out] data A pointer to the beginning of the tensor content if lock is successful.
 * @param[in] tensorHandle A tensor handle
 * @return DW_SUCCESS whether the lock is successful or not, <br>
 *         DW_INVALID_ARGUMENT if the given data pointer, isLocked or tensor handle is invalid.
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_tryLock(bool* const isLocked, void** const data, dwDNNTensorHandle_t const tensorHandle);

/**
 * Unlocks the tensor, enabling other threads to lock the tensor and modify the content.
 *
 * @param[in] tensorHandle A tensor handle
 * @return DW_SUCCESS if the tensor is successfully unlocked, <br>
 *         DW_INVALID_ARGUMENT if the tensor handle is invalid.
 */
DW_API_PUBLIC
dwStatus dwDNNTensor_unlock(dwDNNTensorHandle_t const tensorHandle);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_DNN_TENSOR_H_
