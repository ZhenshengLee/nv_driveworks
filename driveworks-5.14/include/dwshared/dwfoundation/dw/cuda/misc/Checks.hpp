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
// SPDX-FileCopyrightText: Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFOUNDATION_DW_CUDA_CHECKS_HPP_
#define DWFOUNDATION_DW_CUDA_CHECKS_HPP_

#include <dwshared/dwfoundation/dw/core/ConfigChecks.h>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>

#include <cuda_runtime.h>

// mellery: making this check unconditional
// since coverity cuda checks generally require
// error checking
//#if DW_RUNTIME_CHECKS()
/**
 * @brief Checks for cuda errors and throws an exception if an error is found.
 *        Can be used for cuda function calls: <tt>DW_CHECK_CUDA_ERROR(cudaMemcpy(...));</tt>
 *
 * @param X CUDA function call results to check
 *
 * @note For kernel calls use #DW_CHECK_CUDA_ERROR_VOID
 *
 */
#define DW_CHECK_CUDA_ERROR(X)                    \
    {                                             \
        static_cast<void>(X);                     \
        dw::cuda::checkError(__FILE__, __LINE__); \
    }

/**
 * @brief Checks for cuda errors and throws an exception if an error is found. Executes no statements.
 *        Useful to check errors after kernel calls: <tt>myKernel<<<1,1>>>(...); DW_CHECK_CUDA_ERROR_VOID();</tt>
 */
#define DW_CHECK_CUDA_ERROR_VOID() dw::cuda::checkError(__FILE__, __LINE__)

namespace dw
{

/**
 * Provides helper methods for cuda code
 */
namespace cuda
{

/*
 * Throws an exception if cudaGetLastError reports an error.
 * Do not call this directly. Use the DW_CHECK_CUDA_ERROR macro instead.
 */
inline void checkError(const char* file, int32_t line)
{
    //Note: to properly check for an error we must sync with the gpu. Otherwise
    //      an error might be reported much later and it is hard to trace back to the
    //      offending kernel. However, this would introduce many syncs and would hurt performance.
    //      Thus, this is commented. Uncomment if you're trying to find the source of a cuda error.
    // code-comment " cudaDeviceSynchronize();"

    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw dw::core::CudaException(err, ". Line: ", file, ":", line);
    }
}

} // namespace cuda
} // namespace dw

#endif
