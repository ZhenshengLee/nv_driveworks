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

#ifndef DW_FRAMEWORK_EXCEPTION_HPP_
#define DW_FRAMEWORK_EXCEPTION_HPP_

#include <dw/core/base/Exception.hpp>
#include <dw/core/base/Status.h>
#include <dw/core/container/BaseString.hpp>
#include <dw/core/logger/Logger.hpp>

#define THROW_ON_PARAM_NULL(param)                                                                                        \
    if (param == nullptr)                                                                                                 \
    {                                                                                                                     \
        throw dw::core::ExceptionWithStatus(DW_INVALID_ARGUMENT, #param " == nullptr ", DW_FUNCTION_NAME, ":", __LINE__); \
    }

//------------------------------------------------------------------------------
// macro to easily check for dw errors
#define FRWK_CHECK_DW_ERROR(x)                                                                           \
    {                                                                                                    \
        dwStatus result{};                                                                               \
        result = (x);                                                                                    \
        if (result != DW_SUCCESS)                                                                        \
        {                                                                                                \
            throw dw::core::ExceptionWithStatus(result, __FILE__, ":", __LINE__, " - DriveWorks Error"); \
        }                                                                                                \
    };
#define GET_STRING(s) #s
#define FRWK_CHECK_DW_ERROR_IGNORE_SOME(x, fallback, ...)                                                                                                                             \
    {                                                                                                                                                                                 \
        dwStatus result        = x;                                                                                                                                                   \
        dwStatus ignoreErros[] = {__VA_ARGS__};                                                                                                                                       \
        if (result != DW_SUCCESS)                                                                                                                                                     \
        {                                                                                                                                                                             \
            if (std::find(std::begin(ignoreErros), std::end(ignoreErros), result) != std::end(ignoreErros))                                                                           \
            {                                                                                                                                                                         \
                DW_LOGD << __FILE__                                                                                                                                                   \
                        << "(" << __LINE__ << ") "                                                                                                                                    \
                        << "Ignoring Error: "                                                                                                                                         \
                        << dwGetStatusName(result) << ". Falling back on calling " << GET_STRING(fallback)                                                                            \
                        << dw::core::Logger::State::endl;                                                                                                                             \
                result = fallback;                                                                                                                                                    \
                if (result != DW_SUCCESS)                                                                                                                                             \
                {                                                                                                                                                                     \
                    throw dw::core::ExceptionWithStatus(result, "After ignoring errors from ignore list, fallback operation %s encountered DriveWorks error.", GET_STRING(fallback)); \
                }                                                                                                                                                                     \
            }                                                                                                                                                                         \
        }                                                                                                                                                                             \
        if (result != DW_SUCCESS)                                                                                                                                                     \
        {                                                                                                                                                                             \
            throw dw::core::ExceptionWithStatus(result, "DriveWorks error not in ignore list.");                                                                                      \
        }                                                                                                                                                                             \
    };

#define FRWK_CHECK_DW_ERROR_NOTHROW(x)                         \
    {                                                          \
        dwStatus result = x;                                   \
        if (result != DW_SUCCESS)                              \
        {                                                      \
            DW_LOGE << __FILE__                                \
                    << "(" << __LINE__ << ") "                 \
                    << "DriveWorks exception but not thrown: " \
                    << dwGetStatusName(result)                 \
                    << dw::core::Logger::State::endl;          \
        }                                                      \
    };

#define FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME(x, fallback, ...)                                       \
    {                                                                                                   \
        dwStatus result        = x;                                                                     \
        dwStatus ignoreErros[] = {__VA_ARGS__};                                                         \
        if (std::find(std::begin(ignoreErros), std::end(ignoreErros), result) != std::end(ignoreErros)) \
        {                                                                                               \
            result = fallback;                                                                          \
        }                                                                                               \
        if (result != DW_SUCCESS)                                                                       \
        {                                                                                               \
            DW_LOGE << __FILE__                                                                         \
                    << "(" << __LINE__ << ") "                                                          \
                    << "DriveWorks exception but not thrown: "                                          \
                    << dwGetStatusName(result)                                                          \
                    << dw::core::Logger::State::endl;                                                   \
        }                                                                                               \
    };

#define FRWK_CHECK_DW_ERROR_MSG(x, description)                         \
    {                                                                   \
        dwStatus result{};                                              \
        result = (x);                                                   \
        if (result != DW_SUCCESS)                                       \
        {                                                               \
            throw dw::core::ExceptionWithStatus(result, (description)); \
        }                                                               \
    };

//------------------------------------------------------------------------------
// macro to easily check for cuda errors
#define FRWK_CHECK_CUDA_ERROR(x)                                                            \
    {                                                                                       \
        x;                                                                                  \
        auto result = cudaGetLastError();                                                   \
        if (result != cudaSuccess)                                                          \
        {                                                                                   \
            throw dw::core::ExceptionWithStatus(DW_CUDA_ERROR, cudaGetErrorString(result)); \
        }                                                                                   \
    };

#define FRWK_CHECK_CUDA_ERROR_NOTHROW(x)              \
    {                                                 \
        x;                                            \
        auto result = cudaGetLastError();             \
        if (result != cudaSuccess)                    \
        {                                             \
            DW_LOGE << __FILE__                       \
                    << "(" << __LINE__ << ") "        \
                    << "CUDA error but not thrown: "  \
                    << cudaGetErrorString(result)     \
                    << dw::core::Logger::State::endl; \
        }                                             \
    };

#define FRWK_CHECK_NVMEDIA_ERROR(e)                                                         \
    {                                                                                       \
        auto FRWK_CHECK_NVMEDIA_ERROR_ret = (e);                                            \
        if (FRWK_CHECK_NVMEDIA_ERROR_ret != NVMEDIA_STATUS_OK)                              \
        {                                                                                   \
            throw dw::core::ExceptionWithStatus(DW_NVMEDIA_ERROR, "NvMedia error occured"); \
        }                                                                                   \
    }

#endif // DW_FRAMEWORK_TYPES_HPP_
