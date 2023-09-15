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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dw/core/base/Status.h>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

#define THROW_ON_PARAM_NULL(param)                                                                                        \
    if (param == nullptr)                                                                                                 \
    {                                                                                                                     \
        throw dw::core::ExceptionWithStatus(DW_INVALID_ARGUMENT, #param " == nullptr ", DW_FUNCTION_NAME, ":", __LINE__); \
    }

#define GET_STRING(s) #s

//------------------------------------------------------------------------------
// Macro to easily check for DW errors (non-DW_SUCCESS status) and throw exception. Exception message includes file name/line number, actual return code and user-provided description
#define FRWK_CHECK_DW_ERROR_MSG(x, description)                                                                                                                                             \
    {                                                                                                                                                                                       \
        dwStatus FRWK_CHECK_DW_ERROR_result{x};                                                                                                                                             \
        if (FRWK_CHECK_DW_ERROR_result != DW_SUCCESS)                                                                                                                                       \
        {                                                                                                                                                                                   \
            throw dw::core::ExceptionWithStatus(FRWK_CHECK_DW_ERROR_result, __FILE__, ":", __LINE__, " DriveWorks Error ", dwGetStatusName(FRWK_CHECK_DW_ERROR_result), ": ", description); \
        }                                                                                                                                                                                   \
    };
// Macro that includes the failing code line as default description. Use FRWK_CHECK_DW_ERROR_MSG to provide a customized, hand-written description.
#define FRWK_CHECK_DW_ERROR(x) FRWK_CHECK_DW_ERROR_MSG(x, GET_STRING(x));

#define FRWK_CHECK_DW_ERROR_IGNORE_SOME(x, fallback, ...)                                                                                                                                                             \
    {                                                                                                                                                                                                                 \
        dwStatus FRWK_CHECK_DW_ERROR_IGNORE_SOME_result{x};                                                                                                                                                           \
        if (FRWK_CHECK_DW_ERROR_IGNORE_SOME_result != DW_SUCCESS)                                                                                                                                                     \
        {                                                                                                                                                                                                             \
            dwStatus ignoreErros[]{__VA_ARGS__};                                                                                                                                                                      \
            if (std::find(std::begin(ignoreErros), std::end(ignoreErros), FRWK_CHECK_DW_ERROR_IGNORE_SOME_result) != std::end(ignoreErros))                                                                           \
            {                                                                                                                                                                                                         \
                DW_LOGD << __FILE__ << ":" << __LINE__                                                                                                                                                                \
                        << " Ignoring Error: "                                                                                                                                                                        \
                        << dwGetStatusName(FRWK_CHECK_DW_ERROR_IGNORE_SOME_result) << ". Falling back on calling " << GET_STRING(fallback)                                                                            \
                        << dw::core::Logger::State::endl;                                                                                                                                                             \
                FRWK_CHECK_DW_ERROR_IGNORE_SOME_result = fallback;                                                                                                                                                    \
                if (FRWK_CHECK_DW_ERROR_IGNORE_SOME_result != DW_SUCCESS)                                                                                                                                             \
                {                                                                                                                                                                                                     \
                    throw dw::core::ExceptionWithStatus(FRWK_CHECK_DW_ERROR_IGNORE_SOME_result, "After ignoring errors from ignore list, fallback operation %s encountered DriveWorks error.", GET_STRING(fallback)); \
                }                                                                                                                                                                                                     \
            }                                                                                                                                                                                                         \
        }                                                                                                                                                                                                             \
        if (FRWK_CHECK_DW_ERROR_IGNORE_SOME_result != DW_SUCCESS)                                                                                                                                                     \
        {                                                                                                                                                                                                             \
            throw dw::core::ExceptionWithStatus(FRWK_CHECK_DW_ERROR_IGNORE_SOME_result, "DriveWorks error not in ignore list.");                                                                                      \
        }                                                                                                                                                                                                             \
    };

#define FRWK_CHECK_DW_ERROR_NOTHROW(x)                                     \
    {                                                                      \
        dwStatus FRWK_CHECK_DW_ERROR_NOTHROW_result{x};                    \
        if (FRWK_CHECK_DW_ERROR_NOTHROW_result != DW_SUCCESS)              \
        {                                                                  \
            DW_LOGE << __FILE__ << ":" << __LINE__                         \
                    << " DriveWorks exception but not thrown: "            \
                    << dwGetStatusName(FRWK_CHECK_DW_ERROR_NOTHROW_result) \
                    << dw::core::Logger::State::endl;                      \
        }                                                                  \
    };

#define FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME(x, fallback, ...)                                                                               \
    {                                                                                                                                           \
        dwStatus FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME_result{x};                                                                             \
        dwStatus ignoreErros[]{__VA_ARGS__};                                                                                                    \
        if (std::find(std::begin(ignoreErros), std::end(ignoreErros), FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME_result) != std::end(ignoreErros)) \
        {                                                                                                                                       \
            FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME_result = fallback;                                                                          \
        }                                                                                                                                       \
        if (FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME_result != DW_SUCCESS)                                                                       \
        {                                                                                                                                       \
            DW_LOGE << __FILE__ << ":" << __LINE__                                                                                              \
                    << " DriveWorks exception but not thrown: "                                                                                 \
                    << dwGetStatusName(FRWK_CHECK_DW_ERROR_NOTHROW_IGNORE_SOME_result)                                                          \
                    << dw::core::Logger::State::endl;                                                                                           \
        }                                                                                                                                       \
    };

//------------------------------------------------------------------------------
// macro to easily check for cuda errors
#define FRWK_CHECK_CUDA_ERROR(x)                                                                                  \
    {                                                                                                             \
        x;                                                                                                        \
        cudaError_t FRWK_CHECK_CUDA_ERROR_result{cudaGetLastError()};                                             \
        if (FRWK_CHECK_CUDA_ERROR_result != cudaSuccess)                                                          \
        {                                                                                                         \
            throw dw::core::ExceptionWithStatus(DW_CUDA_ERROR, cudaGetErrorString(FRWK_CHECK_CUDA_ERROR_result)); \
        }                                                                                                         \
    };

#define FRWK_CHECK_CUDA_ERROR_NOTHROW(x)                                        \
    {                                                                           \
        x;                                                                      \
        cudaError_t FRWK_CHECK_CUDA_ERROR_NOTHROW_result{cudaGetLastError()};   \
        if (FRWK_CHECK_CUDA_ERROR_NOTHROW_result != cudaSuccess)                \
        {                                                                       \
            DW_LOGE << __FILE__ << ":" << __LINE__                              \
                    << " CUDA error but not thrown: "                           \
                    << cudaGetErrorString(FRWK_CHECK_CUDA_ERROR_NOTHROW_result) \
                    << dw::core::Logger::State::endl;                           \
        }                                                                       \
    };

#define FRWK_CHECK_NVMEDIA_ERROR(e)                                                         \
    {                                                                                       \
        NvMediaStatus FRWK_CHECK_NVMEDIA_ERROR_ret{e};                                      \
        if (FRWK_CHECK_NVMEDIA_ERROR_ret != NVMEDIA_STATUS_OK)                              \
        {                                                                                   \
            throw dw::core::ExceptionWithStatus(DW_NVMEDIA_ERROR, "NvMedia error occured"); \
        }                                                                                   \
    }

namespace dw
{
namespace framework
{

// coverity[autosar_cpp14_a0_1_6_violation]
class ExceptionGuard
{
    ExceptionGuard() = delete;

public:
    template <typename TryBlock>
    static dwStatus guardWithReturn(TryBlock const& tryBlock, ::dw::core::Logger::Verbosity verbosity = ::dw::core::Logger::Verbosity::ERROR)
    {
        return guardWithReturnFunction(tryBlock, verbosity);
    }

    template <typename TryBlock>
    static dwStatus guard(TryBlock const& tryBlock, ::dw::core::Logger::Verbosity verbosity = ::dw::core::Logger::Verbosity::ERROR)
    {
        static_assert(std::is_same<void, typename std::result_of<TryBlock()>::type>::value,
                      "tryBlock must return void");
        dw::core::Function<dwStatus()> tryBlockFunc{[&]() -> dwStatus {
            tryBlock();
            return DW_SUCCESS;
        }};
        return guardWithReturnFunction(tryBlockFunc, verbosity);
    }

private:
    static dwStatus guardWithReturnFunction(dw::core::Function<dwStatus()> const& tryBlock, dw::core::Logger::Verbosity verbosity);
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_EXCEPTION_HPP_
