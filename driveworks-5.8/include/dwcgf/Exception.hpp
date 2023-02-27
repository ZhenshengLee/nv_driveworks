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

#include <dwcgf/logger/Logger.hpp>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/base/Status.h>
#include <dw/core/container/BaseString.hpp>
#include <nvscierror.h>

#define THROW_ON_PARAM_NULL(param)                                                                                   \
    if (param == nullptr)                                                                                            \
    {                                                                                                                \
        throw dw::framework::Exception(DW_INVALID_ARGUMENT, #param " == nullptr ", DW_FUNCTION_NAME, ":", __LINE__); \
    }

namespace dw
{
namespace framework
{

////////////////////////////////////////////////////////////////////////////////////////

class Exception : public dw::core::ExceptionBase
{
public:
    static constexpr char8_t LOG_TAG[] = "Exception";

    Exception(dwStatus statusCode, const char8_t* messageStr)
        : dw::core::ExceptionBase(dwGetStatusName(statusCode))
        , m_statusCode(statusCode)
        , m_messageBegin(0)
    {
        m_what += ": ";
        m_messageBegin = m_what.length();
        m_what += messageStr;
    }

    ~Exception() noexcept override = default;

    //////////////////////////////////////
    // These templated constructors allow direct construction of the message
    template <class... Tothers>
    explicit Exception(dwStatus statusCode, const char8_t* messageStr, Tothers... others)
        : Exception(statusCode, messageStr)
    {
        (void)std::initializer_list<int32_t>{(static_cast<void>(m_what << others), 0)...};
    }

    dwStatus status() const
    {
        return m_statusCode;
    }

    char8_t const* messageStr() const noexcept
    {
        return what() + m_messageBegin;
    }

    /**
    * Simple try/catch handler to catch dw::core::Exception and return
    * a dwStatus error code if the given function block throws.
    * Signature of tryBlock should be "dwStatus tryBlock()".
    *
    * All errors should be reported through exceptions. If an exception is thrown
    * it will be caught by the guard, a log message with the given severity and
    * the exception message print and it's status code will be returned.
    *
    * A template is used to allow inlining of the tryBlock.
    *
    */
    template <typename TryBlock>
    static dwStatus guardWithReturn(TryBlock tryBlock, dw::core::Logger::Verbosity verbosity = dw::core::Logger::Verbosity::DEBUG);

    /**
    * Same as previous guard but with a simpler tryBlock with signature 'void tryBlock()'
    * Always returns DW_SUCCESS unless an exception is thrown.
    */
    template <typename TryBlock>
    static dwStatus guard(TryBlock tryBlock, dw::core::Logger::Verbosity verbosity = dw::core::Logger::Verbosity::DEBUG);

    template <typename TryBlock>
    static dwStatus guardWithNoPrint(TryBlock tryBlock);

private:
    dwStatus m_statusCode;
    size_t m_messageBegin;
};

////////////////////////////////////////////////////////////////////////////////////////
// Cannot compile with NVCC because of the generic lambda
template <typename TryBlock>
dwStatus Exception::guardWithReturn(TryBlock tryBlock, dw::core::Logger::Verbosity verbosity)
{
    using FixedString = dw::core::BaseString<40>;

    // logging exception
    auto const logException = [verbosity](const dwStatus status, const auto& ex, FixedString errorMessage) -> dwStatus {

        // Disabling rule A5-1-1 here as literals are allowed for logging
        // coverity[autosar_cpp14_a5_1_1_violation]
        DW_LOG(verbosity) << errorMessage
                          << dwGetStatusName(status)
                          << ", "
                          << ex.what()
                          << Logger::State::endl
                          << ex.stackTrace()
                          << Logger::State::endl;

        return status;
    };

    try
    {
        return tryBlock();
    }
    catch (const Exception& ex)
    {
        // TODO(lindax): Make this message a warning after RR1.0 fix all their
        // channel problem

        DW_LOG(verbosity) << "Framework exception thrown: "
                          << dwGetStatusName(ex.status())
                          << ", "
                          << ex.what()
                          << Logger::State::endl;
        return ex.status();
    }
    catch (const BufferFullException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]

        return logException(DW_BUFFER_FULL, ex, FixedString("Framework exception thrown: "));
    }
    catch (const BufferEmptyException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_NOT_AVAILABLE, ex, FixedString("Framework exception thrown: "));
    }
    catch (const OutOfBoundsException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_OUT_OF_BOUNDS, ex, FixedString("Framework exception thrown: "));
    }
    catch (const InvalidArgumentException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_INVALID_ARGUMENT, ex, FixedString("Framework exception thrown: "));
    }
    catch (const BadAlignmentException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_BAD_ALIGNMENT, ex, FixedString("Framework exception thrown: "));
    }
    catch (const CudaException& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_CUDA_ERROR, ex, FixedString("Framework exception thrown: "));
    }
    catch (const ExceptionWithStackTrace& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]
        return logException(DW_FAILURE, ex, FixedString("Framework exception thrown: "));
    }
    catch (const std::exception& ex)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]

        DW_LOG(verbosity) << "std::exception thrown: "
                          << dwGetStatusName(DW_FAILURE)
                          << ", "
                          << ex.what()
                          << Logger::State::endl;
        return DW_FAILURE;
    }
    catch (...)
    {
        // Disabling rule A4-5-1 here. Coverity confuses 'status' used as parameter in lambda function with using enum as operand in overloaded "operator ()"
        // coverity[autosar_cpp14_a4_5_1_violation]

        DW_LOG(verbosity) << "Unknown exception thrown, "
                          << dwGetStatusName(DW_FAILURE)
                          << Logger::State::endl;

        return DW_FAILURE;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
template <typename TryBlock>
dwStatus Exception::guard(TryBlock tryBlock, dw::core::Logger::Verbosity verbosity)
{
    static_assert(std::is_same<void, typename std::result_of<TryBlock()>::type>::value,
                  "tryBlock must return void");

    return guardWithReturn([&] {
        tryBlock();
        return DW_SUCCESS;
    },
                           verbosity);
}

////////////////////////////////////////////////////////////////////////////////////////
template <typename TryBlock>
dwStatus Exception::guardWithNoPrint(TryBlock tryBlock)
{
    try
    {
        tryBlock();
        return DW_SUCCESS;
    }
    catch (const Exception& ex)
    {
        return ex.status();
    }
}

const char* nvSciGetEventName(uint32_t event);
const char* nvSciGetErrorName(uint32_t error);

} // namespace framework
} // namespace dw

//------------------------------------------------------------------------------
// macro to easily check for dw errors
#define FRWK_CHECK_DW_ERROR(x)                                                                      \
    {                                                                                               \
        dwStatus result = x;                                                                        \
        if (result != DW_SUCCESS)                                                                   \
        {                                                                                           \
            throw dw::framework::Exception(result, __FILE__, ":", __LINE__, " - DriveWorks Error"); \
        }                                                                                           \
    };
#define GET_STRING(s) #s
#define FRWK_CHECK_DW_ERROR_IGNORE_SOME(x, fallback, ...)                                                                                                                        \
    {                                                                                                                                                                            \
        dwStatus result        = x;                                                                                                                                              \
        dwStatus ignoreErros[] = {__VA_ARGS__};                                                                                                                                  \
        if (result != DW_SUCCESS)                                                                                                                                                \
        {                                                                                                                                                                        \
            if (std::find(std::begin(ignoreErros), std::end(ignoreErros), result) != std::end(ignoreErros))                                                                      \
            {                                                                                                                                                                    \
                DW_LOGD << __FILE__                                                                                                                                              \
                        << "(" << __LINE__ << ") "                                                                                                                               \
                        << "Ignoring Error: "                                                                                                                                    \
                        << dwGetStatusName(result) << ". Falling back on calling " << GET_STRING(fallback)                                                                       \
                        << dw::core::Logger::State::endl;                                                                                                                        \
                result = fallback;                                                                                                                                               \
                if (result != DW_SUCCESS)                                                                                                                                        \
                {                                                                                                                                                                \
                    throw dw::framework::Exception(result, "After ignoring errors from ignore list, fallback operation %s encountered DriveWorks error.", GET_STRING(fallback)); \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
        }                                                                                                                                                                        \
        if (result != DW_SUCCESS)                                                                                                                                                \
        {                                                                                                                                                                        \
            throw dw::framework::Exception(result, "DriveWorks error not in ignore list.");                                                                                      \
        }                                                                                                                                                                        \
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

#define FRWK_CHECK_DW_ERROR_MSG(x, description)                    \
    {                                                              \
        dwStatus result = (x);                                     \
        if (result != DW_SUCCESS)                                  \
        {                                                          \
            throw dw::framework::Exception(result, (description)); \
        }                                                          \
    };

//------------------------------------------------------------------------------
// macro to easily check for cuda errors
#define FRWK_CHECK_CUDA_ERROR(x)                                                       \
    {                                                                                  \
        x;                                                                             \
        auto result = cudaGetLastError();                                              \
        if (result != cudaSuccess)                                                     \
        {                                                                              \
            throw dw::framework::Exception(DW_CUDA_ERROR, cudaGetErrorString(result)); \
        }                                                                              \
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

#define FRWK_CHECK_NVMEDIA_ERROR(e)                                                    \
    {                                                                                  \
        auto FRWK_CHECK_NVMEDIA_ERROR_ret = (e);                                       \
        if (FRWK_CHECK_NVMEDIA_ERROR_ret != NVMEDIA_STATUS_OK)                         \
        {                                                                              \
            throw dw::framework::Exception(DW_NVMEDIA_ERROR, "NvMedia error occured"); \
        }                                                                              \
    }

#define FRWK_CHECK_NVSCI_ERROR(e)                                                      \
    {                                                                                  \
        auto FRWK_CHECK_NVSCI_ERROR_ret = (e);                                         \
        if (FRWK_CHECK_NVSCI_ERROR_ret != NvSciError_Success)                          \
        {                                                                              \
            DW_LOGE << "Failed with " << nvSciGetErrorName(FRWK_CHECK_NVSCI_ERROR_ret) \
                    << "(" << FRWK_CHECK_NVSCI_ERROR_ret << ")"                        \
                    << " in " << __FILE__                                              \
                    << ":" << __LINE__ << Logger::State::endl;                         \
            if (FRWK_CHECK_NVSCI_ERROR_ret == NvSciError_Timeout)                      \
                throw Exception(DW_TIME_OUT, "NvSci API Timeout");                     \
            else                                                                       \
                throw Exception(DW_INTERNAL_ERROR, "NvSci internal error occured");    \
        }                                                                              \
    }

#endif // DW_FRAMEWORK_TYPES_HPP_
