/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2016-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_EXCEPTION_HPP_
#define DW_CORE_EXCEPTION_HPP_

#include <dw/core/container/BaseString.hpp>
#include <dw/core/container/Span.hpp>
#include <dw/core/base/Config.h>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/language/cxx14.hpp>
#include <dw/core/logger/Logger.hpp>

#include "Status.h"
#include "Types.h"

#include <initializer_list>
namespace dw
{
namespace core
{
/**
* @brief Driveworks exception with message string and dwStatus but without a stacktrace
*/
class ExceptionWithStatus : public dw::core::ExceptionBase
{
public:
    /// dwStatus enum constants do not exceed this length, when represented as strings.
    // TODO(dwplc): FP - variable STATUS_STRING_MAX_LENGTH is used more than once in Exception.hpp
    // coverity[autosar_cpp14_a0_1_1_violation]
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t STATUS_STRING_MAX_LENGTH = 64;

    // NOTE: constexpr parameters need to be wrapped with strip_constexpr(...). Otherwise, an 'undefined reference'
    //       error for the parameter is shown during build. This is because C++14 does not allow constexpr types to be
    //       passed by reference.
    //
    /// Constructor
    template <class... Tothers>
    ExceptionWithStatus(dwStatus const statusCode, char8_t const* const messageStr, Tothers&&... others) // clang-tidy NOLINT
        : dw::core::ExceptionBase(dwGetStatusName(statusCode), SEPARATOR, messageStr, std::forward<Tothers>(others)...)
        , m_statusCode(statusCode)
        , m_messageBegin(0)
    {
        static_assert(STATUS_STRING_MAX_LENGTH < std::numeric_limits<size_t>::max(), "Operation below may overflow");
        constexpr size_t SEPARATOR_LEN = sizeof(ExceptionWithStatus::SEPARATOR) - 1;
        static_assert(SEPARATOR_LEN < std::numeric_limits<size_t>::max() - STATUS_STRING_MAX_LENGTH, "Operation below may overflow");
        m_messageBegin = strnlen(dwGetStatusName(statusCode), STATUS_STRING_MAX_LENGTH) + SEPARATOR_LEN;
    }
    ExceptionWithStatus()                           = delete;
    ExceptionWithStatus(ExceptionWithStatus const&) = default;
    ExceptionWithStatus(ExceptionWithStatus&&)      = default;
    ExceptionWithStatus& operator=(ExceptionWithStatus const&) = delete;
    ExceptionWithStatus& operator=(ExceptionWithStatus&&) = delete;
    /// Destructor
    ~ExceptionWithStatus() noexcept override = default;
    /// Get exception status
    dwStatus status() const
    {
        return m_statusCode;
    }
    /// Get exception message
    char8_t const* message() const noexcept
    {
        // the message output of an exception shall not throw
        try
        {
            return &m_what.get().at(m_messageBegin);
        }
        catch (std::exception const&)
        {
            // this should never happen. If it does, return entire message
            return m_what.c_str();
        }
    }

private:
    dwStatus m_statusCode;
    size_t m_messageBegin;
    /**
     * Separator between status name and exception message
     */
    // TODO(dwplc): FP - variable SEPARATOR is used more than once in this file
    // coverity[autosar_cpp14_a0_1_1_violation]
    static constexpr char8_t SEPARATOR[] = ": ";
};

/**
* @brief Driveworks exception with message string and dwStatus
*/
class Exception : public dw::core::ExceptionWithStackTrace
{
public:
    // TODO(dwplc): FP - variable LOG_TAG is used more than once in this file in the DW_LOG macro
    // coverity[autosar_cpp14_a0_1_1_violation]
    static constexpr char8_t LOG_TAG[] = "NO_TAG";

    /// dwStatus enum constants do not exceed this length, when represented as strings.
    // TODO(dwplc): FP - variable STATUS_STRING_MAX_LENGTH is used more than once in Exception.hpp
    // coverity[autosar_cpp14_a0_1_1_violation]
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t STATUS_STRING_MAX_LENGTH = 64;

    // NOTE: constexpr parameters need to be wrapped with strip_constexpr(...). Otherwise, an 'undefined reference'
    //       error for the parameter is shown during build. This is because C++14 does not allow constexpr types to be
    //       passed by reference.
    //
    /// Constructor
    template <class... Tothers>
    Exception(dwStatus const statusCode, char8_t const* const messageStr, Tothers&&... others) // clang-tidy NOLINT
        : dw::core::ExceptionWithStackTrace(dwGetStatusName(statusCode), SEPARATOR, messageStr, std::forward<Tothers>(others)...)
        , m_statusCode(statusCode)
        , m_messageBegin(0)
    {
        static_assert(STATUS_STRING_MAX_LENGTH < std::numeric_limits<size_t>::max(), "Operation below may overflow");
        static_assert(SEPARATOR_LEN < std::numeric_limits<size_t>::max() - STATUS_STRING_MAX_LENGTH, "Operation below may overflow");

        m_messageBegin = strnlen(dwGetStatusName(statusCode), STATUS_STRING_MAX_LENGTH) + SEPARATOR_LEN;
        setLastException(statusCode, m_what.c_str());
    }

    Exception() = delete;

    Exception(Exception const&) = default;
    Exception(Exception&&)      = default;
    Exception& operator=(Exception const&) = delete;
    Exception& operator=(Exception&&) = delete;

    /// Destructor
    ~Exception() noexcept override = default;

    /// Get exception status
    dwStatus status() const
    {
        return m_statusCode;
    }

    /// Get exception message
    char8_t const* message() const noexcept
    {
        // the message output of an exception shall not throw
        try
        {
            return &m_what.get().at(m_messageBegin);
        }
        catch (std::exception const&)
        {
            // this should never happen. If it does, return entire message
            return m_what.c_str();
        }
    }

    /// Get exception stacktrace
    char8_t const* stackTrace() const noexcept
    {
        return m_stackTrace.c_str();
    }

    /**
    * Simple try/catch handler to catch dw::core::Exception and return
    * a dwStatus error code if the given function block throws.
    * Signature of tryBlock should be "dwStatus tryBlock()".
    *
    * All errors should be reported through exceptions. If an exception is thrown
    * it will be caught by the guard, stored in a thread-local buffer and it's status
    * code will be returned.
    *
    * A template is used to allow inlining of the tryBlock.
    *
    */
    template <typename TryBlock>
    static dwStatus guardWithReturn(TryBlock const& tryBlock, dw::core::Logger::Verbosity verbosity = dw::core::Logger::Verbosity::ERROR)
    {
        try
        {
            return tryBlock();
        }
        catch (BufferFullException const& ex)
        {
            return logException(DW_BUFFER_FULL, ex, verbosity);
        }
        catch (BufferEmptyException const& ex)
        {
            return logException(DW_NOT_AVAILABLE, ex, verbosity);
        }
        catch (OutOfBoundsException const& ex)
        {
            return logException(DW_OUT_OF_BOUNDS, ex, verbosity);
        }
        catch (InvalidArgumentException const& ex)
        {
            return logException(DW_INVALID_ARGUMENT, ex, verbosity);
        }
        catch (BadAlignmentException const& ex)
        {
            return logException(DW_BAD_ALIGNMENT, ex, verbosity);
        }
        catch (CudaException const& ex)
        {
            return logException(cudaErrorToDWError(ex.getCudaError()), ex, verbosity);
        }
        catch (EndOfStreamException const& ex)
        {
            return logException(DW_END_OF_STREAM, ex, verbosity);
        }
        catch (dw::core::Exception const& ex)
        {
            return logException(ex.status(), ex, verbosity);
        }
        catch (ExceptionWithStackTrace const& ex)
        {
            return logException(DW_FAILURE, ex, verbosity);
        }
        catch (ExceptionWithStatus const& exception)
        {
            //Store a copy
            setLastException(exception.status(), exception.what());

            DW_LOG(verbosity) << "Driveworks exception thrown: "
                              << exception.what()
                              << Logger::State::endl;

            return exception.status();
        }
        catch (std::exception const& exception)
        {
            //Store a copy
            setLastException(DW_FAILURE, exception.what());

            DW_LOG(verbosity) << "std::exception thrown: "
                              << exception.what()
                              << Logger::State::endl;

            return DW_FAILURE;
        }
        catch (...)
        {
            //Store a copy
            setLastException(DW_FAILURE, "Unknown exception");

            DW_LOG(verbosity) << "Unknown exception thrown"
                              << Logger::State::endl;

            return DW_FAILURE;
        }
    }

    /**
    * Same as guardWithReturn but with a simpler tryBlock with signature 'void tryBlock()'
    * Always returns DW_SUCCESS unless an exception is thrown.
    */
    template <typename TryBlock>
    static dwStatus guard(TryBlock const& tryBlock, dw::core::Logger::Verbosity verbosity = dw::core::Logger::Verbosity::ERROR)
    {
        static_assert(std::is_same<void, typename std::result_of<TryBlock()>::type>::value,
                      "tryBlock must return void");
        auto const tryBlockLambda = [&tryBlock]() -> dwStatus {
            tryBlock();
            return DW_SUCCESS;
        };
        return guardWithReturn(tryBlockLambda, verbosity);
    }

    /**
    * Return a copy of the last exception caught by the guard() method
    * and resets the stored exception.
    */
    static void getLastException(dwStatus* const status, char8_t const** const msg);

    /**
     * Convert CUDA error(cudaError_t) to dwStatus
     */
    static dwStatus cudaErrorToDWError(cudaError_t const error);

private:
    dwStatus m_statusCode;
    size_t m_messageBegin;

    /**
     * Separator between status name and exception message
     */
    static constexpr char8_t SEPARATOR[] = ": ";

    /**
     * Length of separator
     */
    static constexpr size_t SEPARATOR_LEN = (sizeof(Exception::SEPARATOR) - 1);

    /**
    * Store last exception to be returned later. The function will copy the message,
    * so the buffer can be reused afterwards.
    **/
    static void setLastException(dwStatus const status, char8_t const* const msg);

    template <typename T, typename = typename std::enable_if<std::is_base_of<ExceptionBase, T>::value>::type>
    static dwStatus logException(dwStatus const status, T const& ex, dw::core::Logger::Verbosity verbosity)
    {
        // Store a copy
        setLastException(status, ex.what());

        DW_LOG(verbosity) << "Driveworks exception thrown: "
                          << ex.what()
                          << Logger::State::endl
                          << ex.stackTrace()
                          << Logger::State::endl;
        return status;
    }
};

} // namespace core
} // namespace dw

////////////////////////////////////////////////////////////////////////////////////////
// Exception throwing macros
#define THROW_IF_PARAM_NULL_MSG(param, msg)                                                                         \
    if ((param) == nullptr)                                                                                         \
    {                                                                                                               \
        /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */                                           \
        throw dw::core::Exception(DW_INVALID_ARGUMENT, __FILE__, ":", __LINE__, " - " #param " == nullptr. ", msg); \
    }

#define THROW_IF_PARAM_NULL(param) THROW_IF_PARAM_NULL_MSG(param, "")

////////////////////////////////////////////////////////////////////////////////////////
// Implementation
namespace dw
{
namespace core
{

/** Raises an exception for either GPU or CPU.
 * In CPU it raises a dw::Exception.
 * In GPU it prints and causes a segfault.
 */
CUDA_BOTH_INLINE void assertException(dwStatus const status, char8_t const* const msg)
{
// clang-format off
    #ifdef __CUDA_ARCH__
        printf("dwException: %d - %s\n", static_cast<int32_t>(status), msg);

        // Cannot throw, force the kernel to fail
        #ifndef NDEBUG
            constexpr bool forceException = false;
            assert(forceException); // Only defined when NDEBUG is not defined
        #else
             __trap();
        #endif
    #else
        throw dw::Exception(status, msg);
    #endif
    // clang-format on
}

/**
 * The method verifies if failedCondition is true and throws an exception in this case with errno and strerror in it's message
 */
void assertSystemCallExceptionIf(bool const failedCondition, dwStatus const status, char8_t const* const msg);

} // namespace core
} // namespace dw

#define DW_THROW_ON_MISMATCH(expected, actual, message)          \
    if ((expected) != (actual))                                  \
    {                                                            \
        throw dw::core::Exception(DW_INVALID_ARGUMENT, message); \
    }

#define DW_THROW_ON_MISMATCH_STACK(expected, actual, message) \
    if ((expected) != (actual))                               \
    {                                                         \
        throw dw::core::ExceptionWithStackTrace(message);     \
    }

#define DW_THROW_ON_ERROR(expected, actual, message) DW_THROW_ON_MISMATCH(expected, actual, message)
#define DW_THROW_ON_ERROR_STACK(expected, actual, message) DW_THROW_ON_MISMATCH_STACK(expected, actual, message)

#define DW_THROW_ON_SAME(_first, _second, message)               \
    if ((_first) == (_second))                                   \
    {                                                            \
        throw dw::core::Exception(DW_INVALID_ARGUMENT, message); \
    }

#endif // DW_CORE_EXCEPTION_HPP_
