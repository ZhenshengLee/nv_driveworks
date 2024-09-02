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

#ifndef DWSHARED_CORE_EXCEPTIONWITHSTACKTRACE_HPP_
#define DWSHARED_CORE_EXCEPTIONWITHSTACKTRACE_HPP_

#include "ExceptionBase.hpp"

#include <driver_types.h>
#include <type_traits>
#include <cuda.h>
#ifdef DW_SHARED_STACK_TRACE_ENABLED
#include "Backtrace.hpp"
#endif // DW_SHARED_STACK_TRACE_ENABLED

#define ASSERT_EXCEPTION_IF(exceptionType, condition, msg) dw::core::assertExceptionIf<exceptionType>(condition, msg " ((" #condition ")==false)");
#define ASSERT_ARGUMENT(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidArgumentException, condition, msg)
#define ASSERT_STATE(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidStateException, condition, msg)

namespace dw
{
namespace core
{

class ExceptionWithStackTrace : public ExceptionBase
{
public:
    explicit ExceptionWithStackTrace(char8_t const* const message)
        : ExceptionBase(message)
    {
        traceStack();
    }

    ~ExceptionWithStackTrace() override;

    //////////////////////////////////////
    // These templated constructors allow direct construction of the message
    // NOTE: constexpr parameters need to be wrapped with strip_constexpr(...). Otherwise, an 'undefined reference'
    //       error for the parameter is shown during build. This is because C++14 does not allow constexpr types to be
    //       passed by reference.
    //
    template <class... Tothers>
    explicit ExceptionWithStackTrace(char8_t const* const whatstr, Tothers&&... others)
        : ExceptionBase(whatstr, std::forward<Tothers>(others)...)
    {
        traceStack();
    }

    char8_t const* stackTrace() const
    {
        return m_stackTrace.c_str();
    }

    /**
    * The maximum depth of a stack.
    */
    using STACK_TRACE_MAX_DEPTH = std::integral_constant<size_t, 40u>;
    /**
    * The maximum length of a symbol name
    */
    using STACK_TRACE_MAX_SYMBOL_LENGTH = std::integral_constant<size_t, 512u>;
    /**
    * The maximum length of a stack trace
    */
    using STACK_TRACE_MAX_LENGTH = std::integral_constant<size_t, STACK_TRACE_MAX_DEPTH::value*(STACK_TRACE_MAX_SYMBOL_LENGTH::value + 1) + 25>;

protected:
    ExceptionWithStackTrace() = default;

    void traceStackDemangleSymbol(char8_t const* const symbolMangled, char8_t* begin_name, char8_t* begin_offset, char8_t* const endOffset);

    StringBuffer<STACK_TRACE_MAX_LENGTH::value> m_stackTrace{};

    void traceStack();
};

/////////////////////////////////////////////////
/// Special types of exceptions
///

/// Buffer is full and cannot fit more data
class BufferFullException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// Buffer is empty
class BufferEmptyException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// Tried to access out of bounds
class OutOfBoundsException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// Some argument passed to the method was invalid
class InvalidArgumentException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// A numerical error occurred (e.g. tried to invert a singular matrix)
class NumericException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// An invalid conversion between types occurred, such as string-to-decimal
class InvalidConversionException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// The state of the object/program does not allow the requested operation.
/// Either it is corrupted (something went wrong) or the call is not allowed
/// in this state (e.g. allocate() called after using the object).
class InvalidStateException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// An unaligned pointer was used for a type that requires alignemnt
class BadAlignmentException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// A function was called that has not been implemented yet
class NotImplementedException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

class EndOfStreamException : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;
};

/// An unexpected cuda error occurred
class CudaException : public ExceptionWithStackTrace
{
public:
    template <class... Tothers>
    explicit CudaException(cudaError_t const error, char8_t const* const whatstr, Tothers&&... others)
        : ExceptionWithStackTrace(whatstr, std::forward<Tothers>(others)...)
        , m_cudaError(error)
    {
        appendError(error);
    }

    CudaException(cudaError_t const error, char8_t const* const msg);
    CudaException(CUresult const error, char8_t const* const msg);

    cudaError_t getCudaError() const { return m_cudaError; }

protected:
    void appendError(cudaError_t const error);

private:
    cudaError_t m_cudaError{};

    /// Map from CUResult to cudaError_t
    /// @note: Only few cuX driver API are enabled in cuda safety build. The list of their return values is limited,
    ///        hence we limit here also only to a subset of errors.
    static cudaError_t mapCUResult(CUresult const error);
};

} // end namespace core
} // end namespace dw

#endif // DWSHARED_CORE_EXCEPTIONWITHSTACKTRACE_HPP_
