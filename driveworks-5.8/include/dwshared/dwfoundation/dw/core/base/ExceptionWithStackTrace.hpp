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

#ifndef DW_SHARED_CORE_EXCEPTION_HPP_
#define DW_SHARED_CORE_EXCEPTION_HPP_

#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/container/BaseString.hpp>

#include <exception>
#include <functional>
#include <initializer_list>

#include <driver_types.h>
#include <cuda.h>

#include <dw/core/safety/MathErrors.hpp>
#include <dw/core/language/TypeAliases.hpp>

#define ASSERT_EXCEPTION_IF(exceptionType, condition, msg) dw::core::assertExceptionIf<exceptionType>(condition, msg " ((" #condition ")==false)");
#define ASSERT_ARGUMENT(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidArgumentException, condition, msg)
#define ASSERT_STATE(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidStateException, condition, msg)

namespace dw
{
namespace core
{

///////////////////////////////////////////////////////////

class ExceptionBase : public std::exception
{

protected:
    // MessageInitializer is an auxiliary class that appends a list of arguments to m_what. It statically converts
    // literal strings (char*) to char8_t* to avoid autosar_cpp14_a3_9_1_violation: use of type without size
    template <class W, class... T>
    struct MessageInitializer;

public:
    explicit ExceptionBase(char8_t const* const message)
        : m_what(message)
    {
    }

    ~ExceptionBase() override;

    //////////////////////////////////////
    // The templated constructor allows direct construction of the message
    template <class... Tothers>
    explicit ExceptionBase(char8_t const* const whatstr, Tothers&&... others)
        : ExceptionBase(whatstr)
    {
        MessageInitializer<decltype(m_what), Tothers...>(m_what, std::forward<Tothers>(others)...);
    }

    char8_t const* what() const noexcept override
    {
        return m_what.c_str();
    }

    /**
    * The maximum size of an exception message.
    */
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t WHAT_SIZE = 4096;

protected:
    ExceptionBase() = default;

    FixedString<WHAT_SIZE> m_what{};

    // Generic argument
    template <class W, class T>
    struct AppendToWhat
    {
        AppendToWhat(W& m_what, T&& v)
        {
            static_cast<void>(m_what << v);
        }
    };

    // Consider char* argument as char8_t*
    template <class W>
    struct AppendToWhat<W, char*>
    {
        AppendToWhat(W& m_what, char8_t const*&& v)
        {
            static_cast<void>(m_what << v);
        }
    };

    // Consider char* const argument as char8_t* const
    template <class W>
    struct AppendToWhat<W, char* const>
    {
        AppendToWhat(W& m_what, char8_t const* const&& v)
        {
            static_cast<void>(m_what << v);
        }
    };

    // Expand arguments one by one
    template <class W, class T, class... Tothers>
    struct MessageInitializer<W, T, Tothers...>
    {
        MessageInitializer(W& m_what, T&& first, Tothers&&... others)
        {
            AppendToWhat<W, T>(m_what, std::forward<T>(first));
            MessageInitializer<W, Tothers...>(m_what, std::forward<Tothers>(others)...);
        }
    };

    // Last argument
    template <class W, class T>
    struct MessageInitializer<W, T>
    {
        MessageInitializer(W& m_what, T&& first)
        {
            AppendToWhat<W, T>(m_what, std::forward<T>(first));
        }
    };
};

class ExceptionWithStackTrace : public ExceptionBase
{
public:
    // TODO(dwplc): FP - Declaring a forwarding constructor that is constrained (via SFINAE) to not match any other overloads
    // coverity[autosar_cpp14_a13_3_1_violation]
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
    static constexpr size_t STACK_TRACE_MAX_DEPTH = 40;
    /**
    * The maximum length of a symbol name
    */
    static constexpr size_t STACK_TRACE_MAX_SYMBOL_LENGTH = 512;
    /**
    * The maximum length of a stack trace
    */
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t STACK_TRACE_MAX_LENGTH = STACK_TRACE_MAX_DEPTH * (STACK_TRACE_MAX_SYMBOL_LENGTH + 1) + 25;

private:
    void traceStackDemangleSymbol(char8_t const* const symbolMangled, char8_t* begin_name, char8_t* begin_offset, char8_t* const endOffset);

protected:
    ExceptionWithStackTrace() = default;
    FixedString<STACK_TRACE_MAX_LENGTH> m_stackTrace{};

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

private:
    cudaError_t m_cudaError{};

    void appendError(cudaError_t const error);

    /// Map from CUResult to cudaError_t
    /// @note: Only few cuX driver API are enabled in cuda safety build. The list of their return values is limited,
    ///        hence we limit here also only to a subset of errors.
    static cudaError_t mapCUResult(CUresult const error);
};

/// Raises an exception for either GPU or CPU
/// In CPU it raises a TException(msg)
/// In GPU it prints and causes a segfault
template <class TException = dw::core::ExceptionWithStackTrace>
CUDA_BOTH_INLINE void assertException(char8_t const* const msg) // clang-tidy NOLINT(readability-avoid-const-params-in-decls)
{
// clang-format off
    #ifdef __CUDA_ARCH__
        // TODO: it'd be nice to print the exception type
        // but typeid(TException).name is not supported in device code
        printf("Exception: %s\n", msg);

        // Cannot throw, force the kernel to fail
        #ifndef NDEBUG
            constexpr bool forceException = false;
            assert(forceException); // Only defined when NDEBUG is not defined
        #else
            asm("trap;");
        #endif
    #else
        throw TException(msg);
    #endif
    // clang-format on
}

template <class TException = dw::core::ExceptionWithStackTrace>
CUDA_BOTH_INLINE void assertExceptionIf(bool const condition, char8_t const* const msg)
{
    if (!condition)
    {
        assertException<TException>(msg);
    }
}
}
}
#endif // DW_CORE_EXCEPTION_HPP_
