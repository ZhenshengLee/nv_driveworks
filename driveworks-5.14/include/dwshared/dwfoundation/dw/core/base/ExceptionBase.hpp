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

#ifndef DW_SHARED_CORE_EXCEPTION_BASE_HPP_
#define DW_SHARED_CORE_EXCEPTION_BASE_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>

#include <exception>
#include <functional>
#include <initializer_list>
#include <type_traits>

#include "TypeAliases.hpp"
#include "StringBuffer.hpp"

namespace dw
{
namespace core
{

// forward declarations
class ExceptionWithStackTrace;

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
        : std::exception()
    {
        m_what += message;
    }

    template <size_t N>
    explicit ExceptionBase(StringBuffer<N> const& message)
    {
        static_assert(N < WHAT_SIZE::value, "Will be truncated, adjust caller code");
        m_what += message.c_str();
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
    using WHAT_SIZE = std::integral_constant<size_t, 4096u>;

protected:
    ExceptionBase() = default;

    StringBuffer<WHAT_SIZE::value> m_what{};

    // Generic argument
    template <class W, class T>
    struct AppendToWhat
    {
        AppendToWhat(W& m_what, T&& v)
        {
            m_what += v;
        }
    };

    // Consider char* argument as char8_t*
    template <class W>
    struct AppendToWhat<W, char*>
    {
        AppendToWhat(W& m_what, char8_t const*&& v)
        {
            m_what += v;
        }
    };

    // Consider char* const argument as char8_t* const
    template <class W>
    struct AppendToWhat<W, char* const>
    {
        AppendToWhat(W& m_what, char8_t const* const&& v)
        {
            m_what += v;
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

/** Raises an exception for either GPU or CPU.
 * In CPU it raises a dw::Exception.
 * In GPU it prints and causes a segfault.
 */
template <class TException = dw::core::ExceptionWithStackTrace>
CUDA_BOTH_INLINE constexpr void assertException(char8_t const* const msg) // clang-tidy NOLINT(readability-avoid-const-params-in-decls)
{
// clang-format off
    #ifdef __CUDA_ARCH__
        // TODO: it'd be nice to print the exception type
        // but typeid(TException).name is not supported in device code
    #ifndef DW_IS_SAFETY
        printf("Exception: %s\n", msg);
    #endif
        // Cannot throw, force the kernel to fail
        #ifndef NDEBUG
            constexpr bool forceException = false;
            assert(forceException); // Only defined when NDEBUG is not defined
        #else
            __trap(); // asm("trap;");
        #endif
    #else
        throw TException(msg);
    #endif
    // clang-format on
}

template <typename StatusCodeT, class TExceptionWithCode>
CUDA_BOTH_INLINE constexpr void assertException(StatusCodeT const status, char8_t const* const msg)
{
// clang-format off
    #ifdef __CUDA_ARCH__
        // TODO: it'd be nice to print the exception type
        // but typeid(TException).name is not supported in device code
    #ifndef DW_IS_SAFETY
        printf("Exception: %d - %s\n", static_cast<int32_t>(status), msg);
    #endif
        // Cannot throw, force the kernel to fail
        #ifndef NDEBUG
            constexpr bool forceException = false;
            assert(forceException); // Only defined when NDEBUG is not defined
        #else
            __trap(); // asm("trap;");
        #endif
    #else
        throw TExceptionWithCode(status, msg);
    #endif
    // clang-format on
}

template <class TException = dw::core::ExceptionWithStackTrace>
CUDA_BOTH_INLINE constexpr void assertExceptionIf(bool const condition, char8_t const* const msg)
{
    if (!condition)
    {
        assertException<TException>(msg);
    }
}

/// A string buffer properly-sized for use in the @ref errorString function. The size of this buffer is the same as what
/// is used by the GNU C library (but is not part of the public API). While it is probably overkill, it is sufficient to
/// prevent the XSI @c strerror_r from failing with @c ERANGE.
///
/// @see errorString
using ErrorStringBuffer = StringBuffer<1024U>;

/**
 * @brief Get a string representation of the system @a errnum, potentially using @a buf to store the representation.
 *
 * @param errnum The error code to get a representation of.
 * @param buf A string buffer to potentially store the representation in. On GNU platforms (such as Linux), this buffer
 *           will be unused if @a errnum is a known value like @c EINVAL -- the result will be a pointer to a table of
 *           known error strings.
 * @return char8_t const*  A string representation of the error. This will always be null-terminated.
 */
char8_t const* errorString(std::int32_t const errnum, ErrorStringBuffer& buf);

} // end namespace core
} // end namespace dw
#endif // DW_SHARED_CORE_EXCEPTION_BASE_HPP_
