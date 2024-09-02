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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_EXCEPTION_WITH_STATUS_HPP_
#define DWSHARED_CORE_EXCEPTION_WITH_STATUS_HPP_

#include "ExceptionBase.hpp"

namespace dw
{
namespace core
{

/**
* @brief An exception with message string and a status but without a stacktrace
*/
template <typename StatusCodeT, typename StatusCodeToStrFunc>
class ExceptionWithStatusCode : public ExceptionBase
{
public:
    /// Status constants do not exceed this length, when represented as strings.
    static constexpr size_t statusStringMaxLength()
    {
        // using a function instead of a variable to avoid requirement for a defintion
        // of the member variable, because in c++14 (used) there is no support for `inline static`
        return 64;
    }

    /// Separator between status name and exception message
    static constexpr std::array<char8_t, 3> separator()
    {
        // using a function instead of a variable to avoid requirement for a defintion
        // of the member variable, because in c++14 (used) there is no support for `inline static`
        return {':', ' ', '\0'};
    }

    // NOTE: constexpr parameters need to be wrapped with strip_constexpr(...). Otherwise, an 'undefined reference'
    //       error for the parameter is shown during build. This is because C++14 does not allow constexpr types to be
    //       passed by reference.
    //
    /// Constructor
    template <class... Tothers>
    ExceptionWithStatusCode(StatusCodeT const statusCode, char8_t const* const messageStr, Tothers&&... others) // clang-tidy NOLINT
        : dw::core::ExceptionBase(StatusCodeToStrFunc{}(statusCode).c_str(), separator().data(), messageStr, std::forward<Tothers>(others)...)
        , m_statusCode(statusCode)
        , m_messageBegin(0)
    {
        static_assert(statusStringMaxLength() + separator().size() - 1 < ExceptionBase::WHAT_SIZE::value, "Operation below may overflow");
        m_messageBegin = strnlen(StatusCodeToStrFunc{}(statusCode).c_str(), statusStringMaxLength()) + strnlen(separator().data(), 2);
    }

    ExceptionWithStatusCode()                               = delete;
    ExceptionWithStatusCode(ExceptionWithStatusCode const&) = default;
    ExceptionWithStatusCode(ExceptionWithStatusCode&&)      = default;
    ExceptionWithStatusCode& operator=(ExceptionWithStatusCode const&) = delete;
    ExceptionWithStatusCode& operator=(ExceptionWithStatusCode&&) = delete;
    /// Destructor
    ~ExceptionWithStatusCode() noexcept override = default;

    /// Return actual status code
    StatusCodeT statusCode() const
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

    /// An intermediate object enabling implicit conversion of ExceptionWithStatusCode::StatusCodeT to a target status type
    struct StatusT
    {
        StatusCodeT status;
        explicit StatusT(StatusCodeT code)
            : status(code)
        {
        }

        /// implicit conversion to a target type, to avoid an explicit call to the cast operator
        template <typename TargetStatusType>
        operator TargetStatusType() const // clang-tidyc NOLINT(google-explicit-constructor)
        {
            return static_cast<TargetStatusType>(status);
        }

        template <typename TargetStatusType>
        bool operator==(TargetStatusType const& rhs) const
        {
            return static_cast<TargetStatusType>(status) == rhs;
        }
    };

    /// Get exception status.
    /// @note The method returns a castable object, which can be casted to the actual status type.
    ///       The status type has to support static_cast from ExceptionWithStatusCode::StatusCodeT
    StatusT status() const
    {
        return StatusT(statusCode());
    }

private:
    StatusCodeT m_statusCode;
    size_t m_messageBegin;
};

namespace detail
{
/// @brief Helper method to convert an integer to a string
struct IntToStr
{
    StringBuffer<32> operator()(int32_t i);
};
};

//! Default exception with status working on int32_t type
using ExceptionWithStatus = ExceptionWithStatusCode<int32_t, detail::IntToStr>;
}
}
#endif // DWSHARED_CORE_EXCEPTION_WITH_STATUS_HPP_
