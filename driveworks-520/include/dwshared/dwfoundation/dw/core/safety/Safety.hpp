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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Utility functions to adhere to safety standard requirements (MISRA / AUTOSAR etc.)
 */

#ifndef DW_CORE_SAFETY_SAFETY_HPP_
#define DW_CORE_SAFETY_SAFETY_HPP_

#include <cstdint>
#include <dwshared/dwfoundation/dw/core/language/CheckedIntegerCast.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/language/Math.hpp>
#include <dwshared/dwfoundation/dw/core/language/StaticIf.hpp>
#include <dwshared/dwfoundation/dw/core/meta/Conjunction.hpp>
#include <dwshared/dwfoundation/dw/core/meta/TypeIdentity.hpp>
#include <dwshared/dwfoundation/dw/core/language/Optional.hpp>
#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>
#include <dwshared/dwfoundation/dw/core/language/Limits.hpp>

#include "impl/SafetyException.hpp"
#include "MathErrors.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <type_traits>

namespace dw
{
namespace core
{
// The 'SafetyResult' type is returned from all failable 'safe*' functions - emtpy values indicates a
// failed execution that needs to be treated accordingly on the call-site.
// Accessing an empty 'SafetyResult' will result in an exception of type 'BadSafetyResultAccess'
// indicating the encountered issue

enum class SafetyIssueType : uint8_t
{
// Prevent 3rdparties (EGL) from steeling these common names
#ifdef None
#undef None
#endif
    None,
    UnspecificError,
    DomainError,
    RangeError,
    PoleError,
    OverflowError,
    UnderflowError,
    PrecisionLossError,
};

// move getErrorMessageForIssue from cpp to hpp, otherwise compiling .cu file will get Unresolved extern function error
/// Get the exception error message associate to the given @a code.
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE char8_t const* getErrorMessageForIssue(SafetyIssueType const code)
{
    char8_t const* result{nullptr};
    switch (code)
    {
    case SafetyIssueType::None:
        result = "No error";
        break;
    case SafetyIssueType::UnspecificError:
        result = "Bad access of safety result (unspecific error)";
        break;
    case SafetyIssueType::DomainError:
        result = "Bad access of safety result (domain error)";
        break;
    case SafetyIssueType::RangeError:
        result = "Bad access of safety result (range error)";
        break;
    case SafetyIssueType::PoleError:
        result = "Bad access of safety result (pole error)";
        break;
    case SafetyIssueType::OverflowError:
        result = "Bad access of safety result (overflow error)";
        break;
    case SafetyIssueType::UnderflowError:
        result = "Bad access of safety result (underflow error)";
        break;
    case SafetyIssueType::PrecisionLossError:
        result = "Bad access of safety result (precision loss error)";
        break;
    default:
        result = "Bad access of safety result (undefined error)";
        break;
    }

    return result;
}

template <class T>
class SafetyResult : public BaseOptional<T, SafetyResult>
{
    // Use inherited constructors
    using Base = BaseOptional<T, SafetyResult>;
    using Base::Base;

public:
    using IssueType = SafetyIssueType;

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE SafetyResult() // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
        : SafetyResult(IssueType::None)
    {
    }

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE constexpr SafetyResult(NullOpt) noexcept // clang-tidy NOLINT(google-explicit-constructor)
        : Base(), m_issueType(IssueType::None)
    {
    }

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE explicit SafetyResult(const BaseOptional<T, SafetyResult>& other)
        : Base(other), m_issueType(other.m_issueType) {}

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE explicit SafetyResult(Base&& other)
        : Base(other), m_issueType(other.m_issueType) {}

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE SafetyResult(const T& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
        : Base(valueIn), m_issueType(IssueType::None)
    {
    }

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE SafetyResult(T&& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
        : Base(std::move(valueIn)), m_issueType(IssueType::None)
    {
    }

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE explicit SafetyResult(IssueType const issueType)
        : Base(), m_issueType(issueType) {}

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void handleInvalidAccess() const
    {
        if (m_issueType == IssueType::None)
        {
            core::assertException<BadOptionalAccess>(BadOptionalAccess::MESSAGE);
        }
        else
        {
            core::assertException<BadSafetyResultAccess>(getErrorMessageForIssue(m_issueType));
        }
    }

    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto getIssueType() const -> IssueType { return m_issueType; }
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE bool isSafe() const { return Base::has_value() && m_issueType == IssueType::None; }

    // Decay a SafetyResult to a regular Optional, keeping the result value but dropping the IssueType
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto toOptional() const -> dw::Optional<T>
    {
        return isSafe() ? dw::Optional<T>{Base::value()} : dw::NULLOPT;
    }

private:
    IssueType m_issueType;
};

// --------------------------------------------------------------------------------------------------------

namespace detail
{

template <typename Ta,
          typename Tb,
          bool uTa = std::is_unsigned<Ta>::value,
          bool uTb = std::is_unsigned<Tb>::value,
          typename = std::enable_if_t<true>>
struct Add
{
    static auto impl(Ta a, Tb b) -> SafetyResult<Ta>;
};

// unsigned = unsigned + unsigned
template <typename Ta, typename Tb>
struct Add<Ta, Tb, true, true>
{
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        Ta const left{checkedIntegerCast<Ta>(core::numeric_limits<Ta>::max() - a)};
        if (b >= left && a != static_cast<Ta>(0) && b != static_cast<Tb>(0))
        {
            return SafetyResult<Ta>(SafetyIssueType::OverflowError);
        }
        // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
        // coverity[cert_int30_c_violation] FP: nvbugs/4536859
        return SafetyResult<Ta>(checkedIntegerCast<Ta>(a + b));
    }
};

// unsigned = unsigned + signed
template <typename Ta, typename Tb>
struct Add<Ta, Tb, true, false>
{
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        Ta const left{checkedIntegerCast<Ta>(core::numeric_limits<Ta>::max() - a)};
        // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
        // coverity[cert_int31_c_violation] FP: nvbugs/4536859
        Ta const absB{static_cast<Ta>(std::abs(b))};
        if (b > 0)
        {
            if (absB > left)
            {
                return SafetyResult<Ta>(SafetyIssueType::OverflowError);
            }
            // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
            // coverity[cert_int30_c_violation] FP: nvbugs/4536859
            return SafetyResult<Ta>(checkedIntegerCast<Ta>(a + absB));
        }
        else
        {
            if (absB > a)
            {
                return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
            }
            return SafetyResult<Ta>(checkedIntegerCast<Ta>(a - absB));
        }
    }
};

// unsigned = signed + unsigned
template <typename Ta, typename Tb>
struct Add<Ta, Tb, false, true>
{
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Tb> { return Add<Tb, Ta>::impl(b, a); }
};

// signed = signed + signed
template <typename Ta, typename Tb>
struct Add<Ta, Tb, false, false, std::enable_if_t<sizeof(Ta) < sizeof(Tb)>>
{
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        if (a > 0)
        {
            if (b > 0)
            {
                if (b > core::numeric_limits<Ta>::max() - a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::OverflowError);
                }
                // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
                // coverity[autosar_cpp14_m5_0_6_violation] FP: nvbugs/4536859
                return SafetyResult<Ta>(a + b);
            }
            else // b <= 0
            {
                // Tb is wider than Ta so underflow must be checked.
                if (b - core::numeric_limits<Ta>::lowest() < a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
                }
                // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
                // coverity[autosar_cpp14_m5_0_6_violation] FP: nvbugs/4536859
                return SafetyResult<Ta>(a + b);
            }
        }
        else // a <= 0
        {
            if (b > 0)
            {
                // Tb is wider than Ta so overflow must be checked.
                if (b > core::numeric_limits<Ta>::max() && b - core::numeric_limits<Ta>::max() > std::abs(a))
                {
                    return SafetyResult<Ta>(SafetyIssueType::OverflowError);
                }
                // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
                // coverity[autosar_cpp14_m5_0_6_violation] FP: nvbugs/4536859
                return SafetyResult<Ta>(a + b);
            }
            else // b <= 0
            {
                if (b < core::numeric_limits<Ta>::lowest() - a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
                }
                // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
                // coverity[autosar_cpp14_m5_0_6_violation] FP: nvbugs/4536859
                return SafetyResult<Ta>(a + b);
            }
        }
    }
};

// signed = signed + signed
// sizeof(Ta) >= sizeof(Tb)
template <typename Ta, typename Tb>
struct Add<Ta, Tb, false, false, std::enable_if_t<sizeof(Ta) >= sizeof(Tb)>>
{
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        if (a > 0)
        {
            if (b > 0)
            {
                if (b > core::numeric_limits<Ta>::max() - a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::OverflowError);
                }
                return SafetyResult<Ta>(a + b);
            }
            else // b <= 0
            {
                // Since Ta is wider than Tb, no underflow.
                return SafetyResult<Ta>(a + b);
            }
        }
        else // a <= 0
        {
            if (b > 0)
            {
                // Ta is wider than Tb, so no underflow.
                return SafetyResult<Ta>(a + b);
            }
            else // b <= 0
            {
                if (b < core::numeric_limits<Ta>::lowest() - a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
                }
                return SafetyResult<Ta>(a + b);
            }
        }
    }
};
// constexpr as required for compile-time block-size computations
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool withinRange(T v, T lo, T hi)
{
    return (v >= lo) && (v <= hi);
}
/// std::clamp() will arrive with C++17, until then:
/// clamps a value within a range
template <class T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto clamp(const T& v, const T& lo, const T& hi) -> const T
{
    return core::min(hi, core::max(v, lo));
}
/// Helper struct to determine the type of the expression a/b
template <class A, class B>
struct DivideResult
{
    // TODO(dwplc): FP - basic numerical type "int" and "unsigned int" aren't used here
    // coverity[autosar_cpp14_a3_9_1_violation] FP: nvbugs/4536859
    using type = decltype(std::declval<A>() / std::declval<B>());
};

template <typename TInt, std::enable_if_t<std::is_integral<TInt>::value>* = nullptr>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable() noexcept;

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<uint64_t>() noexcept
{
    // Actual maximum value of uint64_t (18446744073709551615) cannot be represented as float64_t. In
    // fact, all integers in range [18446744073709550592, 18446744073709551616] are represented as
    // 18446744073709551616, which is greater than max int64_t. Therefore, we use the next lowest
    // accurately representable integer 18446744073709549568 as a bound.
    constexpr float64_t MAX_UINT64_AS_FLOAT{static_cast<float64_t>(18446744073709549568U)};
    return MAX_UINT64_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<int64_t>() noexcept
{
    // Actual maximum value of int64_t (9223372036854775807) cannot be represented as float64_t. In
    // fact, all integers in range [9223372036854775296, 9223372036854775808] are represented as
    // 9223372036854775808, which is greater than max int64_t. Therefore, we use the next lowest
    // accurately representable integer 9223372036854774784 as a bound.
    // coverity[cert_flp36_c_violation] FP: nvbugs/4492201
    constexpr float64_t MAX_INT64_AS_FLOAT{static_cast<float64_t>(9223372036854774784)};
    return MAX_INT64_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<uint32_t>() noexcept
{
    // Maximum 32 bit unsigned integer represented as a floating point number.
    constexpr float64_t MAX_UINT32_AS_FLOAT{4294967295.F};
    return MAX_UINT32_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<int32_t>() noexcept
{
    // Maximum 32 bit signed integer represented as a floating point number.
    constexpr float64_t MAX_INT32_AS_FLOAT{2147483647.F};
    return MAX_INT32_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<uint16_t>() noexcept
{
    // Maximum 16 bit unsigned integer represented as a floating point number.
    constexpr float64_t MAX_UINT16_AS_FLOAT{65535.F};
    return MAX_UINT16_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<int16_t>() noexcept
{
    // Maximum 16 bit signed integer represented as a floating point number.
    constexpr float64_t MAX_INT16_AS_FLOAT{32767.F};
    return MAX_INT16_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<uint8_t>() noexcept
{
    // Maximum 8 bit unsigned integer represented as a floating point number.
    constexpr float64_t MAX_UINT8_AS_FLOAT{255.F};
    return MAX_UINT8_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t maxFloatRepresentable<int8_t>() noexcept
{
    // Maximum 8 bit signed integer represented as a floating point number.
    constexpr float64_t MAX_INT8_AS_FLOAT{127.F};
    return MAX_INT8_AS_FLOAT;
}

template <typename TInt, std::enable_if_t<std::is_integral<TInt>::value && std::is_signed<TInt>::value>* = nullptr>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable() noexcept;

template <>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable<int64_t>() noexcept
{
    // Minimum 64 bit signed integer represented as a floating point number.
    constexpr float64_t MIN_INT64_AS_FLOAT{-9223372036854775808.F};
    return MIN_INT64_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable<int32_t>() noexcept
{
    // Minimum 32 bit signed integer represented as a floating point number.
    constexpr float64_t MIN_INT32_AS_FLOAT{-2147483648.F};
    return MIN_INT32_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable<int16_t>() noexcept
{
    // Minimum 16 bit signed integer represented as a floating point number.
    constexpr float64_t MIN_INT16_AS_FLOAT{-32768.F};
    return MIN_INT16_AS_FLOAT;
}

template <>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable<int8_t>() noexcept
{
    // Minimum 8 bit signed integer represented as a floating point number.
    constexpr float64_t MIN_INT8_AS_FLOAT{-128.F};
    return MIN_INT8_AS_FLOAT;
}

template <typename TInt, std::enable_if_t<std::is_integral<TInt>::value && std::is_unsigned<TInt>::value>* = nullptr>
CUDA_BOTH_INLINE constexpr float64_t minFloatRepresentable() noexcept
{
    return 0.F;
}

} // namespace detail

/**
 * Add two values together in a safe AUTOSAR/MISRA compliant way to avoid integer over- or under-flow.
 *
 * @return SafetyResult result of a+b with the type consistent with the implicit type promotion of 'a+b'
 * (e.g., unsigned type if adding a signed and unsigned integer)
 *
 * TODO(atevs) revisit the function definition when we get C++17's if constexpr.
 **/
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAdd(const Ta a, const Tb b) -> decltype(detail::Add<Ta, Tb>::impl(a, b))
{
    return detail::Add<Ta, Tb>::impl(a, b);
}
/**
 * Safe increment. Throws if increment would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void safeIncrement(Ta& var, const Tb incr)
{
    var = safeAdd(var, incr).value();
}
/**
 * Safe Substract.
 * Throws InvalidArgumentException if 'b' cannnot be negated into its corresponding signed type.
 * Throws BadSafetyResultAccess if 'a-b' would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSub(const Ta a, const uint8_t b) -> decltype(detail::Add<Ta, typename std::make_signed<decltype(b)>::type>::impl(a, b))
{
    // Handle cert_str34_c_violation "Casting character to larger integer size directly. -- Cast characters to unsigned char before converting to larger integer sizes."
    // Return type is specified by the signed version of Tb.
    // Unary - operator performs integral promotion where necessary, but it shouldn't be a problem as it will increase the size.
    //coverity[autosar_cpp14_m5_3_2_violation] FP: nvbugs/4536859
    return safeAdd(a, -b);
}
/**
 * Safe Substract.
 * Throws InvalidArgumentException if 'b' cannnot be negated into its corresponding signed type.
 * Throws BadSafetyResultAccess if 'a-b' would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSub(const Ta a, const int8_t b) -> decltype(detail::Add<Ta, decltype(b)>::impl(a, b))
{
    // Handle cert_str34_c_violation "Casting character to larger integer size directly. -- Cast characters to unsigned char before converting to larger integer sizes."
    // Return type is specified by the signed version of Tb.
    // Unary - operator performs integral promotion where necessary, but it shouldn't be a problem as it will increase the size.
    return safeAdd(a, -b);
}
/**
 * Safe Substract.
 * Throws InvalidArgumentException if 'b' cannnot be negated into its corresponding signed type.
 * Throws BadSafetyResultAccess if 'a-b' would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSub(const Ta a, const Tb b) -> decltype(detail::Add<Ta, typename std::make_signed<Tb>::type>::impl(a, b))
{
    // Return type is specified by the signed version of Tb.
    // Unary - operator performs integral promotion where necessary, meaning
    // further steps in the chain may need to recast integers to smaller types.
    // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
    // coverity[cert_int32_c_violation] FP: nvbugs/4536859
    return safeAdd(a, -checkedIntegerCast<typename std::make_signed<Tb>::type>(b));
}
/**
 * Safe decrement.
 * Throws InvalidArgumentException if 'decr' cannnot be negated into its corresponding signed type.
 * Throws BadSafetyResultAccess if decrement would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void safeDecrement(Ta& var, const Tb decr)
{
    // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4536859
    // coverity[cert_int32_c_violation] FP: nvbugs/4536859
    var = safeAdd(var, -checkedIntegerCast<typename std::make_signed<Tb>::type>(decr)).value();
}
/**
 * Multiply @a a and @a b and check for overflow. This function is only enabled if @c TInteger is a signed integer.
 *
 * @return A @ref SafetyResult a @c TInteger value type. The result will only have a value if the multiplication did not
 *  overflow. If it did overflow, the result will have an error code of @c SafetyIssueType::OverflowError.
 */
template <typename TInteger, std::enable_if_t<std::is_signed<TInteger>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeMul(TInteger const& a, meta::TypeIdentityT<TInteger> const& b) -> SafetyResult<TInteger>
{
    constexpr TInteger ZERO{0};
    constexpr TInteger ONE{1};
    constexpr TInteger MAX{core::numeric_limits<TInteger>::max()};
    constexpr TInteger MIN{core::numeric_limits<TInteger>::min()};
    if ((a == ZERO) || (b == ZERO))
    {
        return SafetyResult<TInteger>(ZERO);
    }
    else if (a == ONE)
    {
        return SafetyResult<TInteger>(b);
    }
    else if (b == ONE)
    {
        return SafetyResult<TInteger>(a);
    }
    else if ((a == MIN) || (b == MIN))
    {
        return SafetyResult<TInteger>(SafetyIssueType::OverflowError);
    }
    else
    {
        TInteger const aAbs{static_cast<TInteger>(a < ZERO ? -a : a)};
        TInteger const bAbs{static_cast<TInteger>(b < ZERO ? -b : b)};
        if (MAX / aAbs < bAbs)
        {
            return SafetyResult<TInteger>(SafetyIssueType::OverflowError);
        }
        else
        {
            return SafetyResult<TInteger>(a * b);
        }
    }
}
/**
 * Multiply @a a and @a b and check for overflow. This function is only enabled if @c TInteger is an unsigned integer.
 *
 * @return A @ref SafetyResult a @c TInteger value type. The result will only have a value if the multiplication did not
 *  overflow. If it did overflow, the result will have an error code of @c SafetyIssueType::OverflowError.
 */
template <typename TInteger, std::enable_if_t<std::is_unsigned<TInteger>::value, bool> = false>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeMul(TInteger const& a, meta::TypeIdentityT<TInteger> const& b) -> SafetyResult<TInteger>
{
    constexpr TInteger ZERO{0};
    constexpr TInteger MAX{core::numeric_limits<TInteger>::max()};
    if ((a == ZERO) || (b == ZERO))
    {
        return SafetyResult<TInteger>(ZERO);
    }
    else if (MAX / a < b)
    {
        return SafetyResult<TInteger>(SafetyIssueType::OverflowError);
    }
    else
    {
        return SafetyResult<TInteger>(a * b);
    }
}
/**
 * Divide @a a by @a b and check for divide-by-zero. This function is only enabled if @c Tb is a floating point.
 *
 * @return A @ref SafetyResult a @c T value type. The result will only have a value if the @a b is not zero. If @a b is zero, the result will have an error code of @c SafetyIssueType::PoleError.
 */
// template <typename Ta, typename Tb, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
template <typename Ta, typename Tb, typename Tc = typename detail::DivideResult<Ta, Tb>::type, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeDiv(Ta const& a, Tb const& b) -> SafetyResult<Tc>
{
    if (!core::isValidDivisor(b))
    {
        return SafetyResult<Tc>(SafetyIssueType::PoleError);
    }
    else
    {
        return SafetyResult<Tc>(a / b);
    }
}
/**
 * Divide @a a by @a b and check for divide-by-zero. This function is only enabled if @c Tb is a integer.
 *
 * @return A @ref SafetyResult a @c T value type. The result will only have a value if the @a b is not zero. If @a b is zero, the result will have an error code of @c SafetyIssueType::PoleError.
 */
// template <typename Ta, typename Tb, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
template <typename Ta, typename Tb, typename Tc = typename detail::DivideResult<Ta, Tb>::type, std::enable_if_t<std::is_integral<Tb>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeDiv(Ta const& a, Tb const& b) -> SafetyResult<Tc>
{
    if (!core::isValidDivisor(b))
    {
        return SafetyResult<Tc>(SafetyIssueType::PoleError);
    }
    else
    {
        // TODO(dwplc): FP - b required to be integral type when it is 0
        // coverity[autosar_cpp14_m5_0_5_violation] FP: nvbugs/4536859
        return SafetyResult<Tc>(a / b);
    }
}
/**
 * Divide @a a by @a b and check for divide-by-zero. This function is only enabled if @c T is a signed integer.
 *
 * @return A @ref SafetyResult a @c T value type. The result will only have a value if @a b is not zero and not hit overflow case. If @a b is zero, the result will have an error code of @c SafetyIssueType::PoleError. If hitting overflow case, the result will have an error code of @c SafetyIssueType::OverflowError.
 */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeDiv(T const& a, T const& b) -> SafetyResult<T>
{
    if (!core::isValidDivisor(b))
    {
        return SafetyResult<T>(SafetyIssueType::PoleError);
    }
    else if (a == core::numeric_limits<T>::min() && b == -1)
    {
        return SafetyResult<T>(SafetyIssueType::OverflowError);
    }
    else
    {
        return SafetyResult<T>(a / b);
    }
}

/**
 * Add two values in a safe AUTOSAR/MISRA compliant way to avoid integer over- or under-flow.
 *
 * @param[out] sum Result of a + b, if necessary clamped (check return argument)
 * @param[in] a
 * @param[in] b
 * @return false if result had to be clamped to numerical limits, true otherwise
 *
 **/
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool clampedAddition(T& sum, const T a, const T b) noexcept
{
    // allowable range for b
    T min;
    T max;

    // coverity[cert_str34_c_violation] RFD Pending: TID-2614
    if (a >= 0)
    {
        min = core::numeric_limits<T>::min();
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        max = core::numeric_limits<T>::max() - a;
    }
    else
    {
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        min = core::numeric_limits<T>::min() - a;
        max = core::numeric_limits<T>::max();
    }

    // nvbugs/4417023 applicable to:
    //     - [cert_int30_c_violation] unsigned integers
    //     - [cert_int32_c_violation] signed integers
    //     - [autosar_cpp14_a4_7_1_violation]
    // the issue in all cases is the same, the checker cannot detect clamping that took place with min/max in detail::clamp.
    // coverity[cert_int30_c_violation] FP: nvbugs/4417023
    // coverity[cert_int32_c_violation] FP: nvbugs/4417023
    // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4417023
    // coverity[cert_int31_c_violation] FP: nvbugs/4479779
    // coverity[cert_str34_c_violation] RFD Pending: TID-2614
    sum = a + detail::clamp(b, min, max);
    return detail::withinRange(b, min, max);
}

/**
 * Subtract two values in a safe AUTOSAR/MISRA compliant way to avoid integer over- or under-flow.
 *
 * @param[out] diff Result of a - b, if necessary clamped (check return argument)
 * @param[in] a
 * @param[in] b
 * @return false if result had to be clamped to numerical limits, true otherwise
 *
 **/
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool clampedSubtraction(T& diff, const T a, const T b) noexcept
{
    // allowable range for b
    T min;
    T max;

    // coverity[cert_str34_c_violation] RFD Pending: TID-2614
    if (a >= 0)
    {
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        min = std::is_signed<T>::value ? (a - core::numeric_limits<T>::max()) : 0;
        max = std::is_signed<T>::value ? core::numeric_limits<T>::max() : a;
    }
    else
    {
        min = core::numeric_limits<T>::min();
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        max = a - core::numeric_limits<T>::min();
    }

    // nvbugs/4417023 applicable to:
    //     - [cert_int30_c_violation] unsigned integers
    //     - [cert_int32_c_violation] signed integers
    //     - [autosar_cpp14_a4_7_1_violation]
    // the issue in all cases is the same, the checker cannot detect clamping that took place with min/max in detail::clamp.
    // coverity[cert_int30_c_violation] FP: nvbugs/4417023
    // coverity[cert_int32_c_violation] FP: nvbugs/4417023
    // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4417023
    // coverity[cert_int31_c_violation] FP: nvbugs/4479779
    // coverity[cert_str34_c_violation] RFD Pending: TID-2614
    diff = a - detail::clamp(b, min, max);
    return detail::withinRange(b, min, max);
}

/**
 * Multiply two values in a safe AUTOSAR/MISRA compliant way to avoid integer over- or under-flow.
 *
 * @param[out] mult Result of a * b, if necessary clamped (check return argument)
 * @param[in] a
 * @param[in] b
 * @return false if result had to be clamped to numerical limits, true otherwise
 *
 **/
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool clampedMultiplication(T& mult, const T a, const T b) noexcept
{
    // allowable range for b
    T min{core::numeric_limits<T>::min()}; // a == 0
    T max{core::numeric_limits<T>::max()}; // a == 0

    // coverity[cert_str34_c_violation] RFD Pending: TID-2614
    if (a > 0)
    {
        // coverity[cert_int31_c_violation] FP: nvbugs/4479779
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        min = core::numeric_limits<T>::min() / a;
        // coverity[cert_int31_c_violation] FP: nvbugs/4479779
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        max = core::numeric_limits<T>::max() / a;
    }

    // a workaround for the missing if constexpr (std::is_signed<T>) which is available from c++17 only
    core::staticIf(
        std::is_signed<T>{},
        [&min, &max](T const& _a) {
            // coverity[cert_str34_c_violation] RFD Pending: TID-2614
            if (_a < 0)
            {
                // coverity[cert_int31_c_violation] FP: nvbugs/4479779
                // coverity[cert_str34_c_violation] RFD Pending: TID-2614
                min = core::numeric_limits<T>::max() / _a;
                // coverity[cert_int31_c_violation] FP: nvbugs/4479779
                // coverity[cert_str34_c_violation] RFD Pending: TID-2614
                max = _a < static_cast<T>(-1) ? (core::numeric_limits<T>::min() / _a) : core::numeric_limits<T>::max();
            }
        },
        a);

    if (b > max)
    {
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        mult = a > 0 ? core::numeric_limits<T>::max() : core::numeric_limits<T>::min();
        return false;
    }
    else if (b < min)
    {
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        mult = a > 0 ? core::numeric_limits<T>::min() : core::numeric_limits<T>::max();
        return false;
    }
    else
    {
        // coverity[cert_str34_c_violation] RFD Pending: TID-2614
        mult = a * b;
        return true;
    }
}

/** Add two values in a safe AUTOSAR/MISRA compliant way with wrapping behavior for unsigned integer over-flows.
 *
 *  @param[out] sum Result of a + b, if necessary wrapped
 *  @param[in] a
 *  @param[in] b
 *  @return false if result had to be wrapped around numerical limits, true otherwise
 *
 **/
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool> = true>
CUDA_BOTH_INLINE bool wrappedAddition(T& sum, const T a, const T b) noexcept
{
    const T headRoom{static_cast<T>(core::numeric_limits<T>::max() - a)};

    if (b <= headRoom)
    {
        // nvbugs/4479331 was filed for AUTOSAR violation, but also applies for CERT-C equivalent
        // coverity[cert_int30_c_violation] FP: nvbugs/4479331
        // coverity[autosar_cpp14_a4_7_1_violation] FP: nvbugs/4479331
        // coverity[cert_int31_c_violation] FP: nvbugs/4479779
        sum = a + b;
        return false;
    }
    else
    {
        // coverity[cert_int31_c_violation] FP: nvbugs/4479779
        sum = b - headRoom - 1;
        return true;
    }
}

// --------------------------------------------------------------------------------------------------------
// Safe evaluation wrappers for 'std::' match functions. Function return empty result in case of domain and
// range / pole errors to fullfill AUTOSAR rules

namespace detail
{
// TODO(vsamokhvalov): move out of the detail namespace
// Wrapper to capture floating point errors and exceptions.
// Used to verify some math functions for safe execution according to AUTOSAR rules.
template <typename T>
auto CUDA_BOTH_INLINE checkMathErrors(const T& val) -> SafetyResult<T>
{
#ifdef __CUDA_ARCH__
    if (core::isnan(val))
    {
        return SafetyResult<T>(SafetyIssueType::DomainError);
    }
#else
    if (math_errhandling & MATH_ERREXCEPT)
    {
        if (fetestexcept(FE_INVALID) != 0)
        {
            return SafetyResult<T>(SafetyIssueType::DomainError);
        }
        else if (fetestexcept(FE_DIVBYZERO) != 0)
        {
            return SafetyResult<T>(SafetyIssueType::PoleError);
        }
        else if (fetestexcept(FE_OVERFLOW) != 0)
        {
            return SafetyResult<T>(SafetyIssueType::OverflowError);
        }
        // The last two cases are permissible and are not treated as errors.
        // This can be promoted to full errors later if needed.
        else if (fetestexcept(FE_UNDERFLOW) != 0)
        {
            return SafetyResult<T>(val);
        }
        else if (fetestexcept(FE_INEXACT) != 0)
        {
            return SafetyResult<T>(val);
        }
    }
    else if (math_errhandling & MATH_ERRNO)
    {
        if (errno == EDOM)
        {
            return SafetyResult<T>(SafetyIssueType::DomainError);
        }
        else if (errno == ERANGE)
        {
            return SafetyResult<T>(SafetyIssueType::RangeError);
        }
    }
#endif

    return SafetyResult<T>(val);
}
} // namespace detail

// --------------------------------------------------------------------------------------------------------
// All functions defined here that start with 'safe' return a SafetyResult<T>, except for those ending with 'Clamped'.
// Clamped math operations are guaranteed to produce a valid output, except is the input is invalid (e.g. NaN).

// Performs a safe evaluation of 'std::sqrt' by clamping the argument to the valid range [0.0, Inf)
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSqrtClamped(T const& x) -> const T
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto clamped = detail::clamp(x, static_cast<T>(0), core::numeric_limits<T>::max());
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::sqrt(clamped)).value();
}

// Performs a safe evaluation of 'std::acos' by clamping the argument to the valid range [-1.0, 1.0]
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAcosClamped(T const& x) -> const T
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto clamped = detail::clamp(x, static_cast<T>(-1), static_cast<T>(1));
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::acos(clamped)).value();
}

// Performs a safe evaluation of 'std::asin' by clamping the argument to the valid range [-1.0, 1.0]
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAsinClamped(T const& x) -> const T
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto const clamped = detail::clamp(x, static_cast<T>(-1), static_cast<T>(1));
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::asin(clamped)).value();
}

// Performs a safe evaluation of 'std::acosh' by clamping the argument to the valid range [1.0, Inf)
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAcoshClamped(T const& x) -> const T
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto clamped = detail::clamp(x, static_cast<T>(1), core::numeric_limits<T>::max());
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::acosh(clamped)).value();
}

// --------------------------------------------------------------------------------------------------------
// Math operations with error checking wrapper safeMathEvaluation

// Performs a safe evaluation of 'std::atan' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAtan(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::atan(x));
}
// Performs a safe evaluation of 'std::atan2' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAtan2(T const& y, T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::atan2(y, x));
}
// Performs a safe evaluation of 'std::exp2' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeExp2(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::exp2(x));
}
// Performs a safe evaluation of 'std::fmod' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeFmod(T const& x, T const& y) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::fmod(x, y));
}
// Performs a safe evaluation of 'std::remainder' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeRemainder(T const& x, T const& y) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::remainder(x, y));
}
// Performs a safe evaluation of 'std::sqrt' by checking the evaluation for validity
// Return value type is the same as in std::sqrt(). The value type may be different than input type
//   because std::sqrt() casts non-ovrerloaded types to float64_t

template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSqrt(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
#ifdef __CUDA_ARCH__
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(sqrt(x));
#else
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::sqrt(x));
#endif
}

// Performs a safe evaluation of 'std::log10' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeLog2(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::log2(x));
}
// Performs a safe evaluation of 'std::log10' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeLog10(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::log10(x));
}
// Performs a safe evaluation of 'std::exp' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeExp(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::exp(x));
}
// Performs a safe evaluation of 'std::pow' by checking the evaluation for validity
template <typename TBase, typename TExp>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safePow(TBase const& base, TExp const& exp) -> SafetyResult<TBase>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<TBase>(std::pow(base, exp));
}
// Performs a safe evaluation of 'std::acos' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAcos(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::acos(x));
}
// Performs a safe evaluation of 'std::asin' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAsin(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::asin(x));
}
// Performs a safe evaluation of 'std::cosh' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeCosh(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::cosh(x));
}
// Performs a safe evaluation of 'std::sinh' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeSinh(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::sinh(x));
}
// Performs a safe evaluation of 'std::acosh' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAcosh(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::acosh(x));
}
// Performs a safe evaluation of 'std::asinh' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeAsinh(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::asinh(x));
}
// Performs a safe evaluation of 'std::erf' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeErf(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::erf(x));
}
// Performs a safe evaluation of 'std::erfc' by checking the evaluation for validity
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeErfc(T const& x) -> SafetyResult<T>
{
    dw::core::resetMathErrors();
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481101
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481115
    // coverity[autosar_cpp14_a0_4_4_violation] FP: nvbugs/4481154
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481101
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481115
    // coverity[cert_flp32_c_violation] FP: nvbugs/4481154
    return detail::checkMathErrors<T>(std::erfc(x));
}

// --------------------------------------------------------------------------------------------------------
// Safe casting operations involving floating point numbers

/**
 * @brief Narrow cast float64_t to float32_t.
 *
 * Casts the input value to a 32 bit float, making sure to return an error if the value
 * is not within the representable range of the destination type. By using the optional parameter
 * @c zeroThreshold, one can configure if conversion of values in range [DBL_MIN, FLT_MIN] should be
 * treated as errors (as advised by the CERT Coding Standard), or such values should simply be
 * interpreted as zero (default behavior).
 * 
 * @param x The value to cast.
 * @param zeroThreshold Threshold under which all values are treated as zero.
 * @return An object representing either the cast value, or a potential casting error.
 */
inline SafetyResult<float32_t> safeFloatCast(float64_t const& x, float64_t const zeroThreshold = std::numeric_limits<float64_t>::epsilon())
{
    if (std::isnan(x))
    {
        return SafetyResult<float32_t>(SafetyIssueType::UnspecificError);
    }

    bool isZero{std::abs(x) <= zeroThreshold};
    if (isZero)
    {
        return SafetyResult<float32_t>{0.F};
    }

    const bool isSmallerThanSmallestFl32{std::fabs(x) < core::numeric_limits<float32_t>::min()};
    const bool isGreaterThanGreatestFl32{std::fabs(x) > core::numeric_limits<float32_t>::max()};

    // On QNX, __DBL_EPSILON__ is greater than FLT_MIN, so with default value of zeroThreshold, values
    // in range [DBL_MIN, FLT_MIN] will be treated as zero and cast to float32_t instead of an error
    // being returned.
    if (isSmallerThanSmallestFl32)
    {
        return SafetyResult<float32_t>(SafetyIssueType::UnderflowError);
    }
    if (isGreaterThanGreatestFl32)
    {
        return SafetyResult<float32_t>(SafetyIssueType::OverflowError);
    }

    return SafetyResult<float32_t>(static_cast<float32_t>(x));
}

/**
 * @brief Cast a floating point number safely into a signed integer type
 *
 * Check if floating point number can be represented in the range of the destination
 * integer type, and performs the cast in that case. Returns error information otherwise.
 *
 * @tparam TInt Destination integer type.
 * @tparam TFloat Source floating point type.
 * @param x The value to cast.
 * @return An object representing either the cast value, or a potential casting error.
 */
template <typename TInt, typename TFloat, std::enable_if_t<std::is_integral<TInt>::value, bool>* = nullptr>
CUDA_BOTH_INLINE SafetyResult<TInt> safeCastFloatToInt(TFloat const& x)
{
    if (core::isnan(x))
    {
        return SafetyResult<TInt>(SafetyIssueType::UnspecificError);
    }

    if (static_cast<float64_t>(x) < detail::minFloatRepresentable<TInt>() || static_cast<float64_t>(x) > detail::maxFloatRepresentable<TInt>())
    {
        return SafetyResult<TInt>(SafetyIssueType::PrecisionLossError);
    }

    return SafetyResult<TInt>(static_cast<TInt>(x));
}

/**
 * @brief Safely cast an unsigned integer to a floating point number.
 *
 * Checks if the argument is within the representable range and performs the cast in that case.
 * User of this function shall check the return value for validity before trying to use the
 * underlying value.
 *
 * @param x Unsigned integer to be cast
 * @return An object representing either the cast value, or a potential precision loss.
 * @note Using this function in code that is executed at high frequency (for example, in CUDA
 * kernels) can lead to performance degradation. Please consider refactoring your code or try
 * to assess an upper bound for the values you are converting. Maybe you don't even need to
 * use this checked cast.
 */
template <typename TInt, typename TFloat, std::enable_if_t<std::is_integral<TInt>::value && std::is_unsigned<TInt>::value>* = nullptr>
CUDA_BOTH_INLINE SafetyResult<TFloat> safeCastUnsignedIntToFloat(TInt const& x)
{
    static_assert(std::is_same<TFloat, float32_t>::value || std::is_same<TFloat, float64_t>::value, "safeCastUnsignedIntToFloat only supports float32_t and float64_t");
    // 32 bit floating point number has a 24 bits significand precision
    // 64 bit floating point number has a 53 bits significand precision
    constexpr uint64_t MAX_ACCURATE_INTEGER_AS_FLOAT{std::is_same<TFloat, float32_t>::value ? 16777216 : 9007199254740992};
    // coverity[result_independent_of_operands] RFD Pending: TID-2638
    if (static_cast<uint64_t>(x) < MAX_ACCURATE_INTEGER_AS_FLOAT)
    {
        return SafetyResult<TFloat>(static_cast<TFloat>(x));
    }
    return SafetyResult<TFloat>(SafetyIssueType::PrecisionLossError);
}

/**
 * @brief Safely cast a signed integer to a floating point number.
 *
 * Checks if the argument is within the representable range and performs the cast in that case.
 * User of this function shall check the return value for validity before trying to use the
 * underlying value.
 *
 * @param x Signed integer to be cast
 * @return An object representing either the cast value, or a potential precision loss.
 * @note Using this function in code that is executed at high frequency (for example, in CUDA
 * kernels) can lead to performance degradation. Please consider refactoring your code or try
 * to assess an upper bound for the values you are converting. Maybe you don't even need to
 * use this checked cast.
 */
template <typename TInt, typename TFloat, std::enable_if_t<std::is_integral<TInt>::value && std::is_signed<TInt>::value>* = nullptr>
CUDA_BOTH_INLINE SafetyResult<TFloat> safeCastSignedIntToFloat(TInt const& x)
{
    static_assert(std::is_same<TFloat, float32_t>::value || std::is_same<TFloat, float64_t>::value, "safeCastSignedIntToFloat only supports float32_t and float64_t");
    // 32 bit floating point number has a 24 bits significand precision
    // 64 bit floating point number has a 53 bits significand precision
    constexpr int64_t MAX_ACCURATE_INTEGER_AS_FLOAT{std::is_same<TFloat, float32_t>::value ? 16777216 : 9007199254740992};
    if (static_cast<int64_t>(x) > -MAX_ACCURATE_INTEGER_AS_FLOAT && static_cast<int64_t>(x) < MAX_ACCURATE_INTEGER_AS_FLOAT)
    {
        return SafetyResult<TFloat>(static_cast<TFloat>(x));
    }
    return SafetyResult<TFloat>(SafetyIssueType::PrecisionLossError);
}

// --------------------------------------------------------------------------------------------------------
// The system library's htons, ntohs, htonl, ntohl, htonll, ntohll violate AUSTOSAR rules, define compliant versions.

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__)
static_assert(false, "__BYTE_ORDER__ or __ORDER_LITTLE_ENDIAN__ are not defined, cannot continue.");
#endif

inline uint16_t safe_htons(uint16_t const host)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap16(host);
#else
    return host;
#endif
}

inline uint16_t safe_ntohs(uint16_t const network)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap16(network);
#else
    return network;
#endif
}

inline uint32_t safe_htonl(uint32_t const host)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap32(host);
#else
    return host;
#endif
}

inline uint32_t safe_ntohl(uint32_t const network)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap32(network);
#else
    return network;
#endif
}

inline uint64_t safe_htonll(uint64_t const host)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(host);
#else
    return host;
#endif
}

inline int64_t safe_htonll(int64_t const host)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto forByteSwap = safeReinterpretCast<uint64_t const*>(&host);
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto swapped = __builtin_bswap64(*forByteSwap);
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto forReturn = safeReinterpretCast<int64_t*>(&swapped);
    return *forReturn; // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#else
    return host;
#endif
}

inline uint64_t safe_ntohll(uint64_t const network)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(network);
#else
    return network;
#endif
}

inline int64_t safe_ntohll(int64_t const network)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto forByteSwap = safeReinterpretCast<uint64_t const*>(&network);
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto swapped = __builtin_bswap64(*forByteSwap);
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto forReturn = safeReinterpretCast<int64_t*>(&swapped);
    return *forReturn; // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#else
    return host;
#endif
}

/**
 * @brief Perform a safe conversion from boolean to uint8_t
 *
 * @param value input boolean
 * @return constexpr uint8_t output value
 */
inline constexpr uint8_t safeBoolToUint8(const bool value) noexcept
{
    uint8_t ret{0U};
    if (value)
    {
        ret = static_cast<uint8_t>(1U);
    }
    return ret;
}

} // namespace core
} // namespace dw

#endif // DWSHARED_CORE_SAFETY_SAFETY_HPP_