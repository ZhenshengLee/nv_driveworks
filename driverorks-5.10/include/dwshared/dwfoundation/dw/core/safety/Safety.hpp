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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dw/core/language/CheckedIntegerCast.hpp>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/language/Math.hpp>
#include <dw/core/language/StaticIf.hpp>
#include <dw/core/meta/Conjunction.hpp>
#include <dw/core/meta/TypeIdentity.hpp>
#include <dw/core/language/Optional.hpp>
#include "impl/SafetyException.hpp"
#include "MathErrors.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
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
    UnderflowError
};

//move getErrorMessageForIssue from cpp to hpp, otherwise compiling .cu file will get Unresolved extern function error
/// Get the exception error message associate to the given @a code.
CUDA_BOTH_INLINE char8_t const* getErrorMessageForIssue(SafetyIssueType const code)
{
    char8_t const* result = nullptr;
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
    // TODO(dwplc): FP - The SafetyIssueType is a enum type and this template is not accessing any of its member.
    // coverity[autosar_cpp14_m14_6_1_violation]
    using IssueType = SafetyIssueType;

    CUDA_BOTH_INLINE SafetyResult() // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
        : SafetyResult(IssueType::None)
    {
    }

    // TODO(dwplc): FP - This NullOpt is not used.
    // coverity[autosar_cpp14_m14_6_1_violation]
    CUDA_BOTH_INLINE constexpr SafetyResult(NullOpt) noexcept // clang-tidy NOLINT(google-explicit-constructor)
        : Base(), m_issueType(IssueType::None)
    {
    }

    CUDA_BOTH_INLINE explicit SafetyResult(const BaseOptional<T, SafetyResult>& other)
        : Base(other), m_issueType(other.m_issueType) {}

    CUDA_BOTH_INLINE explicit SafetyResult(Base&& other)
        : Base(other), m_issueType(other.m_issueType) {}

    CUDA_BOTH_INLINE SafetyResult(const T& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
        : Base(valueIn), m_issueType(IssueType::None)
    {
    }

    CUDA_BOTH_INLINE SafetyResult(T&& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
        : Base(std::move(valueIn)), m_issueType(IssueType::None)
    {
    }

    CUDA_BOTH_INLINE explicit SafetyResult(IssueType const issueType)
        : Base(), m_issueType(issueType) {}

    CUDA_BOTH_INLINE void handleInvalidAccess() const
    {
        if (m_issueType == IssueType::None)
        {
            // TODO(dwplc): FP - assertException will only use constructor of the template.
            // coverity[autosar_cpp14_m14_6_1_violation]
            core::assertException<BadOptionalAccess>(BadOptionalAccess::MESSAGE);
        }
        else
        {
            // TODO(dwplc): FP - assertException will only use constructor of the template.
            // coverity[autosar_cpp14_m14_6_1_violation]
            core::assertException<BadSafetyResultAccess>(getErrorMessageForIssue(m_issueType));
        }
    }

    CUDA_BOTH_INLINE auto getIssueType() const -> IssueType { return m_issueType; }
    CUDA_BOTH_INLINE bool isSafe() const { return Base::has_value() && m_issueType == IssueType::None; }

    // Decay a SafetyResult to a regular Optional, keeping the result value but dropping the IssueType
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
          bool uTb = std::is_unsigned<Tb>::value>
struct Add
{
    static auto impl(Ta a, Tb b) -> SafetyResult<Ta>;
};

// unsigned = unsigned + unsigned
template <typename Ta, typename Tb>
struct Add<Ta, Tb, true, true>
{
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        Ta const left = core::numeric_limits<Ta>::max() - a;
        if (b >= left)
        {
            return SafetyResult<Ta>(SafetyIssueType::OverflowError);
        }

        return SafetyResult<Ta>(a + b);
    }
};

// unsigned = unsigned + signed
template <typename Ta, typename Tb>
struct Add<Ta, Tb, true, false>
{
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Ta>
    {
        Ta const left = core::numeric_limits<Ta>::max() - a;
        Ta const absB = static_cast<Ta>(std::abs(b));

        if (b > 0)
        {
            if (absB > left)
            {
                return SafetyResult<Ta>(SafetyIssueType::OverflowError);
            }
            return SafetyResult<Ta>(a + absB);
        }
        else
        {
            if (absB > a)
            {
                return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
            }
            return SafetyResult<Ta>(a - absB);
        }
    }
};

// unsigned = signed + unsigned
template <typename Ta, typename Tb>
struct Add<Ta, Tb, false, true>
{
    CUDA_BOTH_INLINE static auto impl(const Ta a, const Tb b) -> SafetyResult<Tb> { return Add<Tb, Ta>::impl(b, a); }
};

// signed = signed + signed
template <typename Ta, typename Tb>
struct Add<Ta, Tb, false, false>
{
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
                // note: Tb might be wider than Ta, we need to check for underflow in Ta
                // TODO(dwplc): FP - The left expression of && is not always false. Tb may be larger size than Ta
                // coverity[result_independent_of_operands]
                if (b < core::numeric_limits<Ta>::lowest() && b - core::numeric_limits<Ta>::lowest() > a)
                {
                    return SafetyResult<Ta>(SafetyIssueType::UnderflowError);
                }

                return SafetyResult<Ta>(a + b);
            }
        }
        else // a <= 0
        {
            if (b > 0)
            {
                // note: Tb might be wider than Ta, we need to check for overflow in Ta
                // TODO(dwplc): FP - The left expression of && is not always false.
                // coverity[result_independent_of_operands]
                if (b > core::numeric_limits<Ta>::max() && b - core::numeric_limits<Ta>::max() > std::abs(a))
                {
                    return SafetyResult<Ta>(SafetyIssueType::OverflowError);
                }
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
CUDA_BOTH_INLINE constexpr bool withinRange(T v, T lo, T hi)
{
    return (v >= lo) && (v <= hi);
}

/// std::clamp() will arrive with C++17, until then:
/// clamps a value within a range
template <class T>
CUDA_BOTH_INLINE auto clamp(const T& v, const T& lo, const T& hi) -> const T
{
    return core::min(hi, core::max(v, lo));
}

/// Helper struct to determine the type of the expression a/b
template <class A, class B>
struct DivideResult
{
    // TODO(dwplc): FP - basic numerical type "int" and "unsigned int" aren't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    using type = decltype(std::declval<A>() / std::declval<B>());
};
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
CUDA_BOTH_INLINE auto safeAdd(const Ta a, const Tb b) -> decltype(detail::Add<Ta, Tb>::impl(a, b))
{
    return detail::Add<Ta, Tb>::impl(a, b);
}

/**
 * Safe increment. Throws if increment would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta, typename Tb>
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
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE auto safeSub(const Ta a, const Tb b) -> decltype(detail::Add<Ta, Tb>::impl(a, b))
{
    return safeAdd(a, -checkedIntegerCast<typename std::make_signed<Tb>::type>(b));
}

/**
 * Safe decrement.
 * Throws InvalidArgumentException if 'decr' cannnot be negated into its corresponding signed type.
 * Throws BadSafetyResultAccess if decrement would cause over- or under-flow.
 * @see safeAdd.
 */
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE void safeDecrement(Ta& var, const Tb decr)
{
    var = safeAdd(var, -checkedIntegerCast<typename std::make_signed<Tb>::type>(decr)).value();
}

/**
 * Multiply @a a and @a b and check for overflow. This function is only enabled if @c TInteger is a signed integer.
 *
 * @return A @ref SafetyResult a @c TInteger value type. The result will only have a value if the multiplication did not
 *  overflow. If it did overflow, the result will have an error code of @c SafetyIssueType::OverflowError.
 */
template <typename TInteger, std::enable_if_t<std::is_signed<TInteger>::value, bool> = true>
CUDA_BOTH_INLINE auto safeMul(TInteger const& a, meta::TypeIdentityT<TInteger> const& b) -> SafetyResult<TInteger>
{
    constexpr TInteger ZERO{0};
    constexpr TInteger ONE{1};
    constexpr TInteger MAX = core::numeric_limits<TInteger>::max();
    constexpr TInteger MIN = core::numeric_limits<TInteger>::min();

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
        TInteger const aAbs = a < ZERO ? -a : a;
        TInteger const bAbs = b < ZERO ? -b : b;

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
CUDA_BOTH_INLINE auto safeMul(TInteger const& a, meta::TypeIdentityT<TInteger> const& b) -> SafetyResult<TInteger>
{
    constexpr TInteger ZERO{0};
    constexpr TInteger MAX = core::numeric_limits<TInteger>::max();

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
//template <typename Ta, typename Tb, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
template <typename Ta, typename Tb, typename Tc = typename detail::DivideResult<Ta, Tb>::type, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
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
//template <typename Ta, typename Tb, std::enable_if_t<std::is_floating_point<Tb>::value, bool> = true>
template <typename Ta, typename Tb, typename Tc = typename detail::DivideResult<Ta, Tb>::type, std::enable_if_t<std::is_integral<Tb>::value, bool> = true>
CUDA_BOTH_INLINE auto safeDiv(Ta const& a, Tb const& b) -> SafetyResult<Tc>
{
    if (!core::isValidDivisor(b))
    {
        return SafetyResult<Tc>(SafetyIssueType::PoleError);
    }
    else
    {
        // TODO(dwplc): FP - b required to be integral type when it is 0
        // coverity[autosar_cpp14_m5_0_5_violation]
        return SafetyResult<Tc>(a / b);
    }
}

/**
 * Divide @a a by @a b and check for divide-by-zero. This function is only enabled if @c T is a signed integer.
 *
 * @return A @ref SafetyResult a @c T value type. The result will only have a value if @a b is not zero and not hit overflow case. If @a b is zero, the result will have an error code of @c SafetyIssueType::PoleError. If hitting overflow case, the result will have an error code of @c SafetyIssueType::OverflowError.
 */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
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
CUDA_BOTH_INLINE bool clampedAddition(T& sum, const T a, const T b) noexcept
{
    // allowable range for b
    T min;
    T max;

    // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
    // coverity[cert_str34_c_violation]
    if (a >= 0)
    {
        min = core::numeric_limits<T>::min();
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        max = core::numeric_limits<T>::max() - a;
    }
    else
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        min = core::numeric_limits<T>::min() - a;
        max = core::numeric_limits<T>::max();
    }

    // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
    // coverity[cert_str34_c_violation]
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
CUDA_BOTH_INLINE bool clampedSubtraction(T& diff, const T a, const T b) noexcept
{
    // allowable range for b
    T min;
    T max;

    // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
    // coverity[cert_str34_c_violation]
    if (a >= 0)
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        min = std::is_signed<T>::value ? (a - core::numeric_limits<T>::max()) : 0;
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        max = std::is_signed<T>::value ? core::numeric_limits<T>::max() : a;
    }
    else
    {
        min = core::numeric_limits<T>::min();
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        max = a - core::numeric_limits<T>::min();
    }

    // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
    // coverity[cert_str34_c_violation]
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
CUDA_BOTH_INLINE bool clampedMultiplication(T& mult, const T a, const T b) noexcept
{
    // allowable range for b
    T min = core::numeric_limits<T>::min(); // a == 0
    T max = core::numeric_limits<T>::max(); // a == 0

    // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
    // coverity[cert_str34_c_violation]
    if (a > 0)
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        min = core::numeric_limits<T>::min() / a;
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        max = core::numeric_limits<T>::max() / a;
    }

    // a workaround for the missing if constexpr (std::is_signed<T>) which is available from c++17 only
    core::staticIf(
        std::is_signed<T>{},
        [&min, &max](T const& _a) {
            // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
            // coverity[cert_str34_c_violation]
            if (_a < 0)
            {
                // TODO(dwplc): FP - The 'a' is not unsigned. So this is not dead code
                // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
                // coverity[cert_str34_c_violation]
                // coverity[autosar_cpp14_m0_1_9_violation]
                // coverity[dead_error_begin]
                min = core::numeric_limits<T>::max() / _a;
                // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
                // coverity[cert_str34_c_violation]
                max = _a < static_cast<T>(-1) ? (core::numeric_limits<T>::min() / _a) : core::numeric_limits<T>::max();
            }
        },
        a);

    if (b > max)
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        mult = a > 0 ? core::numeric_limits<T>::max() : core::numeric_limits<T>::min();
        return false;
    }
    else if (b < min)
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        mult = a > 0 ? core::numeric_limits<T>::min() : core::numeric_limits<T>::max();
        return false;
    }
    else
    {
        // TODO(dwplc): FP - Complains cast char to larger integer size. But all variables are type T. No cast here.
        // coverity[cert_str34_c_violation]
        mult = a * b;
        return true;
    }
}

// --------------------------------------------------------------------------------------------------------
// Safe evaluation wrappers for 'std::' match functions. Function return empty result in case of domain and
// range / pole errors to fullfill AUTOSAR rules

namespace detail
{

// Wrapper to capture floating point errors and exceptions.
// Used to verify some math functions for safe execution according to AUTOSAR rules.
template <class F>
auto CUDA_BOTH_INLINE safeMathEvaluation(F&& f) -> decltype(f())
{
    // reset error and exception states
    dw::core::resetMathErrors();

    // evaluate function
    auto const result = std::forward<F>(f)();

    using TRet = typename std::remove_const<decltype(result)>::type;

#ifdef __CUDA_ARCH__
    // CUDA math function returns NaN for domain error
    auto const v = result.value();
    if (core::isnan(v))
    {
        return TRet(SafetyIssueType::DomainError);
    }
    else
    {
        return result;
    }
#else

    // check evaluation for errors
    // TODO(dwplc): FP - complains that function may set errno and that I should check if return value indicates an error
    // Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
    // coverity[cert_err30_c_violation]
    // coverity[autosar_cpp14_m5_0_21_violation]
    if (((math_errhandling & MATH_ERRNO) != 0) &&
        (errno == EDOM))
    {
        // Handle domain (errno = EDOM) error, range errors are handled by exceptions.
        // We don't want to handle ERANGE here, which might correspond to a permissible FE_UNDERFLOW exception
        return TRet(SafetyIssueType::DomainError);
    }
    // Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
    // coverity[autosar_cpp14_m5_0_21_violation]
    else if (((math_errhandling & MATH_ERREXCEPT) != 0) &&
             (fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) !=
              0 /* perform quick check for any type of failure */))
    {
        // Handle domain (FE_INVALID) and specific range exceptions: pole FE_DIVBYZERO and overflow
        // FE_OVERFLOW exceptions
        if (fetestexcept(FE_INVALID) != 0)
        {
            return TRet(SafetyIssueType::DomainError);
        }
        if (fetestexcept(FE_DIVBYZERO) != 0)
        {
            return TRet(SafetyIssueType::PoleError);
        }
        if (fetestexcept(FE_OVERFLOW) != 0)
        {
            return TRet(SafetyIssueType::OverflowError);
        }

        return TRet(SafetyIssueType::UnspecificError); // shouldn't be reached

        // Note: we are explicitly not checking for FE_UNDERFLOW / FE_INEXACT exceptions - these can still
        //       be promoted to full errors later if needed
    }
    else
    {
        return result;
    }

#endif
}

} // namespace detail

// --------------------------------------------------------------------------------------------------------
// All functions defined here that start with 'safe' return a SafetyResult<T>, except for those ending with 'Clamped'.
// Clamped math operations are guaranteed to produce a valid output, except is the input is invalid (e.g. NaN).

// Performs a safe evaluation of 'std::sqrt' by clamping the argument to the valid range [0.0, Inf)
template <typename T>
CUDA_BOTH_INLINE auto safeSqrtClamped(T const& x) -> const T
{
    auto clamped = detail::clamp(x, static_cast<T>(0), core::numeric_limits<T>::max());
    // TODO(dwplc): FP - "Values are clamped to valid input domain"
    // coverity[autosar_cpp14_a0_4_4_violation]
    T result = std::sqrt(clamped);
    dw::core::resetMathErrors();
    return result;
}

// Performs a safe evaluation of 'std::acos' by clamping the argument to the valid range [-1.0, 1.0]
template <typename T>
CUDA_BOTH_INLINE auto safeAcosClamped(T const& x) -> const T
{
    auto clamped = detail::clamp(x, static_cast<T>(-1), static_cast<T>(1));
    // TODO(dwplc): FP - "Values are clamped to valid input domain"
    // coverity[autosar_cpp14_a0_4_4_violation]
    T result = std::acos(clamped);
    dw::core::resetMathErrors();
    return result;
}

// Performs a safe evaluation of 'std::asin' by clamping the argument to the valid range [-1.0, 1.0]
template <typename T>
CUDA_BOTH_INLINE auto safeAsinClamped(T const& x) -> const T
{
    auto const clamped = detail::clamp(x, static_cast<T>(-1), static_cast<T>(1));
    // TODO(dwplc): FP - "Values are clamped to valid input domain"
    // coverity[autosar_cpp14_a0_4_4_violation]
    T const result = std::asin(clamped);
    dw::core::resetMathErrors();
    return result;
}

// Performs a safe evaluation of 'std::acosh' by clamping the argument to the valid range [1.0, Inf)
template <typename T>
CUDA_BOTH_INLINE auto safeAcoshClamped(T const& x) -> const T
{
    auto clamped = detail::clamp(x, static_cast<T>(1), core::numeric_limits<T>::max());
    // TODO(dwplc): FP - "Values are clamped to valid input domain"
    // coverity[autosar_cpp14_a0_4_4_violation]
    T result = std::acosh(clamped);
    dw::core::resetMathErrors();
    return result;
}

// --------------------------------------------------------------------------------------------------------
// Math operations with error checking wrapper safeMathEvaluation

// Performs a safe evaluation of 'std::atan' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAtan(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::atan(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::atan2' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAtan2(T const& y, T const& x) -> SafetyResult<T>
{
    auto f = [y, x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto const v = std::atan2(y, x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::fmod' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeFmod(T const& x, T const& y) -> SafetyResult<T>
{
    auto f = [y, x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto const v = std::fmod(x, y);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::sqrt' by checking the evaluation for validity
// Return value type is the same as in std::sqrt(). The value type may be different than input type
//   because std::sqrt() casts non-ovrerloaded types to float64_t

template <typename T>
CUDA_BOTH_INLINE auto safeSqrt(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
#ifdef __CUDA_ARCH__
        auto const v = sqrt(x);
#else
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto const v = std::sqrt(x);
#endif
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::log10' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeLog10(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::log10(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::exp' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeExp(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::exp(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::pow' by checking the evaluation for validity
template <typename TBase, typename TExp>
CUDA_BOTH_INLINE auto safePow(TBase const& base, TExp const& exp) -> SafetyResult<TBase>
{
    auto f = [base, exp]() -> SafetyResult<TBase> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto const v = std::pow(base, exp);
        return SafetyResult<TBase>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::acos' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAcos(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::acos(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::asin' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAsin(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::asin(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::cosh' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeCosh(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::cosh(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::sinh' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeSinh(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::sinh(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::acosh' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAcosh(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::acosh(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
}

// Performs a safe evaluation of 'std::asinh' by checking the evaluation for validity
template <typename T>
CUDA_BOTH_INLINE auto safeAsinh(T const& x) -> SafetyResult<T>
{
    auto f = [x]() -> SafetyResult<T> {
        // TODO(dwplc): FP - "Math error checking is done by safeMathEvaluation wrapper"
        // coverity[autosar_cpp14_a0_4_4_violation]
        auto v = std::asinh(x);
        return SafetyResult<T>(v);
    };
    return detail::safeMathEvaluation(f);
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
    auto forByteSwap = safeReinterpretCast<uint64_t const*>(&host);
    auto swapped     = __builtin_bswap64(*forByteSwap);
    auto forReturn   = safeReinterpretCast<int64_t*>(&swapped);
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
    auto forByteSwap = safeReinterpretCast<uint64_t const*>(&network);
    auto swapped     = __builtin_bswap64(*forByteSwap);
    auto forReturn   = safeReinterpretCast<int64_t*>(&swapped);
    return *forReturn; // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#else
    return host;
#endif
}

/**
 * String conversion wrappers check and print relevant errors before returning
 */
inline float64_t safeStrtod(char8_t const* const str, char8_t** const endptr)
{
    errno               = 0;
    float64_t const ret = strtod(str, endptr);
    if (errno != 0)
    {
        int32_t const errno_copy = errno;
        errno                    = 0;
        // TODO(dwplc): FP - The return type of std::strerror is char* instead of char. This is a FP.
        // coverity[autosar_cpp14_a3_9_1_violation]
        throw InvalidConversionException("Invalid float64_t conversion errno: ", std::strerror(errno_copy));
    }
    return ret;
}

inline int64_t safeStrtol(char8_t const* const str, char8_t** const endptr, int32_t const base)
{
    errno             = 0;
    int64_t const ret = strtol(str, endptr, base);
    if (errno != 0)
    {
        int32_t const errno_copy = errno;
        errno                    = 0;
        throw InvalidConversionException("Invalid int64_t conversion errno: ", std::strerror(errno_copy));
    }
    return ret;
}

inline uint64_t safeStrtoul(char8_t const* const str, char8_t** const endptr, int32_t const base)
{
    errno              = 0;
    uint64_t const ret = strtoul(str, endptr, base);
    if (errno != 0)
    {
        int32_t const errno_copy = errno;
        errno                    = 0;
        throw InvalidConversionException("Invalid uint64_t conversion errno: ", std::strerror(errno_copy));
    }
    return ret;
}

inline float32_t safeStrtof(char8_t const* const str, char8_t** const endptr)
{
    errno               = 0;
    float32_t const ret = strtof(str, endptr);
    if (errno != 0)
    {
        int32_t const errno_copy = errno;
        errno                    = 0;
        throw InvalidConversionException("Invalid float32_t conversion errno: ", std::strerror(errno_copy));
    }
    return ret;
}
} // namespace core
} // namespace dw

#endif // DWSHARED_CORE_SAFETY_SAFETY_HPP_
