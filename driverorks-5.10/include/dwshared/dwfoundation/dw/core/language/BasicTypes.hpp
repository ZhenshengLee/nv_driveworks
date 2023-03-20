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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_TYPES_HPP_
#define DWSHARED_CORE_TYPES_HPP_

#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/base/ExceptionWithStackTrace.hpp>

#include <dw/core/base/TypeAliases.hpp>

#include <cstdint>
#include <cstring>
#include <array>
#include <limits>
#include <complex>
#include <mutex>

/////////////////////////
/// Namespace includes
namespace dw
{
namespace core
{
}
// Make dw::core namespace available in dw
using namespace core; // clang-tidy NOLINT(google-build-using-namespace)
}

//////////////////////////
/*
* Traits for dwFloat16_t
* dwFloat16_t is a CUDA specific type. Traits and numeric_limits<> are not defined for. Define here so that Eigen classes compile for it and we can ask about traits within templates.
*/
namespace std
{

template <>
struct is_floating_point<dwFloat16_t> : true_type
{
};

template <>
struct is_signed<dwFloat16_t> : true_type
{
};

template <>
struct numeric_limits<dwFloat16_t> // clang-tidy NOLINT(readability-identifier-naming)
{

    // TODO(dwplc): RFD - this is intended API, as in std::numeric_limits.
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_specialized = true; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_signed = true; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_integer = false; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_exact = false; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_bounded = true; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr bool is_modulo = false; // clang-tidy NOLINT(readability-identifier-naming)

    // The following are disabled because dwFloat16_t can only be used in CUDA for now
    /*
    static constexpr dwFloat16_t min() noexcept { return __FLT16_MIN__; }
    static constexpr dwFloat16_t max() noexcept { return __FLT16_MAX__; }
    static constexpr dwFloat16_t lowest() noexcept { return -__FLT16_MAX__; }
    static constexpr int32_t digits = __FLT16_MANT_DIG__;
    static constexpr int32_t digits10 = __FLT16_DIG__;
    static constexpr int32_t max_digits10 = __glibcxx_max_digits10 (__FLT_MANT_DIG__);
    static constexpr int32_t radix = __FLT_RADIX__;
    static constexpr dwFloat16_t epsilon() noexcept { return __FLT16_EPSILON__; }
    static constexpr dwFloat16_t round_error() noexcept { return 0.5F; }
    static constexpr int32_t min_exponent = __FLT_MIN_EXP__;
    static constexpr int32_t min_exponent10 = __FLT_MIN_10_EXP__;
    static constexpr int32_t max_exponent = __FLT_MAX_EXP__;
    static constexpr int32_t max_exponent10 = __FLT_MAX_10_EXP__;
    static constexpr bool has_infinity = __FLT_HAS_INFINITY__;
    static constexpr bool has_quiet_NaN = __FLT_HAS_QUIET_NAN__;
    static constexpr bool has_signaling_NaN = has_quiet_NaN;
    static constexpr float_denorm_style has_denorm  = bool(__FLT_HAS_DENORM__) ? denorm_present : denorm_absent;
    static constexpr bool has_denorm_loss = __glibcxx_float_has_denorm_loss;
    static constexpr dwFloat16_t infinity() noexcept { return __builtin_huge_valf(); }
    static constexpr dwFloat16_t quiet_NaN() noexcept { return __builtin_nanf(""); }
    static constexpr dwFloat16_t signaling_NaN() noexcept { return __builtin_nansf(""); }
    static constexpr dwFloat16_t denorm_min() noexcept { return __FLT_DENORM_MIN__; }
    static constexpr bool is_iec559  = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
    static constexpr bool traps = __glibcxx_float_traps;
    static constexpr bool tinyness_before = __glibcxx_float_tinyness_before;
    static constexpr float_round_style round_style = round_to_nearest;
    */
};
}

namespace dw
{
namespace core
{

/// Type trait to determine if T is std::complex
template <class T>
struct is_complex : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};
template <class T>
struct is_complex<std::complex<T>> : std::true_type
{
};

/**
 * Checks whether an address is aligned
*/
template <class T>
CUDA_BOTH_INLINE bool isAligned(T* const ptr, size_t const alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment == 0); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

//////////////////////////////////////////////////////
/// Notes on aliasing and type punning
///
/// Modern C++ strict aliasing rules make reinterpret_cast and unions
/// almost useless. Many of the traditional C uses produce undefined
/// behaviour in C++. See good discussions:
///     http://blog.qt.io/blog/2011/06/10/type-punning-and-strict-aliasing/
///     https://stackoverflow.com/questions/4163126/dereferencing-type-punned-pointer-will-break-strict-aliasing-rules-warning/4163223
///
/// The only safe reinterpret is to char*. Thus, one needs to copy bytes manually to,
/// e.g. treat a bit field as an uint32_t. This is a programming pitfall and hassle, but
/// not really a performance problem since the compiler will optimize out the byte copies.
///
/// This function implements this byte-copy to be able to reinterpret types.
template <class Tout, class Tin>
CUDA_BOTH_INLINE auto reinterpretCharCopy(const Tin& in) -> Tout
{
    static_assert(sizeof(Tin) == sizeof(Tout), "Sizes must agree");

    Tout out;                                           // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
    auto* outc = reinterpret_cast<char8_t*>(&out);      // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* inc  = reinterpret_cast<const char8_t*>(&in); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)

    for (uint32_t i = 0; i < sizeof(Tin); i++)
    {
        outc[i] = inc[i];
    }
    return out;
}

/**
* Safe reinterpreting of a pointer. Enforces that both types must have the same sizeof()
* Note: Tout and Tin are not explicitly pointers to keep the same interface as reinterpret_cast<Tout>()
* Example:
*      dwVector2ui *a;
*      dw::Vector2ui *aCpp = safeReinterpretCast<dw::Vector2ui *>(a);
*/
// TODO(dwplc):  This function is based on reinterpret_cast, which violates rule AUTOSAR A5-2-4.
//               A strategy is needed for how to resolve all the places it is called across the project.
template <class ToutPointer, class TinPointer>
CUDA_BOTH_INLINE auto safeReinterpretCast(TinPointer const ptr) -> ToutPointer
{
    static_assert(std::is_pointer<ToutPointer>::value, "Tout must be a pointer type");
    static_assert(std::is_pointer<TinPointer>::value, "Tin must be a pointer type");

    using Tout = typename std::remove_pointer<ToutPointer>::type;
    using Tin  = typename std::remove_pointer<TinPointer>::type;
    static_assert(sizeof(Tout) == sizeof(Tin), "Size of input and output types do not match");

    // Always do the runtime alignment check even if the types have the same alignment requirement
    // This because we are unsure if the used C compiler will respect the alignment properly.
    // It should, but this is a small price to pay for safety.
    // To do the check only when the alignment requirements don't match (i.e. if we trust the user compiler)
    // uncomment this if line:
    //
    // code-comment " if (alignof(Tout) > alignof(Tin))"
    {
        // Type alignments do not match
        if (!isAligned(ptr, alignof(Tout)))
        {
            core::assertException<BadAlignmentException>("Mismatching alignment in conversion");
        }
    }
    return reinterpret_cast<Tout*>(ptr); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

/**
 * Safe conversion of stronly-typed enumeration instances to references / pointers of their underlying type
 */
template <class EnumT>
auto safeEnumAsReference(EnumT& e) ->
    typename std::enable_if<std::is_enum<EnumT>::value, typename std::underlying_type<EnumT>::type&>::type
{
    // Reinterpret is safe as only enabled for compatible underlying types. static_cast can't be used
    // as EnumT and std::underlying_type<EnumT>::type are not related by inheritance
    // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<typename std::underlying_type<EnumT>::type&>(e);
}
template <class EnumT>
auto safeEnumAsPointer(EnumT& e) ->
    typename std::enable_if<std::is_enum<EnumT>::value, typename std::underlying_type<EnumT>::type*>::type
{
    // Reinterpret is safe as only enabled for compatible underlying types. static_cast can't be used
    // as EnumT and std::underlying_type<EnumT>::type are not related by inheritance
    // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<typename std::underlying_type<EnumT>::type*>(&e);
}

/**
 * Static type conversion of *fully* evaluated values / results of expressions.
 * This is to prevent unintentionally conversions of cvalue expressions.
 */
template <typename T, typename U>
CUDA_BOTH_INLINE constexpr auto as(U&& u) -> T
{
    return static_cast<T>(std::forward<U>(u));
}

/**
* Compile-time condition to test at compile time without triggering a warning.
*/
template <bool b>
struct static_condition // clang-tidy NOLINT(readability-identifier-naming)
{
    static bool test()
    {
        return true;
    }
};
template <>
struct static_condition<false>
{
    static bool test()
    {
        return false;
    }
};

///Compile time condition to determine if a class is a specialization
///of a different class.  ie std::tuple<int,float> is specialization of std::tuple
template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type
{
};

/// T is type to modify, constness of TRef will be copied
/// Traits helper struct to copy the constness of one type onto the other
/// The resulting type will be "const T" if TRef is const, else "T"
/// For example:
///    using AC = copy_const<A, const uint32_t>::type   // AC == const A
///    using AN = copy_const<A, uint32_t>::type         // AN == A
template <class T, class TRef>
struct copy_const // clang-tidy NOLINT(readability-identifier-naming)
{
public:
    using type = typename std::conditional<
        std::is_const<TRef>::value,
        typename std::add_const<T>::type,
        typename std::remove_const<T>::type>::type;
};

///////////////////////////////////////////
/// Type info
/// This struct provides information that is not covered by std::type_info
///
/// Scalar: for simple types T, for Matrix<T,...> types T, for Jet<T,...> types T
/// Example usage:
///    void foo(T a)
///    {
///       using Scalar = typename TypeInfo<T>::Scalar;
///       Scalar threshold = Scalar{0.1f};
///       // If T is an autodiff::Jet, threshold will be scalar so no derivatives are used for it
///       ...
///    }
template <class T>
struct TypeInfo
{
    using Scalar = T;
};

} // namespace core
} // namespace dw

#endif // DW_CORE_TYPES_HPP_
