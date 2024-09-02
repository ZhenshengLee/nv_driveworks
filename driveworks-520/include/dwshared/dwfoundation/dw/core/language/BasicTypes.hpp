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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx17.hpp>

#include <cstdint>
#include <cstring>
#include <array>
#include <limits>
#include <complex>
#include <mutex>
#include <type_traits>

/////////////////////////
/// Namespace includes
namespace dw
{
namespace core
{
}
// Make dw::core namespace available in dw
using namespace core; // clang-tidy NOLINT(google-build-using-namespace)
} // namespace dw

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
struct is_fundamental<dwFloat16_t> : true_type
{
};

template <>
struct numeric_limits<dwFloat16_t> // clang-tidy NOLINT(readability-identifier-naming)
{
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    static constexpr bool is_specialized{true}; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    // coverity[cert_dcl51_cpp_violation] RFD Pending: TID-2461
    static constexpr bool is_signed{true}; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    static constexpr bool is_integer{false}; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    static constexpr bool is_exact{false}; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    static constexpr bool is_bounded{true}; // clang-tidy NOLINT(readability-identifier-naming)
    // coverity[autosar_cpp14_a17_6_1_violation] RFD Pending: TID-2304
    // coverity[cert_dcl58_cpp_violation] RFD Pending: TID-2304
    static constexpr bool is_modulo{false}; // clang-tidy NOLINT(readability-identifier-naming)

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
} // namespace std

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

/** \addtogroup basic_types_functions
 *  @{
 */

/**
 * @brief Checks whether an address is aligned in @b alignment bytes.
 *
 * 1. The function checks if @b alignment is 0, return false if it is 0.
 * 2. The function casts @b ptr to uintptr_t.
 * 3. The function modulates @b ptr converted value with @b alignment to check if the remainder is 0.
 * 
 * @tparam T the input pointer type.
 * @param[in] ptr The pointer to be checked.
 * @param[in] alignment Byte alignment number.
 * 
 * @retval true if the address is aligned in @b alignment bytes.
 * @retval false if the address is not aligned in @b alignment bytes.
 */
template <class T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isAligned(T* const ptr, size_t const alignment)
{
    constexpr size_t FORBIDDEN_ALIGNMENT{0};
    if (alignment == FORBIDDEN_ALIGNMENT)
    {
        return false;
    }

    // coverity[autosar_cpp14_m5_2_9_violation] RFD Pending: TID-2318
    return (reinterpret_cast<uintptr_t>(ptr) % alignment == 0); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

//////////////////////////////////////////////////////
// Notes on aliasing and type punning
//
// Modern C++ strict aliasing rules make reinterpret_cast and unions
// almost useless. Many of the traditional C uses produce undefined
// behaviour in C++. See good discussions:
//     http://blog.qt.io/blog/2011/06/10/type-punning-and-strict-aliasing/
//     https://stackoverflow.com/questions/4163126/dereferencing-type-punned-pointer-will-break-strict-aliasing-rules-warning/4163223
//
// The only safe reinterpret is to char*. Thus, one needs to copy bytes manually to,
// e.g. treat a bit field as an uint32_t. This is a programming pitfall and hassle, but
// not really a performance problem since the compiler will optimize out the byte copies.
//
// This function implements this byte-copy to be able to reinterpret types.
/**
 * @brief Copy @b in whose type is Tin to the object whose type is Tout.
 *
 * 1. The function uses compile time assertions to check that Tin and Tout are equal in size. If the sizes are not equal, a compile-time error is triggered, indicating that the sizes must be the same. <br>
 * 2. The function declares a local variable in type Tout, to store the copied data. <br>
 * 3. The function uses reinterpret_cast to convert out and @b in to pointers of type char8_t*, respectively, for byte-by-byte copying. <br>
 * 4. The function copys byte-by-byte by looping through sizeof(Tin) bytes. <br>
 * 
 * @return The copied data.
 * 
*/
template <class Tout, class Tin>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto reinterpretCharCopy(const Tin& in) -> Tout
{
    static_assert(sizeof(Tin) == sizeof(Tout), "Sizes must agree");

    Tout out; // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto* outc = reinterpret_cast<char8_t*>(&out); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto* inc = reinterpret_cast<const char8_t*>(&in); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)

    for (uint32_t i{0}; i < sizeof(Tin); i++)
    {
        outc[i] = inc[i];
    }
    return out;
}

namespace detail
{

/**
 * @brief Primary class template of @c GetFundamentalType.  The aim of this class is to provide an interface which
 *        allows to extract the fundamental type of a structure or class.
 *
 * @tparam T     The class/struct whose fundamental type is to be determined.
 */
template <typename T, class = void>
struct GetFundamentalType
{
    using type = void;
};

/**
 * @brief Specialization of @c GetFundamentalType for fundamental types, e.g. int32_t, float32_t, etc.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<std::is_fundamental<T>::value>>
{
    using type = T;
};

/// @brief Primary class template to check if a type has a @c value_type variable.
template <typename T, class = void>
struct HasValueType : std::false_type
{
};

/// @brief SFINAE helper to check if a type has a @c value_type member variable.
template <typename T>
struct HasValueType<T, dw::core::void_t<typename T::value_type>> : std::true_type
{
};

/**
 * @brief Specialization of @c GetFundamentalType for types that have a @c value_type, e.g. stl @c std::vector<int>
 *        has fundamental type @c int.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<HasValueType<T>::value>>
{
    using type = typename std::decay_t<typename T::value_type>;
};

/// @brief Primary class template to check if a type has a @c Scalar variable.
template <typename T, class = void>
struct HasScalar : std::false_type
{
};

/// @brief SFINAE helper to check if a type has a @c Scalar variable.
template <typename T>
struct HasScalar<T, dw::core::void_t<typename T::Scalar>> : std::true_type
{
};

/**
 * @brief Specialization of @c GetFundamentalType for types that have a @c Scalar.  This covers all matrix classes defined in
 *        src/dwshared/dwfoundation/dw/core/matrix as well as Eigen vector and matrix types.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<HasScalar<T>::value && !HasValueType<T>::value>>
{
    using type = typename std::decay_t<typename T::Scalar>;
};

/// @brief Primary class template to check if a type has a @c x member variable.
template <typename T, class = void>
struct HasX : std::false_type
{
};

/// @brief SFINAE helper to check if a type has a @c x member variable.
template <typename T>
struct HasX<T, dw::core::void_t<decltype(T::x)>> : std::true_type
{
};

/**
 * @brief Specialization of @c GetFundamentalType for types that have a @c x member variable.  This covers dwVectorXX structs.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<HasX<T>::value && !HasValueType<T>::value && !HasScalar<T>::value>>
{
    using type = typename std::decay_t<decltype(T::x)>;
};

/// @brief Primary class template to check if a type has an @c array member array.
template <typename T, class = void>
struct HasArray : std::false_type
{
};

/// @brief SFINAE helper to check if a type has an @c array member array.
template <typename T>
struct HasArray<T, dw::core::void_t<decltype(T::array)>> : std::true_type
{
};

/**
 * @brief Specialization of @c GetFundamentalType for types that have an @c array member array.  This covers dwMatrixXX,
          and dwTransformationXX structs.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<HasArray<T>::value && !HasValueType<T>::value && !HasScalar<T>::value && !HasX<T>::value>>
{
    using type = typename std::decay_t<std::remove_pointer_t<std::decay_t<decltype(T::array)>>>;
};

/// @brief Primary class template to check if a type has an @c array member array.
template <typename T, class = void>
struct HasConfidence : std::false_type
{
};

/// @brief SFINAE helper to check if a type has an @c array member array.
template <typename T>
struct HasConfidence<T, dw::core::void_t<decltype(T::confidence)>> : std::true_type
{
};

/**
 * @brief Specialization of @c GetFundamentalType for types that have an @c array member array.  This covers dwMatrixXX,
          and dwTransformationXX structs.
 */
template <typename T>
struct GetFundamentalType<T, typename std::enable_if_t<HasConfidence<T>::value && !HasArray<T>::value && !HasValueType<T>::value && !HasScalar<T>::value && !HasX<T>::value>>
{
    using type = typename std::decay_t<decltype(T::confidence)>;
};

} // namespace detail

/**
 * @brief Safe reinterpreting of a pointer. Enforces that both types must have the same sizeof().
 * 
 * 1. The function uses static_assert to perform compile-time assertions. They ensure that ToutPointer and TinPointer are both pointer types. If either of them is not a pointer type, a compile-time error will be triggered. <br>
 * 2. The function uses std::remove_pointer to remove the pointer modifiers from ToutPointer and TinPointer, resulting in the types Tout and Tin, respectively. <br>
 * 3. The function uses static_assert to check if the sizes of Tout and Tin are equal. If the sizes are different, a compile-time error will be triggered. <br>
 * 4. The function does the runtime alignment check. If the alignment does not match, a BadAlignmentException is thrown. <br>
 * 5. The function reinterprets @b ptr as a pointer of type ToutPointer and returns the resulting pointer.
 *
 * @tparam ToutPointer The output pointer type.
 * @tparam TinPointer The input pointer type.
 * @tparam doReinterpretFloatIntCheck Boolean to enable or disable reinterpret_cast between integer and floating point types
 * @param[in] ptr The pointer to be cast.
 *
 * @throw BadAlignmentException if type alignments do not match
 * 
 * @return The output pointer.
*/
// Note: Tout and Tin are not explicitly pointers to keep the same interface as reinterpret_cast<Tout>()
// Example:
//      dwVector2ui *a;
//      dw::Vector2ui *aCpp = safeReinterpretCast<dw::Vector2ui *>(a);

// TODO(dwplc):  This function is based on reinterpret_cast, which violates rule AUTOSAR A5-2-4.
//               A strategy is needed for how to resolve all the places it is called across the project.
template <class ToutPointer, class TinPointer, bool doReinterpretFloatIntCheck = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto safeReinterpretCast(TinPointer const ptr) -> ToutPointer
{
    static_assert(std::is_pointer<ToutPointer>::value, "Tout must be a pointer type");
    static_assert(std::is_pointer<TinPointer>::value, "Tin must be a pointer type");

    using Tout = typename std::remove_pointer<ToutPointer>::type;
    using Tin  = typename std::remove_pointer<TinPointer>::type;
    static_assert(sizeof(Tout) == sizeof(Tin), "Size of input and output types do not match");

    using Fout = typename detail::GetFundamentalType<Tout>::type;
    using Fin  = typename detail::GetFundamentalType<Tin>::type;
    constexpr bool IS_FLOAT_TO_INT_CAST{std::is_floating_point<Fin>::value && std::is_integral<Fout>::value};
    constexpr bool IS_BAD_CAST{doReinterpretFloatIntCheck && IS_FLOAT_TO_INT_CAST};
    static_assert(!IS_BAD_CAST, "Reinterpret cast is not allowed between integer and floating point types!");

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
 * @brief Safe conversion of strongly-typed enumeration instances to references of their underlying type.
 *
 * 1. The function uses std::enable_if and std::is_enum for compile-time conditional checking to ensure that EnumT is indeed an enumeration type. If EnumT is not an enumeration type, a compile-time error will occur. <br>
 * 2. The std::underlying_type is used to obtain the underlying type of EnumT, and the typename keyword is used to declare its type as typename std::underlying_type<EnumT>::type&, which is a reference to the underlying type. <br>
 * 3. The reinterpret_cast is used to reinterpret the enumeration value @b e as a reference to its underlying type.
 *
 * @tparam EnumT the enum type
 * @param[in] e The enum value to parse.
 * 
 * @return Reference of the underlying type.
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

/**
 * @brief Safe conversion of strongly-typed enumeration instances to pointers of their underlying type.
 * 
 * 1. The function uses std::enable_if and std::is_enum for compile-time conditional checking to ensure that EnumT is indeed an enumeration type. If EnumT is not an enumeration type, a compile-time error will occur. <br>
 * 2. The std::underlying_type is used to obtain the underlying type of EnumT, and the typename keyword is used to declare its type as typename std::underlying_type<EnumT>::type*, which is a pointer to the underlying type. <br>
 * 3. The reinterpret_cast is used to reinterpret the enumeration value @b e as a pointer to its underlying type.
 *
 * @tparam EnumT the enum type
 * @param[in] e The enum value to parse.
 * 
 * @return Pointer pointingto the underlying type.
 */
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
 * @brief Static type conversion of *fully* evaluated values / results of expressions.
 * This is to prevent unintentionally conversions of cvalue expressions.
 * 
 * @tparam T The type to be converted.
 * @tparam U The destination type.
 * @param[in] u the input value
 * 
 * @return the converted value in the type T.
 */
template <typename T, typename U>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr auto as(U&& u) -> T
{
    return static_cast<T>(std::forward<U>(u));
}

/** @}*/

/** \addtogroup basic_types_structs
 *  @{
 */

/**
 * @brief Compile-time TRUE condition to test at compile time without triggering a warning.
 */
template <bool b>
struct static_condition // clang-tidy NOLINT(readability-identifier-naming)
{
    /// Return true
    static bool test()
    {
        return true;
    }
};
/**
 * @brief Compile-time FALSE condition to test at compile time without triggering a warning.
 */
template <>
struct static_condition<false>
{
    /// Returns false
    static bool test()
    {
        return false;
    }
};

/**
 * @brief Compile time condition to determine if a class is a specialization of a different class.
 * ie std::tuple<int,float> is specialization of std::tuple
*/
template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

/**
 * @brief Compile time condition to determine if a class is a specialization of a different class,
 * when instantiated with specific template arguments Args....
*/
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type
{
};

struct IsDerivedFrom
{
    // functions are grouped in this struct to avoid autosar_cpp14_a2_10_4_violation

    template <class Derived, template <class... Ts> class Base, class... Ts>
    constexpr static std::true_type testFunction(const Base<Ts...>*);

    template <class Derived, template <class...> class Base>
    constexpr static std::false_type testFunction(...);
};

/**  @brief Compile time condition to determine if a class is publically derived from a single class template.
 *          May yield unexpected results if the class inherits from multiple instantiations of the template.
 */
template <class Derived, template <class...> class Base>
struct is_derived_from_instantiation_of // clang-tidy NOLINT(readability-identifier-naming)
    : std::integral_constant<bool,
                             decltype(IsDerivedFrom::testFunction<Derived, Base>(std::declval<Derived*>()))::value>
{
};

/**
 * @brief Helper variable template for is_derived_from_instantiation_of::value. See its definition.
*/
template <class Derived, template <class...> class Base>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
constexpr bool is_derived_from_instantiation_of_v{// clang-tidy NOLINT(readability-identifier-naming)
                                                  is_derived_from_instantiation_of<Derived, Base>::value};

/**
 * @brief Traits helper struct to copy the constness of one type onto the other.
 * The resulting type will be "const T" if TRef is const, else "T".
 * 
 * @tparam T The type to modify.
 * @tparam TRef The type to refer.
*/
// For example:
//    using AC = copy_const<A, const uint32_t>::type   // AC == const A
//    using AN = copy_const<A, uint32_t>::type         // AN == A
template <class T, class TRef>
struct copy_const // clang-tidy NOLINT(readability-identifier-naming)
{
public:
    /// If TRef is const, add constness to type T, otherwise remove it.
    using type = typename std::conditional<
        std::is_const<TRef>::value,
        typename std::add_const<T>::type,
        typename std::remove_const<T>::type>::type;
};
/**
 * @brief This struct offers a simple way to access the type information of a given type.
 * 
 * @tparam T The given type.
*/
///////////////////////////////////////////
// This struct provides information that is not covered by std::type_info
//
// Scalar: for simple types T, for Matrix<T,...> types T, for Jet<T,...> types T
// Example usage:
//    void foo(T a)
//    {
//       using Scalar = typename TypeInfo<T>::Scalar;
//       Scalar threshold = Scalar{0.1f};
//       // If T is an autodiff::Jet, threshold will be scalar so no derivatives are used for it
//       ...
//    }
template <class T>
struct TypeInfo
{
    using Scalar = T;
};

/** @}*/

} // namespace core
} // namespace dw

#endif // DW_CORE_TYPES_HPP_
