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
// Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DWCBOR_DW_CBOR_CODES_HPP_
#define DWCBOR_DW_CBOR_CODES_HPP_

#include <dwshared/dwfoundation/dw/core/language/cxx17.hpp>
#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>

#include <array>
#include <cstdint>
#include <type_traits>

namespace dw
{
namespace cbor
{

/// Magic numbers used in CBOR.
struct Codes final
{
    /// The portion of the leading byte which contains the major type.
    static constexpr std::uint8_t MAJOR_TYPE_MASK{0xe0U};

    /// The portion of the leading byte which does not contain the major type -- referred to as the "short count" by the
    /// CBOR specification.
    static constexpr std::uint8_t SHORT_COUNT_MASK{0x1fU};

    /// The major type for integer values in CBOR are designated by a special code. Values at or under @c DIRECT_MAX are
    /// encoded as a single byte, offset by the @c BASE_OFFSET value. For example, a string of length @c 5 is encoded as
    /// the byte @c 0x65 -- @c 0x60 for the base offset of the string major type plus @c 5 to denote the value. Values
    /// greater than @c DIRECT_MAX are encoded with a sigil denoting their run length, then are big-endian encoded. For
    /// example, an array with length @c 35 would be encoded as the byte sequence @c 0x98 @c 0x23 -- @c 0x98 is the
    /// sigil for an array with length encoded as an 8-bit integer, then the single byte @c 0x23 for the length (the
    /// content of the array follows this introducer).
    template <std::size_t KBaseOffset,
              std::size_t KUInt8Mark,
              std::size_t KUInt16Mark = KUInt8Mark + 1U,
              std::size_t KUInt32Mark = KUInt8Mark + 2U,
              std::size_t KUInt64Mark = KUInt8Mark + 3U>
    struct BasicPrefixes
    {
        static_assert((KBaseOffset & ~static_cast<std::size_t>(MAJOR_TYPE_MASK)) == 0U, "Base offset must be a major type");

        /// The base offset to add to the directly-encoded values.
        static constexpr std::size_t BASE_OFFSET{KBaseOffset};

        /// Magic number to indicate that an 8-bit number follows.
        static constexpr std::uint8_t SIGIL_8_BIT{KUInt8Mark};

        /// Magic number to indicate that a 16-bit number follows.
        static constexpr std::uint8_t SIGIL_16_BIT{KUInt16Mark};

        /// Magic number to indicate that a 32-bit number follows.
        static constexpr std::uint8_t SIGIL_32_BIT{KUInt32Mark};

        /// Magic number to indicate that a 64-bit number follows.
        static constexpr std::uint8_t SIGIL_64_BIT{KUInt64Mark};
    };

    /// Magic value for @c null.
    ///
    /// @see RawWriter::writeNull
    static constexpr std::uint8_t NULL_SIGIL{0xf6U};

    /// Magic value for @c true.
    ///
    /// @see RawWriter::writeBool
    static constexpr std::uint8_t TRUE_SIGIL{0xf5U};

    /// Magic value for @c false.
    ///
    /// @see RawWriter::writeBool
    static constexpr std::uint8_t FALSE_SIGIL{0xf4U};

    /// Magic value for IEEE-754 16-bit floating-point value.
    ///
    /// @see RawWriter::writeFloat
    static constexpr std::uint8_t FLOAT16_SIGIL{0xf9U};

    /// Magic value for IEEE-754 32-bit floating-point value.
    ///
    /// @see RawWriter::writeFloat
    static constexpr std::uint8_t FLOAT32_SIGIL{0xfaU};

    /// Magic value for IEEE-754 64-bit floating-point value.
    ///
    /// @see RawWriter::writeFloat
    static constexpr std::uint8_t FLOAT64_SIGIL{0xfbU};

    /// This type is used to encode the special values of floating-point numbers (NaN and infinity values).
    using SpecialFloatSigilType = std::array<std::uint8_t, 3U>;

    /// Magic value for a NaN float value.
    static constexpr SpecialFloatSigilType FLOAT_NAN_SIGIL{{
        static_cast<std::uint8_t>(0xf9U),
        static_cast<std::uint8_t>(0x7eU),
        static_cast<std::uint8_t>(0x00U),
    }};

    /// Magic value for the positive infinity float value.
    static constexpr SpecialFloatSigilType FLOAT_POSITIVE_INFINITY_SIGIL{{
        static_cast<std::uint8_t>(0xf9U),
        static_cast<std::uint8_t>(0x7cU),
        static_cast<std::uint8_t>(0x00U),
    }};

    /// Magic value for the negative infinity float value.
    static constexpr SpecialFloatSigilType FLOAT_NEGATIVE_INFINITY_SIGIL{{
        static_cast<std::uint8_t>(0xf9U),
        static_cast<std::uint8_t>(0xfcU),
        static_cast<std::uint8_t>(0x00U),
    }};

    /// Integer values @c <= to this are encoded directly.
    static constexpr std::size_t DIRECT_MAX{0x17U};

    /// Magic numbers for positive integers.
    using PositiveIntPrefixes = BasicPrefixes<0x00U, 0x18U>;

    /// Magic numbers for negative integers.
    using NegativeIntPrefixes = BasicPrefixes<0x20U, 0x38U>;

    /// Magic numbers for the encoded length of a binary blob.
    using BlobPrefixes = BasicPrefixes<0x40U, 0x58U>;

    /// Magic numbers for the encoded length of a UTF-8 string.
    using StringPrefixes = BasicPrefixes<0x60U, 0x78U>;

    /// Magic numbers for the encoded length of an array.
    using ArraySizePrefixes = BasicPrefixes<0x80U, 0x98U>;

    /// Magic numbers for the encoded length of a map.
    using MapSizePrefixes = BasicPrefixes<0xa0U, 0xb8U>;

    /// Magic numbers for writing semantic tags.
    using SemanticTagPrefixes = BasicPrefixes<0xc0U, 0xd8U>;
};

/// When @c TFloat is a supported floating-point type, this structure has three members:
///
/// * @c FloatType -- A member type equal to @c TFloat
/// * @c UIntReprType -- A member type which is an unsigned integer with the same number of bytes as @c FloatType
/// * @c SIGIL -- A member constexpr @c uint8_t value which has the @c Codes sigil for the type
///
/// @see HasFloatCodeTraits
template <typename TFloat>
struct FloatCodeTraits
{
};

namespace detail
{

/// Helper structure for implementing @c FloatCodeTraits.
template <typename TFloat, typename TUIntRepr, std::uint8_t KSigil>
struct FloatCodeTraitsImpl
{
    using FloatType = TFloat;

    using UIntReprType = TUIntRepr;

    static constexpr std::uint8_t SIGIL{KSigil};
};

} // namespace dw::cbor::detail

template <>
struct FloatCodeTraits<float32_t> : detail::FloatCodeTraitsImpl<float32_t, std::uint32_t, Codes::FLOAT32_SIGIL>
{
};

template <>
struct FloatCodeTraits<float64_t> : detail::FloatCodeTraitsImpl<float64_t, std::uint64_t, Codes::FLOAT64_SIGIL>
{
};

namespace detail
{

template <typename TFloat, typename = core::void_t<>>
struct HasFloatCodeTraitsImpl : std::false_type
{
};

// clang-format off
template <typename TFloat>
struct HasFloatCodeTraitsImpl
<
    TFloat,
    core::void_t
    <
        typename FloatCodeTraits<TFloat>::FloatType,
        typename FloatCodeTraits<TFloat>::UIntReprType,
        std::enable_if_t<std::is_convertible<decltype(FloatCodeTraits<TFloat>::SIGIL), std::uint8_t>::value>
    >
> : std::true_type
{ };
// clang-format on

} // namespace dw::cbor::detail

/// Check if the @c TFloat has a valid @c FloatCodeTraits implementation.
///
/// @see HAS_FLOAT_CODE_TRAITS_V
template <typename TFloat>
struct HasFloatCodeTraits : detail::HasFloatCodeTraitsImpl<TFloat>
{
};

/// @see HasFloatCodeTraits
template <typename TFloat>
constexpr bool HAS_FLOAT_CODE_TRAITS_V = HasFloatCodeTraits<TFloat>::value;

} // namespace dw::cbor
} // namespace dw

#endif /*DWCBOR_DW_CBOR_CODES_HPP_*/
