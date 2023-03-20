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

#ifndef DWCBOR_DW_CBOR_TOKEN_HPP_
#define DWCBOR_DW_CBOR_TOKEN_HPP_

#include <dw/core/container/Span.hpp>
#include <dw/core/container/StringView.hpp>
#include <dw/core/language/cxx17.hpp>
#include <dw/core/language/Optional.hpp>
#include <dw/core/language/CheckedIntegerCast.hpp>
#include <dw/core/language/Tag.hpp>

#include <cstdint>

#include "Codes.hpp"

namespace dw
{
namespace cbor
{

/// The "major type" of CBOR encoding is used to select the behavior of the encoded item.
///
/// @note
/// The numbers of this enumeration do not match the CBOR specification (which uses 0-7). Instead, they are shifted
/// left by 5 bits. This allows extraction from a byte by a mask operation rather than a mask and shift.
enum class MajorType : std::uint8_t
{
    /// The token is an integer at or above @c 0.
    POSITIVE_INT = Codes::PositiveIntPrefixes::BASE_OFFSET,
    /// The token is an integer below @c 0.
    NEGATIVE_INT = Codes::NegativeIntPrefixes::BASE_OFFSET,
    /// The token introduces a byte string. The encoded section of the token contains the length of the blob, which
    /// immediately follows the token.
    BLOB = Codes::BlobPrefixes::BASE_OFFSET,
    /// The token introduces a UTF-8 character string. The encoded section of the token contains the number of bytes
    /// (code units in Unicode parlance) in the string, which immediately follows the token.
    STRING = Codes::StringPrefixes::BASE_OFFSET,
    /// This token introduces an array. The encoded section of the token contains the number of elements in the array.
    /// The next @e n data elements which follow this token are considered part of the array.
    ARRAY = Codes::ArraySizePrefixes::BASE_OFFSET,
    /// This token introduces a map. The encoded section of the token contains the number of key-value pairs in the map.
    /// The next @e 2*n data elements which follow this token are key-value pairs, with the even-indexed elements
    /// serving as keys and odd-indexed elements serving as values.
    MAP = Codes::MapSizePrefixes::BASE_OFFSET,
    /// The token is a semantic tag, which behaves identically to @c POSITIVE_INT, but informs the reader that the next
    /// data element should have a special interpretation. For example, a semantic tag with the value of @c 1 should be
    /// followed by a number which is interpreted as a UNIX epoch-based timestamp. The full list of CBOR semantic tags
    /// is managed by <a href="https://www.iana.org/assignments/cbor-tags/cbor-tags.xhtml">the IANA</a>.
    SEMANTIC = Codes::SemanticTagPrefixes::BASE_OFFSET,
    /// Special tags are used for encoding types which do not fit into other major type categories. Their meaning is
    /// based on the short code value. This is where @c true, @c false, and @c null sigils live, as well as floating
    /// point numbers of various precisions.
    SPECIAL = 0xe0U,
};

/// Convert @a src to its integer representation.
constexpr std::uint8_t toInteger(MajorType const src) noexcept
{
    return static_cast<std::uint8_t>(src);
}

/// Extract @a src from its integer representation.
constexpr MajorType fromInteger(core::Tag<MajorType>, std::uint8_t const src) noexcept
{
    // TODO(dwplc): FP -- Coverity is incorrect here, as src & MAJOR_TYPE_MASK is always covered by MajorType
    // coverity[autosar_cpp14_a7_2_1_violation]
    // coverity[cert_int31_c_violation]
    return static_cast<MajorType>(static_cast<std::uint8_t>(src & Codes::MAJOR_TYPE_MASK));
}

/// Get a string representation of the @a src type.
core::StringView getName(MajorType const src) noexcept;

namespace detail
{

/// Throws @c InvalidArgumentException with the given @a value.
[[noreturn]] void throwTokenShortCountInvalid(std::uint8_t const value) noexcept(false);

/// Throws @c InvalidArgumentException when a trailing buffer was specified for a @c MajorType which is not allowed to
/// have one.
[[noreturn]] void throwTokenNoTrailingBufferForMajorType(MajorType const actualType) noexcept(false);

/// Throws @c InvalidArgumentException when calling a @c Token::as when the target type (the returned type, as specified
/// in the tag parameter) is not compatible with the token's major type.
[[noreturn]] void throwTokenInvalidConversionForAs(MajorType const actualType,
                                                   core::Optional<MajorType> const& expectedType) noexcept(false);

/// Throws @c InvalidArgumentException when calling a @c Token::as function when the target type (the returned type, as
/// specified in the tag parameter) requires a special type, but has the wrong short count.
[[noreturn]] void throwTokenInvalidSpecialShortCount(core::StringView const targetType,
                                                     std::uint8_t const shortCount) noexcept(false);

} // namespace dw::cbor::detail

/// A token represents a CBOR encoding element -- a @c MajorType, a @ref shortCount, and optional additional encoded
/// information.
class Token final
{
public:
    /// The buffer type for encoded runs.
    using BufferType = core::span1ub_const;

    /// The size type to use. This will always be a 64-bit unsigned integer.
    using SizeType = BufferType::size_type;
    static_assert(std::is_same<SizeType, std::uint64_t>::value, "Buffer size is expected to be uint64_t");

public:
    /// @param srcMajor The @ref majorType of the token.
    /// @param srcShortCount The @ref shortCount of the token. This must fit within the @c Codes::SHORT_COUNT_MASK or an
    ///                      @c InvalidArgumentException will be thrown.
    /// @param srcEncoded The @ref encoded buffer which completes this token.
    /// @param srcTrailingBuffer The buffer which follows this token. This is only allowed when the @a srcMajor type is
    ///                          @c STRING or @c BLOB -- an @c InvalidArgumentException is thrown for all other cases.
    explicit constexpr Token(MajorType const srcMajor,
                             std::uint8_t const srcShortCount,
                             // TODO(dwplc): FP -- Coverity claims this default value for the parameter is using NULL as
                             //              a null-pointer-constant, but that is not happening here.
                             // coverity[autosar_cpp14_a4_10_1_violation]
                             BufferType srcEncoded = BufferType{},
                             // TODO(dwplc): FP -- same as above
                             // coverity[autosar_cpp14_a4_10_1_violation]
                             BufferType srcTrailingBuffer = BufferType{})
        : Token{in_place, static_cast<std::uint8_t>(toInteger(srcMajor) | srcShortCount), srcShortCount, srcEncoded, srcTrailingBuffer}
    {
        constexpr std::uint8_t const ZERO{0U};
        // TODO(dwplc): FP -- both srcShortCount and SHORT_COUNT_MASK are uint8_ts, so truncating the result of bitwise
        //              not will not result in data loss.
        // coverity[cert_int31_c_violation]
        if (static_cast<std::uint8_t>(srcShortCount & static_cast<std::uint8_t>(~Codes::SHORT_COUNT_MASK)) != ZERO)
        {
            detail::throwTokenShortCountInvalid(srcShortCount);
        }

        if (!srcTrailingBuffer.empty())
        {
            if ((srcMajor != MajorType::BLOB) && (srcMajor != MajorType::STRING))
            {
                detail::throwTokenNoTrailingBufferForMajorType(srcMajor);
            }
        }
    }

    /// The default initialization creates a @c MajorType::SPECIAL token with an invalid @c shortCount.
    explicit constexpr Token() noexcept
        : Token{in_place, INVALID_SHORT_COUNT, INVALID_SHORT_COUNT, BufferType{}, BufferType{}}
    {
    }

    Token(Token const&) = default;
    Token(Token&&)      = default;

    Token& operator=(Token const&) = default;
    Token& operator=(Token&&) = default;

    ~Token() = default;

    /// Parse the @a source buffer into a token.
    ///
    /// To see the number of bytes parsed from @a source, see the @ref tokenSize of the returned value.
    static Token parse(BufferType const source);

    /// The major type which dictates how this token should be interpreted. See the documentation of the enumeration
    /// constants of @c MajorType for more information.
    constexpr MajorType majorType() const noexcept
    {
        return fromInteger(TAG<MajorType>, m_leadingByte);
    }

    /// The exact meaning of the short count depends on the value of @ref majorType, but for every type which is not
    /// @c MajorType::SPECIAL, the number is interpreted similarly.
    ///
    /// * [0 .. 23]: This is a "tiny field." The numeric value of this token is exactly this value.
    /// * [24 .. 27]: The @c encoded string is 2^{shortCount - 23} bytes long and should be interpreted as a big-endian
    ///   value.
    /// * [28 .. 30]: These are reserved for a future version of CBOR (as of RFC 8949).
    /// * 31: For @c BLOB, @c STRING, @c ARRAY, and @c MAP major types, this denotes that the item's length is
    ///   indefinite. For @c SPECIAL, this denotes that the previous indefinite-length item is terminated. This value is
    ///   not legal to use for @c POSITIVE_INT, @c NEGATIVE_INT, and @c SEMANTIC major types.
    constexpr std::uint8_t shortCount() const noexcept
    {
        return m_shortCount;
    }

    /// Either the encoded buffer which follows the leading byte (the major type and short count) or, in the case of a
    /// tiny field, this points to the short count.
    constexpr BufferType encoded() const noexcept
    {
        if (m_encodedBuffer.empty() && (m_shortCount <= Codes::DIRECT_MAX))
        {
            constexpr SizeType const ONE{1U};
            return {&m_shortCount, ONE};
        }
        else
        {
            return m_encodedBuffer;
        }
    }

    /// Get the trailing section of this token. This is only non-empty if the @ref majorType is @c STRING or @c BLOB.
    constexpr BufferType trailing() const noexcept
    {
        return m_trailingBuffer;
    }

    /// Get the size of the encoded token this instance represents. This includes the introducer byte, which encodes the
    /// major and short count.
    constexpr SizeType tokenSize() const noexcept
    {
        if (!valid())
        {
            constexpr SizeType const ZERO{0U};
            return ZERO;
        }

        constexpr SizeType const ONE{1U};
        SizeType out{ONE};

        // TODO(dwplc): RFD -- this will never overflow, as it would mean that the buffers have more than 2^64 bytes,
        //              which means we have run out of memory.
        // coverity[cert_int30_c_violation]
        out += m_encodedBuffer.size_bytes();
        // coverity[cert_int30_c_violation]
        out += m_trailingBuffer.size_bytes();

        return out;
    }

    /// Check if this token is a valid token (it has not been default-constructed).
    constexpr bool valid() const noexcept
    {
        return m_shortCount != INVALID_SHORT_COUNT;
    }

    /// Extract an unsigned integer from this token.
    template <typename TUnsignedInt, std::enable_if_t<std::is_unsigned<TUnsignedInt>::value, bool> = true>
    TUnsignedInt as(core::Tag<TUnsignedInt>) const
    {
        if (majorType() != MajorType::POSITIVE_INT)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::POSITIVE_INT);
        }

        return extractInt(TAG<TUnsignedInt>);
    }

    /// Extract a signed integer from this token.
    template <typename TSignedInt, std::enable_if_t<std::is_signed<TSignedInt>::value, bool> = false>
    TSignedInt as(core::Tag<TSignedInt>) const
    {
        if ((majorType() != MajorType::POSITIVE_INT) && (majorType() != MajorType::NEGATIVE_INT))
        {
            detail::throwTokenInvalidConversionForAs(majorType(), core::NULLOPT);
        }

        using UnsignedInt           = std::make_unsigned_t<TSignedInt>;
        UnsignedInt const extracted = extractInt(TAG<UnsignedInt>);

        if (majorType() == MajorType::POSITIVE_INT)
        {
            return checkedIntegerCast<TSignedInt>(extracted);
        }
        else // negative integer
        {
            // TODO(dwplc): FP -- if TSignedInt is a type definition which includes size and signedness information (per
            //              AUTOSAR requirements), then TSignedInt is that type.
            // coverity[autosar_cpp14_a3_9_1_violation]
            TSignedInt const codedValue = checkedIntegerCast<TSignedInt>(extracted);

            // TODO(dwplc): FP -- if TSignedInt is a type definition which includes size and signedness information (per
            //              AUTOSAR requirements), then TSignedInt is that type.
            // coverity[autosar_cpp14_a3_9_1_violation]
            return static_cast<TSignedInt>(static_cast<TSignedInt>(-1) - codedValue);
        }
    }

    /// @{
    /// Get a view of the UTF-8 encoded string attached to this token.
    core::StringView as(core::Tag<core::StringView>) const &
    {
        if (majorType() != MajorType::STRING)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::STRING);
        }

        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return core::StringView{reinterpret_cast<char8_t const*>(trailing().data()), trailing().size()};
    }

    /// This overload is disabled on rvalues because it returns a non-owning view.
    core::StringView as(core::Tag<core::StringView>) && = delete;
    /// @}

    /// @{
    /// Get a view of the data blob attached to this token.
    BufferType as(core::Tag<BufferType>) const &
    {
        if (majorType() != MajorType::BLOB)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::BLOB);
        }

        return trailing();
    }

    /// This overload is disabled on rvalues because it returns a non-owning view.
    BufferType as(core::Tag<BufferType>) && = delete;
    /// @}

    /// @{
    /// Extract a floating-point value from this token.
    float32_t as(core::Tag<float32_t>) const;
    float64_t as(core::Tag<float64_t>) const;
    /// @}

    /// Extract a boolean value from this token.
    bool as(core::Tag<bool>) const
    {
        if (majorType() != MajorType::SPECIAL)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::SPECIAL);
        }
        else if (m_leadingByte == Codes::TRUE_SIGIL)
        {
            return true;
        }
        else if (m_leadingByte == Codes::FALSE_SIGIL)
        {
            return false;
        }
        else
        {
            detail::throwTokenInvalidSpecialShortCount("bool", shortCount());
        }
    }

    /// Extract a null value. This is barely useful, since a null value is always null.
    std::nullptr_t as(core::Tag<std::nullptr_t>) const
    {
        if (majorType() != MajorType::SPECIAL)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::SPECIAL);
        }
        else if (m_leadingByte == Codes::NULL_SIGIL)
        {
            return nullptr;
        }
        else
        {
            detail::throwTokenInvalidSpecialShortCount("nullptr", shortCount());
        }
    }

    /// Get the semantic tag value of this token.
    template <typename TUnsignedInt, std::enable_if_t<std::is_unsigned<TUnsignedInt>::value, bool> = true>
    TUnsignedInt semantic(core::Tag<TUnsignedInt>) const
    {
        if (majorType() != MajorType::SEMANTIC)
        {
            detail::throwTokenInvalidConversionForAs(majorType(), MajorType::SEMANTIC);
        }

        return extractInt(core::TAG<TUnsignedInt>);
    }

    /// Get the element count of an @c ARRAY or @c MAP token.
    ///
    /// @returns An engaged optional with the length of the array or map if it has a fixed length; @c NULLOPT if the
    ///          token is an array or map with indefinite length.
    core::Optional<SizeType> elementCount() const
    {
        if ((majorType() != MajorType::ARRAY) && (majorType() != MajorType::MAP))
        {
            detail::throwTokenInvalidConversionForAs(majorType(), core::NULLOPT);
        }

        if (shortCount() == Codes::SHORT_COUNT_MASK)
        {
            return core::NULLOPT;
        }
        else
        {
            return extractInt(TAG<SizeType>);
        }
    }

private:
    /// Used to denote a default initialized token.
    static constexpr std::uint8_t INVALID_SHORT_COUNT{static_cast<std::uint8_t>(0xffU)};

    /// Landing constructor used for forwarding from user-facing constructors.
    explicit constexpr Token(in_place_t,
                             std::uint8_t const srcLeadingByte,
                             std::uint8_t const srcShortCount,
                             BufferType srcEncoded,
                             BufferType srcTrailing) noexcept
        : m_leadingByte{srcLeadingByte}
        , m_shortCount{srcShortCount}
        , m_encodedBuffer{srcEncoded}
        , m_trailingBuffer{srcTrailing}
    {
    }

    /// @{
    std::uint8_t extractInt(Tag<std::uint8_t>) const;
    std::uint16_t extractInt(Tag<std::uint16_t>) const;
    std::uint32_t extractInt(Tag<std::uint32_t>) const;
    std::uint64_t extractInt(Tag<std::uint64_t>) const;
    /// @}

private:
    /// Contains both the major type and the short count.
    std::uint8_t m_leadingByte;

    /// The masked short count. In cases where the encoded token is a tiny field, the buffer will point to this field.
    std::uint8_t m_shortCount;

    /// The complete raw token.
    BufferType m_encodedBuffer;

    BufferType m_trailingBuffer;
};

} // namespace dw::cbor
} // namespace dw

#endif /*DWCBOR_DW_CBOR_TOKEN_HPP_*/
