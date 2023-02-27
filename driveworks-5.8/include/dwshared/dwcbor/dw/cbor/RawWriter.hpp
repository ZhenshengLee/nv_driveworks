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
// Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DWCBOR_DW_CBOR_RAWWRITER_HPP_
#define DWCBOR_DW_CBOR_RAWWRITER_HPP_

#include <dw/core/container/Span.hpp>
#include <dw/core/container/StringView.hpp>
#include <dw/core/language/TypeAliases.hpp>
#include <dw/core/utility/Endian.hpp>

#include <cmath>
#include <cstring>
#include <type_traits>

#include "Codes.hpp"
#include "Forwards.hpp"

namespace dw
{
namespace cbor
{
namespace detail
{

/// Throws @c OutOfBoundsException with a message denoting which @a operation threw and the writer's @a bufferSize.
[[noreturn]] void throwRawWriterOutOfBounds(core::StringView const operation, std::size_t const bufferSize);

} // namespace dw::cbor::detail

/// The raw interface used to write CBOR code. It encodes CBOR directly into a byte span. Take great care when using
/// this interface; beyond basic boundary checking, the safety is off. Misuse will result in an encoded byte string
/// which is invalid CBOR with no warning.
class RawWriter
{
public:
    /// The buffer type to encode into -- a byte string.
    using BufferType = core::span1ub;

    /// A @ref BufferType with constant elements.
    using ConstBufferType = core::span1ub_const;

    /// The size type to use. This will always be a 64-bit unsigned integer.
    using SizeType = BufferType::size_type;
    static_assert(std::is_same<SizeType, std::uint64_t>::value, "Buffer size is expected to be uint64_t");

public:
    /// Create a writer writing to the @a destination.
    ///
    /// @param destination The buffer to write into. This is simply a view -- it is the caller's responsibility to
    ///                    ensure the destination remains valid.
    explicit constexpr RawWriter(BufferType destination) noexcept
        : m_buffer{destination}
        , m_offset{0U}
    {
    }

    // Copying and moving are deleted, as it is likely always an error to do so.
    RawWriter(RawWriter const&) = delete;
    RawWriter(RawWriter&&)      = delete;

    RawWriter& operator=(RawWriter const&) = delete;
    RawWriter& operator=(RawWriter&&) = delete;

    ~RawWriter() = default;

    /// Get the encoded section of the buffer. If the various @c write functions have been used correctly, this will be
    /// a CBOR-encoded byte string.
    ConstBufferType encoded() const
    {
        return m_buffer.subspan(0U, m_offset);
    }

    /// Write the null equivalent value.
    void writeNull()
    {
        writeRaw(Codes::NULL_SIGIL);
    }

    /// Write the boolean @a value.
    void writeBool(bool const value)
    {
        writeRaw(value ? Codes::TRUE_SIGIL : Codes::FALSE_SIGIL);
    }

    /// Write the unsigned integer @a value.
    template <typename TUInt>
    std::enable_if_t<std::is_unsigned<TUInt>::value> writeInt(TUInt const value)
    {
        writeCompactUInt(Codes::PositiveIntPrefixes{}, value);
    }

    /// Write the signed integer @a value.
    template <typename TSInt>
    std::enable_if_t<std::is_signed<TSInt>::value> writeInt(TSInt const value)
    {
        using UnsignedType = std::make_unsigned_t<TSInt>;

        if (value >= 0)
        {
            // A signed integer above 0 is treated like an unsigned integer
            writeInt(static_cast<UnsignedType>(value));
        }
        else
        {
            // TODO(dwplc): FP -- Coverity believes TSInt declaration violates A3-9-1 by not being a basic numerical
            //              type. Assuming all the users of this function comply with A3-9-1, this will not be true.
            // coverity[autosar_cpp14_a3_9_1_violation]
            TSInt const convertedValue = static_cast<TSInt>(-1 - value);
            UnsignedType const coded   = static_cast<UnsignedType>(convertedValue);
            writeCompactUInt(Codes::NegativeIntPrefixes{}, coded);
        }
    }

    /// Write the floating-point @a value.
    template <typename TFloat>
    std::enable_if_t<HAS_FLOAT_CODE_TRAITS_V<TFloat>> writeFloat(TFloat const value)
    {
        using std::isnan;
        using std::isinf;

        if (isnan(value))
        {
            writeRawRange(Codes::FLOAT_NAN_SIGIL);
        }
        else if (isinf(value))
        {
            writeRawRange((value < static_cast<TFloat>(0)) ? Codes::FLOAT_NEGATIVE_INFINITY_SIGIL : Codes::FLOAT_POSITIVE_INFINITY_SIGIL);
        }
        else
        {
            using FloatCodeTraits = FloatCodeTraits<TFloat>;
            using UIntReprType    = typename FloatCodeTraits::UIntReprType;

            UIntReprType coded{};
            static_cast<void>(std::memcpy(&coded, &value, sizeof coded));

            writeRaw(FloatCodeTraits::SIGIL);
            writeRawRange(encodeBigEndian(coded));
        }
    }

    /// Write a semantic tag with the given @a value. The next item written is "tagged" with this special number. If
    /// nothing is written after this tag, the output will be invalid CBOR.
    template <typename TUInt>
    void writeSemantic(TUInt const value)
    {
        static_assert(std::is_unsigned<TUInt>::value, "Semantic tags must be unsigned integers");
        writeCompactUInt(Codes::SemanticTagPrefixes{}, value);
    }

    /// Write the string @a value.
    void writeString(core::StringView const value)
    {
        writeCompactUInt(Codes::StringPrefixes{}, value.size());
        writeRawRange(value);
    }

    /// Write the binary blob @a value.
    void writeBlob(span<std::uint8_t const> const value)
    {
        writeCompactUInt(Codes::BlobPrefixes{}, value.size());
        writeRawRange(value);
    }

    /// Begin an array with @a count elements. The next @a count items will be part of an array. There are no checks for
    /// if you actually write @a count items, but the output will not be valid CBOR.
    void beginArray(size_t const count)
    {
        writeCompactUInt(Codes::ArraySizePrefixes{}, count);
    }

    /// Begin a mapping with @a count key-value pairs. The next <tt>2 * count</tt> items will be the keys and values of
    /// a map type. Every even item (0, 2, 4, ...) is a key, while every odd item (1, 3, 5, ...) is a value. The keys
    /// do not have to be strings nor do they have to be unique, but this is recommended. There are no checks for if you
    /// actually write <tt>2 * count</tt> items, but the output will not be valid CBOR.
    void beginMap(size_t const count)
    {
        writeCompactUInt(Codes::MapSizePrefixes{}, count);
    }

private:
    /// Write a raw byte @a value to the output.
    void writeRaw(std::uint8_t const value)
    {
        if (m_offset >= m_buffer.size())
        {
            detail::throwRawWriterOutOfBounds("writeRaw", m_buffer.size());
        }
        else
        {
            m_buffer[m_offset] = value;
            ++m_offset;
        }
    }

    /// Write the raw bytes of the @a src range to the output.
    template <typename TRange>
    void writeRawRange(TRange const& src)
    {
        size_t const endOffset = m_offset + src.size();
        if ((endOffset < m_offset) || (endOffset >= m_buffer.size()))
        {
            detail::throwRawWriterOutOfBounds("writeRawRange", m_buffer.size());
        }
        else
        {
            static_cast<void>(std::memcpy(&m_buffer[m_offset], src.data(), src.size()));
            m_offset = endOffset;
        }
    }

    /// Write a compacted unsigned integer @a value with the type-specific @a prefixes.
    ///
    /// @tparam TPrefixes The prefix sigils used to mark the CBOR major type. See @c Codes::BasicPrefixes.
    /// @tparam T Some form of unsigned integer.
    template <typename TPrefixes, typename T>
    void writeCompactUInt(TPrefixes const prefixes, T const value)
    {
        static_assert(std::is_unsigned<T>::value, "T must be unsigned integer");
        static_assert(sizeof(T) <= sizeof(uint64_t), "Integers larger than 64-bit are not supported");

        if (value <= Codes::DIRECT_MAX)
        {
            writeRaw(static_cast<std::uint8_t>(value + prefixes.BASE_OFFSET));
        }
        // The "result_independent_of_operands" check tests that an if-conditional-expression will always be true or
        // false, regardless of the inputs to it. A uint32_t is always less than or equal to the maximum uint32_t, so
        // the final else block will never be reached. The Coverity check is disabled in these else-if branches as it
        // increases function readability to write it in this manner. The alternative would be to overload this function
        // for different T types, but that would lead to redundant code.
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint8_t>::max())
        {
            writeRaw(prefixes.SIGIL_8_BIT);
            writeRawRange(encodeBigEndian(static_cast<uint8_t>(value)));
        }
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint16_t>::max())
        {
            writeRaw(prefixes.SIGIL_16_BIT);
            writeRawRange(encodeBigEndian(static_cast<uint16_t>(value)));
        }
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint32_t>::max())
        {
            writeRaw(prefixes.SIGIL_32_BIT);
            writeRawRange(encodeBigEndian(static_cast<uint32_t>(value)));
        }
        else
        {
            writeRaw(prefixes.SIGIL_64_BIT);
            writeRawRange(encodeBigEndian(static_cast<uint64_t>(value)));
        }
    }

private:
    /// The destination buffer to output encoded bytes into.
    BufferType m_buffer;

    /// The next position in the buffer to write.
    SizeType m_offset;
};

} // namespace dw::cbor
} // namespace dw

#endif /*DWCBOR_DW_CBOR_RAWWRITER_HPP_*/
