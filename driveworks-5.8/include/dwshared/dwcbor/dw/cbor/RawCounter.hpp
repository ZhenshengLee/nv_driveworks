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

#ifndef DWCBOR_DW_CBOR_RAWCOUNTER_HPP_
#define DWCBOR_DW_CBOR_RAWCOUNTER_HPP_

#include <dw/core/container/Span.hpp>
#include <dw/core/container/StringView.hpp>
#include <dw/core/language/TypeAliases.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Codes.hpp"
#include "Forwards.hpp"

namespace dw
{
namespace cbor
{

/// Counts the number of characters it would take to encode pieces of data. It will never underestimate these counts,
/// but it can overestimate them (for example, the counter does not check floating-point numbers for NaN, which is
/// encoded as 3 bytes instead of 5 or 9 bytes). This is useful for sizing buffers to pass into @c RawWriter.
///
/// @code
/// template <typename TCbor>
/// void encodeImpl(TCbor& cbor, MyType const& x)
/// {
///     // Write encoding here
///     x.writeInt(x.thing);
/// }
///
/// std::size_t encodedSize(MyType const& x)
/// {
///     cbor::RawCounter ctr;
///     encodeImpl(ctr, x);
///     return ctr.encodedSize();
/// }
///
/// span<uint8_t const> encode(MyType const& x, span<uint8_t> buffer)
/// {
///     cbor::RawWriter wtr(buffer);
///     encodeImpl(wtr, x);
///     return wtr.encoded();
/// }
/// @endcode
///
/// @see RawWriter
class RawCounter
{
public:
    /// The size type to use. This will always be a 64-bit unsigned integer.
    using SizeType = std::uint64_t;

public:
    explicit constexpr RawCounter() noexcept = default;

    // Copying and moving are deleted, as it is almost always an error to do so.
    RawCounter(RawCounter const&) = delete;
    RawCounter(RawCounter&&)      = delete;

    RawCounter& operator=(RawCounter const&) = delete;
    RawCounter& operator=(RawCounter&&) = delete;

    ~RawCounter() = default;

    /// Get the number of bytes it would take to encode all the specified items.
    constexpr SizeType encodedSize() const noexcept
    {
        return m_counted;
    }

    /// Count the bytes for the null-equivalent value.
    constexpr void writeNull() noexcept
    {
        add(sizeof(Codes::NULL_SIGIL));
    }

    /// Count the bytes for a boolean.
    constexpr void writeBool(bool) noexcept
    {
        static_assert(sizeof(Codes::TRUE_SIGIL) == sizeof(Codes::FALSE_SIGIL),
                      "True and false sigils must be same size");
        add(sizeof(Codes::TRUE_SIGIL));
    }

    /// Count the bytes for an unsigned integer.
    template <typename T>
    constexpr std::enable_if_t<std::is_unsigned<T>::value> writeInt(T const& value) noexcept
    {
        writeCompactUInt(Codes::PositiveIntPrefixes{}, value);
    }

    /// Count the bytes for a signed integer.
    template <typename TSInt>
    constexpr std::enable_if_t<std::is_signed<TSInt>::value> writeInt(TSInt const& value) noexcept
    {
        using UnsignedType = std::make_unsigned_t<TSInt>;

        if (value >= TSInt{0})
        {
            writeInt(static_cast<UnsignedType>(value));
        }
        else
        {
            UnsignedType const coded = static_cast<UnsignedType>(-1 - value);
            writeCompactUInt(Codes::NegativeIntPrefixes{}, coded);
        }
    }

    /// Count the bytes of a floating-point value.
    ///
    /// @note
    /// For performance reasons, unlike @c RawWriter::writeFloat, this function does not check for infinite and NaN
    /// values. Since infinity and NaN sigils are less than the size of the corresponding finite value encoding, this
    /// can lead to overestimation of required size.
    template <typename T>
    constexpr std::enable_if_t<std::is_floating_point<T>::value> writeFloat(T const&) noexcept
    {
        static_assert(sizeof(Codes::FLOAT32_SIGIL) == sizeof(Codes::FLOAT64_SIGIL),
                      "Float prefixes must be the same size");
        add(sizeof(Codes::FLOAT32_SIGIL) + sizeof(T));
    }

    /// Count the bytes to encode a string with the given @a size. This is useful for calculating the size without
    /// creating the actual string.
    constexpr void writeStringSize(SizeType const size) noexcept
    {
        writeCompactUInt(Codes::StringPrefixes{}, size);
        add(size);
    }

    /// Count the bytes to encode the string @a value.
    constexpr void writeString(core::StringView const& value) noexcept
    {
        return writeStringSize(value.size());
    }

    /// Count the bytes to encode a blob with the given @a size. This is useful for calculating the size without
    /// creating the actual encoded data.
    constexpr void writeBlobSize(SizeType const size) noexcept
    {
        writeCompactUInt(Codes::BlobPrefixes{}, size);
        add(size);
    }

    /// Count the bytes to encode the @a data blob.
    constexpr void writeBlob(span<std::uint8_t const> const data) noexcept
    {
        writeBlobSize(data.size());
    }

    /// Count the bytes to encode a semantic @a tag.
    template <typename T>
    constexpr std::enable_if_t<std::is_unsigned<T>::value> writeSemantic(T const tag) noexcept
    {
        writeCompactUInt(Codes::SemanticTagPrefixes{}, tag);
    }

    /// Add the number of bytes it would take to encode the introducer for an array of @a count items.
    constexpr void beginArray(SizeType const count) noexcept
    {
        writeCompactUInt(Codes::ArraySizePrefixes{}, count);
    }

    /// Add the number of bytes is would take to encode the introducer for a map of @a count key-value pairs.
    constexpr void beginMap(SizeType const count) noexcept
    {
        writeCompactUInt(Codes::MapSizePrefixes{}, count);
    }

private:
    /// Count the compact form of the integer @a value.
    template <typename TPrefixes, typename T>
    constexpr void writeCompactUInt(TPrefixes const prefixes, T const& value) noexcept // clang-tidy NOLINT(clang-diagnostic-undefined-inline)
    {
        static_assert(std::is_unsigned<T>::value, "T must be unsigned integer");
        static_assert(sizeof(T) <= sizeof(uint64_t), "64-bit integer is the maximum supported size");

        if (value <= Codes::DIRECT_MAX)
        {
            add(sizeof(std::uint8_t));
        }
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint8_t>::max())
        {
            add(sizeof(prefixes.SIGIL_8_BIT));
            add(sizeof(uint8_t));
        }
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint16_t>::max())
        {
            add(sizeof(prefixes.SIGIL_16_BIT));
            add(sizeof(uint16_t));
        }
        // coverity[result_independent_of_operands]
        else if (value <= std::numeric_limits<uint32_t>::max())
        {
            add(sizeof(prefixes.SIGIL_32_BIT));
            add(sizeof(uint32_t));
        }
        else
        {
            add(sizeof(prefixes.SIGIL_64_BIT));
            add(sizeof(uint64_t));
        }
    }

    /// Add @a sz bytes to the output.
    constexpr void add(SizeType const sz) noexcept
    {
        SizeType const countedAfter = m_counted + sz;
        if (countedAfter < m_counted)
        {
            m_counted = std::numeric_limits<SizeType>::max();
        }
        else
        {
            m_counted = countedAfter;
        }
    }

private:
    /// The number of bytes that have been counted.
    SizeType m_counted{0U};
};
}
}

#endif /*DWCBOR_DW_CBOR_RAWCOUNTER_HPP_*/
