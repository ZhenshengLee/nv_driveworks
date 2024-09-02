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
// Copyright (c) 2020-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_UTILITY_ENDIAN_HPP_
#define DW_CORE_UTILITY_ENDIAN_HPP_

#include <dwshared/dwfoundation/dw/core/language/Tag.hpp>

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/container/Array.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>

#include <cstdint>
#include <type_traits>

namespace dw
{
namespace core
{

/**
 * \defgroup endian_group Endian Conversion Group of Functions
 * @{
 */

/// Convert @a src to a little-endian encoded byte string.
/// @param src The source unsigned int64 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 8U> encodeLittleEndian(std::uint64_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 8U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 16U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 24U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 32U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 40U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 48U),
        static_cast<std::uint8_t>(src >> 56U),
    };
}

/// Convert @a src to a little-endian encoded byte string.
/// @param src The source unsigned int32 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 4U> encodeLittleEndian(std::uint32_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 8U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 16U),
        static_cast<std::uint8_t>(src >> 24U),
    };
}

/// Convert @a src to a little-endian encoded byte string.
/// @param src The source unsigned int16 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 2U> encodeLittleEndian(std::uint16_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
        static_cast<std::uint8_t>(src >> 8U),
    };
}

/// Convert @a src to a little-endian encoded byte string. A single byte does not have any endianness associated with
/// it, but this overload exists for API completeness.
/// @param src The source unsigned int8 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 1U> encodeLittleEndian(std::uint8_t src) noexcept
{
    return {src};
}

/// Convert @a src to a little-endian encoded byte string. This overload is only enabled if @c TInteger is a signed
/// integer type. The resulting array is the same size as the unsigned version of @c TInteger.
/// @param src The source TInteger to convert
template <typename TInteger>
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_signed<TInteger>::value, decltype(encodeLittleEndian(std::make_unsigned_t<TInteger>()))>
encodeLittleEndian(TInteger src) noexcept
{
    // Casting negative integers into unsigned is an issue in general, but it should
    // be OK in this case, as the encoding/decoding tests prove.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return encodeLittleEndian(static_cast<std::make_unsigned_t<TInteger>>(src));
}

/// Decode the little-endian @a src byte string as an @c uint64_t.
/// @param src The source byte string to decode from. It must be at least 8 bytes long.
CUDA_BOTH_INLINE std::uint64_t decodeLittleEndian(Tag<std::uint64_t>, span1ub_const src)
{
    // clang-format off
    return (static_cast<std::uint64_t>(src[0U]) <<  0U)
         | (static_cast<std::uint64_t>(src[1U]) <<  8U)
         | (static_cast<std::uint64_t>(src[2U]) << 16U)
         | (static_cast<std::uint64_t>(src[3U]) << 24U)
         | (static_cast<std::uint64_t>(src[4U]) << 32U)
         | (static_cast<std::uint64_t>(src[5U]) << 40U)
         | (static_cast<std::uint64_t>(src[6U]) << 48U)
         | (static_cast<std::uint64_t>(src[7U]) << 56U)
         ;
    // clang-format on
}

/// Decode the little-endian @a src byte string as an @c uint32_t.
///
/// @param src The source byte string to decode from. It must be at least 4 bytes long.
CUDA_BOTH_INLINE std::uint32_t decodeLittleEndian(Tag<std::uint32_t>, span1ub_const src)
{
    // clang-format off
    return (static_cast<std::uint32_t>(src[0U]) <<  0U)
         | (static_cast<std::uint32_t>(src[1U]) <<  8U)
         | (static_cast<std::uint32_t>(src[2U]) << 16U)
         | (static_cast<std::uint32_t>(src[3U]) << 24U)
         ;
    // clang-format on
}

/// Decode the little-endian @a src byte string as an @c uint16_t.
///
/// @param src The source byte string to decode from. It must be at least 2 bytes long.
CUDA_BOTH_INLINE std::uint16_t decodeLittleEndian(Tag<std::uint16_t>, span1ub_const src)
{
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[0U]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[1U]) << 8U));
}

/// Decode the little-endian @a src byte string as an @c uint8_t. A single byte does not have endianness associated with
/// it, but this overload exists for API completeness.
///
/// @param src The source byte string to decode from. It must be at least 1 byte long.
CUDA_BOTH_INLINE std::uint8_t decodeLittleEndian(Tag<std::uint8_t>, span1ub_const src)
{
    return src[0U];
}

/// Decode the little-endian @a src byte string as @c TInteger. This overload is only enabled if @c TInteger is a signed
/// integer type.
///
/// @param src The source byte string to decode from. It has the same length requirements as the unsigned version of
///            @c TInteger, which is @c sizeof(TInteger).
template <typename TInteger>
CUDA_BOTH_INLINE
    std::enable_if_t<std::is_signed<TInteger>::value, TInteger>
    decodeLittleEndian(Tag<TInteger>, span1ub_const src)
{
    using UnsignedInt = std::make_unsigned_t<TInteger>;

    // Casting a value from [SIGNED_MAX, UNSIGNED_MAX] results in implementation dependant behavior, which
    // should not be an issue in this case as the project is focusing on a specific platform
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return static_cast<TInteger>(decodeLittleEndian(TAG<UnsignedInt>, std::move(src)));
}

/// Convert @a src to a big-endian encoded byte string.
/// @param src The source unsigned int64 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 8U> encodeBigEndian(std::uint64_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        static_cast<std::uint8_t>(src >> 56U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 48U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 40U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 32U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 24U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 16U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 8U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
    };
}

/// Convert @a src to a big-endian encoded byte string.
/// @param src The source unsigned int32 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 4U> encodeBigEndian(std::uint32_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        static_cast<std::uint8_t>(src >> 24U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 16U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 8U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
    };
}

/// Convert @a src to a big-endian encoded byte string.
/// @param src The source unsigned int16 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 2U> encodeBigEndian(std::uint16_t src) noexcept
{
    // The operations below seemingly lead to data loss, but these are necessary
    // operations to actually implement the encoding. So the violations of AUTOSAR
    // A4-7-1 and CERT INT31 C are suppressed.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return {
        static_cast<std::uint8_t>(src >> 8U),
        // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
        static_cast<std::uint8_t>(src >> 0U),
    };
}

/// Convert @a src to a big-endian encoded byte string. A single byte does not have any endianness associated with it,
/// but this overload exists for API completeness.
/// @param src The source unsigned int8 to convert
CUDA_BOTH_INLINE constexpr Array<std::uint8_t, 1U> encodeBigEndian(std::uint8_t src) noexcept
{
    return {src};
}

/// Convert @a src to a big-endian encoded byte string. This overload is only enabled if @c TInteger is a signed
/// integer type. The resulting array is the same size as the unsigned version of @c TInteger.
/// @param src The source unsigned TInteger to convert
template <typename TInteger>
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_signed<TInteger>::value, decltype(encodeBigEndian(std::make_unsigned_t<TInteger>()))>
encodeBigEndian(TInteger src) noexcept
{
    // Casting negative integers into unsigned is an issue in general, but it should
    // be OK in this case, as the encoding/decoding tests prove.
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return encodeBigEndian(static_cast<std::make_unsigned_t<TInteger>>(src));
}

/// Decode the big-endian @a src byte string as an @c uint64_t.
/// @param src The source byte string to decode from. It must be at least 8 bytes long.
CUDA_BOTH_INLINE std::uint64_t decodeBigEndian(Tag<std::uint64_t>, span1ub_const src)
{
    // clang-format off
    return (static_cast<std::uint64_t>(src[0U]) << 56U)
         | (static_cast<std::uint64_t>(src[1U]) << 48U)
         | (static_cast<std::uint64_t>(src[2U]) << 40U)
         | (static_cast<std::uint64_t>(src[3U]) << 32U)
         | (static_cast<std::uint64_t>(src[4U]) << 24U)
         | (static_cast<std::uint64_t>(src[5U]) << 16U)
         | (static_cast<std::uint64_t>(src[6U]) <<  8U)
         | (static_cast<std::uint64_t>(src[7U]) <<  0U)
         ;
    // clang-format on
}

/// Decode the big-endian @a src byte string as an @c uint32_t.
///
/// @param src The source byte string to decode from. It must be at least 4 bytes long.
CUDA_BOTH_INLINE std::uint32_t decodeBigEndian(Tag<std::uint32_t>, span1ub_const src)
{
    // clang-format off
    return (static_cast<std::uint32_t>(src[0U]) << 24U)
         | (static_cast<std::uint32_t>(src[1U]) << 16U)
         | (static_cast<std::uint32_t>(src[2U]) <<  8U)
         | (static_cast<std::uint32_t>(src[3U]) <<  0U)
         ;
    // clang-format on
}

/// Decode the big-endian @a src byte string as an @c uint16_t.
///
/// @param src The source byte string to decode from. It must be at least 2 bytes long.
CUDA_BOTH_INLINE std::uint16_t decodeBigEndian(Tag<std::uint16_t>, span1ub_const src)
{
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[1U]) | static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[0U]) << 8U));
}

/// Decode the big-endian @a src byte string as an @c uint8_t. A single byte does not have endianness associated with
/// it, but this overload exists for API completeness.
///
/// @param src The source byte string to decode from. It must be at least 1 byte long.
CUDA_BOTH_INLINE std::uint8_t decodeBigEndian(Tag<std::uint8_t>, span1ub_const src)
{
    return src[0U];
}

/// Decode the big-endian @a src byte string as @c TInteger. This overload is only enabled if @c TInteger is a signed
/// integer type.
///
/// @param src The source byte string to decode from. It has the same length requirements as the unsigned version of
///            @c TInteger, which is @c sizeof(TInteger).
template <typename TInteger>
CUDA_BOTH_INLINE
    std::enable_if_t<std::is_signed<TInteger>::value, TInteger>
    decodeBigEndian(Tag<TInteger>, span1ub_const src)
{
    using UnsignedInt = std::make_unsigned_t<TInteger>;

    // Casting a value from [SIGNED_MAX, UNSIGNED_MAX] results in implementation dependant behavior, which
    // should not be an issue in this case as the project is focusing on a specific platform
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    return static_cast<TInteger>(decodeBigEndian(TAG<UnsignedInt>, std::move(src)));
}

/**@}*/

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_UTILITY_ENDIAN_HPP_*/
