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
// Copyright (c) 2015-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_SHA256_HPP_
#define DW_CORE_SHA256_HPP_

#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include <array>
#include <type_traits>

namespace dw
{
namespace core
{

///
/// \brief The SHA256 class Generates the SHA256 sum of a given string
/// Source code comes from https://www.programmingalgorithms.com/algorithm/sha256
/// You may read, print, download, copy, modify and distribute the source code for
/// your own personal, non-commercial and commercial use.
///
class SHA256
{
public:
    static constexpr size_t HASH_LENGTH{32UL};
    using Result       = std::array<uint8_t, HASH_LENGTH>;
    using Base64String = FixedString<2 * HASH_LENGTH + 1>;

    //////////////////////////////////
    // Instance members
    SHA256();

    /// Reset hasher to be fed again
    void reset();

    /// Hash a given array
    void hashArray(span<uint8_t const> const data);

    /// Hash any data. Call this multiple times before finalizing
    template <class T>
    std::enable_if_t<traits::is_span<T>::value == false> hash(T const& data)
    {
        auto dataBytes = reinterpret_cast<uint8_t const*>(&data); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        sha256Update(m_context, make_span(dataBytes, sizeof(T)));
    }

    /// Finalize computation of the hash, i.e. make solution available through getHash()
    void finalize();

    /// Return hashed result
    Result const& getHash() const
    {
        return m_hash;
    }

    /// Return base64 encoding of the hash
    Base64String createHashText() const
    {
        return createHashText(m_hash);
    }

    //////////////////////////////////
    // Static members
    static Base64String get(span<char8_t const> const data);
    static Result getArray(span<uint8_t const> const data);

    // compute base64 encoding of a binary hash value
    static Base64String createHashText(Result const& hash);

private:
    //////////////////////////////////
    // Static members
    struct SHA256_CTX // clang-tidy NOLINT(readability-identifier-naming)
    {
        uint8_t data[64];
        uint32_t datalen;
        uint32_t bitlen[2];
        uint32_t state[8];
    };

    static void sha256Transform(SHA256_CTX& ctx, span<uint8_t const> const data);
    static void sha256Init(SHA256_CTX& ctx);
    static void sha256Update(SHA256_CTX& ctx, span<uint8_t const> const data);
    static Result sha256Final(SHA256_CTX& ctx);

    static uint32_t const K[64];

    //////////////////////////////////
    // Instance members
    SHA256_CTX m_context;

    Result m_hash;
};

} // namespace core
} // namespace dw

#endif // DW_CORE_SHA256_HPP_
