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

#ifndef DWCBOR_DW_CBOR_RAWREADER_HPP_
#define DWCBOR_DW_CBOR_RAWREADER_HPP_

#include <dw/core/container/Span.hpp>
#include <dw/core/language/Optional.hpp>

#include "Token.hpp"

namespace dw
{
namespace cbor
{

/// The raw interface used to read from a CBOR-encoded byte span. Like the @c RawWriter counterpart, this does not do
/// anything beyond the most basic input validation. A byte string which opens 10 arrays and then ends abruptly will not
/// trigger an error. It is the responsibility of higher-level constructs to validate this.
///
/// @code
/// cbor::RawReader reader{encodedSource};
/// while (reader.next())
/// {
///     cbor::Token const& token = reader.current();
///     doSomething(token);
/// }
/// @endcode
class RawReader
{
public:
    /// The buffer type to read from -- a byte string.
    using BufferType = core::span1ub_const;

    using SizeType = BufferType::size_type;

public:
    /// Create an instance to read from the given @a source.
    explicit constexpr RawReader(BufferType const source) noexcept
        : m_source{source}
        , m_nextIndex{0UL}
        , m_currentToken{}
        , m_currentTokenPosition{0UL}
    {
    }

    // Copying and moving are deleted, as it is likely always an error to do so.
    RawReader(RawReader const&) = delete;
    RawReader(RawReader&&)      = delete;

    RawReader& operator=(RawReader const&) = delete;
    RawReader& operator=(RawReader&&) = delete;

    ~RawReader() noexcept = default;

    /// Move to the next token.
    ///
    /// @returns @c true if this successfully moved to the next token; @c false if the end of the input was reached.
    bool next();

    /// Get the current token this reader is pointing at. If @ref next returned @c false or if it has never been called,
    /// the returned token will not be valid.
    constexpr Token const& current() const noexcept
    {
        return m_currentToken;
    }

    /// Get the position of the @ref current token. If @ref next has not been called, this will return 0 (or the last
    /// position passed to the @ref setPosition function); if @ref next returned @c false, this will be the size of the
    /// source (EOF-equivalent).
    constexpr SizeType getPosition() const noexcept
    {
        return m_currentTokenPosition;
    }

    /// Set the position of this reader to @a newPosition.
    ///
    /// @param newPosition The position to set this reader to, which must be the start of a token. This is not checked,
    ///                    as there is no mechanism to perform the check. Setting the position to in the middle of a
    ///                    token will elicit strange behavior.
    void setPosition(SizeType const newPosition) noexcept
    {
        m_currentTokenPosition = newPosition;
        m_nextIndex            = newPosition;
        m_currentToken         = Token{};
    }

private:
    BufferType m_source;
    SizeType m_nextIndex;
    Token m_currentToken;
    SizeType m_currentTokenPosition;
};

} // namespace dw::cbor
} // namespace dw

#endif /*DWCBOR_DW_CBOR_RAWREADER_HPP_*/
