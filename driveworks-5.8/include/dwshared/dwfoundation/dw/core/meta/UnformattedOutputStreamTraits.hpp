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
// SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_DW_CORE_META_UNFORMATTEDOUTPUTSTREAMTRAITS_HPP_
#define DWSHARED_DW_CORE_META_UNFORMATTEDOUTPUTSTREAMTRAITS_HPP_

/// @file
/// This file contains traits types for detection of @c std::ostream like types. The concept of an unformatted output
/// stream is pulled from the "Unformatted output functions" section of the C++20 Standard 29.7.5.3. This includes
/// three member functions (for some @c BaseOutputStream and @c TChar types):
///
/// * <tt>BaseOutputStream&amp; put(T c)</tt>
/// * <tt>BaseOutputStream&amp; write(T const* s, streamsize n)</tt>
/// * <tt>BaseOutputStream&amp; flush()</tt>
///
/// An unformatted output stream @c TOutputStream is one which defines all three of these functions with an identical
/// return type, which is an lvalue reference, where @c BaseOutputStream is either the same as or a public base of
/// @c TOutputStream. The @c BaseOutputStream is said to be the @e fundamental type of @c TOutputStream.
///
/// The most useful type alias in this file is @c UnformattedOutputStreamFundamentalType, which is used for making
/// @c operator<< implementations. See the example in the documentation.

#include <dw/core/language/cxx17.hpp>
#include <dw/core/language/TypeAliases.hpp>

#include <ios>
#include <type_traits>
#include <utility>

#include "Conjunction.hpp"

namespace dw
{
namespace core
{
namespace meta
{
namespace detail
{

template <typename TOutputStream, typename TChar>
using UnformattedOutputStreamImplPutResultT = decltype(std::declval<TOutputStream&>().put(std::declval<TChar>()));

template <typename TOutputStream, typename TChar>
using UnformattedOutputStreamImplWriteResultT = decltype(std::declval<TOutputStream&>().write(std::declval<TChar const*>(), std::declval<std::streamsize>()));

template <typename TOutputStream>
using UnformattedOutputStreamImplFlushResultT = decltype(std::declval<TOutputStream&>().flush());

template <typename TOutputStream, typename TChar, typename = void_t<>>
struct IsBasicUnformattedOutputStreamImpl : std::false_type
{
};

template <typename TOutputStream, typename TChar>
struct IsBasicUnformattedOutputStreamImpl<
    TOutputStream,
    TChar,
    void_t<
        // validate that all of these functions exist or this will not be valid in SFINAE
        UnformattedOutputStreamImplPutResultT<TOutputStream, TChar>,
        UnformattedOutputStreamImplWriteResultT<TOutputStream, TChar>,
        UnformattedOutputStreamImplFlushResultT<TOutputStream>>>
    : Conjunction<
          std::is_same<UnformattedOutputStreamImplFlushResultT<TOutputStream>, UnformattedOutputStreamImplPutResultT<TOutputStream, TChar>>,
          std::is_same<UnformattedOutputStreamImplFlushResultT<TOutputStream>, UnformattedOutputStreamImplWriteResultT<TOutputStream, TChar>>,
          std::is_lvalue_reference<UnformattedOutputStreamImplFlushResultT<TOutputStream>>,
          std::is_base_of<std::remove_reference_t<UnformattedOutputStreamImplFlushResultT<TOutputStream>>, TOutputStream>,
          std::is_convertible<TOutputStream&, UnformattedOutputStreamImplFlushResultT<TOutputStream>>>
{
};

template <typename TOutputStream, typename TChar, bool = IsBasicUnformattedOutputStreamImpl<TOutputStream, TChar>::value>
struct BasicUnformattedOutputStreamTraitsImpl
{
    // implementation empty -- not an output stream
};

template <typename TOutputStream, typename TChar>
struct BasicUnformattedOutputStreamTraitsImpl<TOutputStream, TChar, true>
{
    /// The fundamental type of the stream and what should be returned from @c operator<< implementations. This is
    /// similar to but not necessarily the same as @c TOutputStream. For example, @c std::ostringstream,
    /// @c std::stringstream, and @c std::ofstream all have a common fundamental stream type for @c char values:
    /// @c std::ostream.
    using FundamentalType = std::remove_reference_t<UnformattedOutputStreamImplWriteResultT<TOutputStream, TChar>>;
};

} // namespace dw::core::meta::detail

/// Check if @c TOutputStream is an unformatted output stream for @c TChar values.
///
/// @see UnformattedOutputStreamTraits
/// @see BasicUnformattedOutputStreamTraits
/// @see IsUnformattedOutputStream
/// @see IS_BASIC_UNFORMATTED_OUTPUT_STREAM_V
template <typename TOutputStream, typename TChar>
struct IsBasicUnformattedOutputStream : detail::IsBasicUnformattedOutputStreamImpl<TOutputStream, TChar>
{
};

/// @see IsBasicUnformattedOutputStream
template <typename TOutputStream, typename TChar>
constexpr bool IS_BASIC_UNFORMATTED_OUTPUT_STREAM_V = IsBasicUnformattedOutputStream<TOutputStream, TChar>::value;

/// Check if @c TOutputStream is an unformatted output stream for @c char8_t values.
///
/// @see IsBasicUnformattedOutputStream
/// @see IS_UNFORMATTED_OUTPUT_STREAM_V
template <typename TOutputStream>
using IsUnformattedOutputStream = IsBasicUnformattedOutputStream<TOutputStream, char8_t>;

/// @see IsUnformattedOutputStream
template <typename TOutputStream>
constexpr bool IS_UNFORMATTED_OUTPUT_STREAM_V = IsUnformattedOutputStream<TOutputStream>::value;

/// If @c TOutputStream is an output stream for @c TChar values (@c IS_BASIC_UNFORMATTED_OUTPUT_STREAM_V is @c true),
/// this will have a member type @c FundamentalType. If this is not an output stream, the member will not exist.
///
/// @see UnformattedOutputStreamTraits
/// @see BasicUnformattedOutputStreamFundamentalType
template <typename TOutputStream, typename TChar>
struct BasicUnformattedOutputStreamTraits : detail::BasicUnformattedOutputStreamTraitsImpl<TOutputStream, TChar>
{
};

/// If @c TOutputStream is an output stream for @c char8_t values (@c IS_UNFORMATTED_OUTPUT_STREAM_V is @c true), this
/// will have a member type @c FundamentalType. If this is not an output stream, the member will not exist.
///
/// @see BasicUnformattedOutputStreamTraits
/// @see UnformattedOutputStreamFundamentalType
template <typename TOutputStream>
using UnformattedOutputStreamTraits = BasicUnformattedOutputStreamTraits<TOutputStream, char8_t>;

/// If @c TOutputStream is an unformatted output stream (@c IS_BASIC_UNFORMATTED_OUTPUT_STREAM_V is @c true), this is
/// the type returned from output stream functions. This is useful for making @c operator<< implementations which work
/// for multiple output-stream-like types.
template <typename TOutputStream, typename TChar>
using BasicUnformattedOutputStreamFundamentalType = typename BasicUnformattedOutputStreamTraits<TOutputStream, TChar>::FundamentalType;

/// If @c TOutputStream is an unformatted output stream (@c IS_UNFORMATTED_OUTPUT_STREAM_V is @c true), this is the type
/// returned from output stream functions. This is useful for making @c operator<< implementations which work for
/// multiple output-stream-like types.
///
/// For example, say you want to implement a outputter for @c MyType which works with both @c std::ostream and a
/// @c LoggerStream. You can use @c UnformattedOutputStreamTraits in the signature of the function (as well as a note to
/// Coverity) to make a single function that works with both:
///
/// @code
/// template <typename TOutputStream>
/// // TODO(dwplc): RFD -- allow references to be returned from << when left-hand side is ostream-like
/// // coverity[autosar_cpp14_a13_2_2_violation]
/// meta::UnformattedOutputStreamFundamentalType<TOutputStream>&
/// operator<<(TOutputStream& os, MyType const& value);
/// @endcode
template <typename TOutputStream>
using UnformattedOutputStreamFundamentalType = typename UnformattedOutputStreamTraits<TOutputStream>::FundamentalType;

} // namespace dw::core::meta
} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_DW_CORE_META_UNFORMATTEDOUTPUTSTREAMTRAITS_HPP_*/
