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

#ifndef DW_CORE_LANGUAGE_TAG_HPP_
#define DW_CORE_LANGUAGE_TAG_HPP_

namespace dw
{
namespace core
{

/**
 * \defgroup tag_structs Tag Structures
 * @{
 */

/// A tag for disambiguating function overloads. This is useful for cases where the return type is the only distinct
/// part of the API.
///
// For examples, consider a simple API for parsing.
//
// @code
// std::uint64_t parse(Tag<std::uint64_t>, const std::string& src);
// std::uint32_t parse(Tag<std::uint32_t>, const std::string& src);
// std::uint16_t parse(Tag<std::uint16_t>, const std::string& src);
// std::uint8_t  parse(Tag<std::uint8_t>,  const std::string& src);
//
// template <typename T>
// std::enable_if_t<std::is_signed<T>::value, T> parse(Tag<T>, const std::string& src);
// @endcode
//
// The usage is:
//
// @code
// auto x = parse(TAG<std::uint64_t>, "327891");
// auto y = parse(TAG<std::uint16_t>, "192");
// @endcode
//
/// @note
/// The alternative to the type-tag strategy is function template specialization.
///
/// * You are only allowed full template specialization, so the @c std::is_signed overload can not be expressed.
/// * It interacts better with types in other namespaces through argument-dependent lookup. While full specializations
///   must live in the same namespace as the declaration, overloads can live anywhere. ADL will find a user-defined
///   @c parse function for a @c T through unqualified lookup if it is put into the same namespace as @c T.
/// * A fully-specialized template function is no longer a template, the consequences of which can surprise people. An
///   overloaded function follows all the regular behavior of overloaded functions.
///
template <typename T>
struct Tag
{
    using type = T;
};

/// Used as the value for a @ref Tag.
template <typename T>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
constexpr Tag<T> TAG{};

/**@}*/

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_LANGUAGE_TAG_HPP_*/
