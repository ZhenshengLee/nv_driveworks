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
// Copyright (c) 2020-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_UTIL_CHECKNOTNULLARG_HPP_
#define DW_CORE_UTIL_CHECKNOTNULLARG_HPP_

#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>

namespace dw
{
namespace core
{
namespace detail
{

/// @see checkNotNullArg
[[noreturn]] void throwCheckNotNullArgIsNull(char8_t const* const context, char8_t const* name);

} // namespace dw::core::detail

/**
 * @brief Check that arg is not nullptr and throw an InvalidArgumentException if it is.
 *  While this can be used anywhere, this is primarily meant for use in constructors where a pointer argument is used in member initializers.
 * sample of usage:
 * MyType::MyType(Something* foo)
 *     : m_foo{checkNotNullArg(foo, "MyType::MyType(Something*)", "foo")}
 *     , m_bar{m_foo->doAccess()}
 * { }
 *
 * Since m_foo->doAccess() requires a non-null m_foo, checkNotNullArg is used to verify this requirement as an
 * expression. Doing this check in the constructor function body would lead to undefined behavior, as the check would
 * occur too late.
 *
 * @tparam T the type of the @a arg
 * @param arg The argument to be checked
 * @param context A context message used to create the exception message. This should be the name of the calling function.
 * @param name The name of the argument to use in the exception message.
 * @return T* the same argument if it is not null
 */
template <typename T>
auto checkNotNullArg(T* const arg, char8_t const* const context, char8_t const* const name) -> T*
{
    if (arg == nullptr)
    {
        detail::throwCheckNotNullArgIsNull(context, name);
    }

    return arg;
}

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_UTIL_CHECKNOTNULLARG_HPP_*/
