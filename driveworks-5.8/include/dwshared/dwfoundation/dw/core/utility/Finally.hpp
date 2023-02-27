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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_FINALLY_HPP_
#define DW_CORE_UTILITY_FINALLY_HPP_

#include <type_traits>
#include <functional>

namespace dw
{
namespace core
{
namespace util
{

/*
 * Finally guard (execute functor at end of guard scope)
 *
 * Usage:
 * {
 *   auto resource = "acquire";
 *   auto guard = finally([&resource] { "cleanup" });
 *   ..
 *   <use resource>
 *   ..
 *
 *   <resource will be cleaned up, even if scope was left early> (e.g., due to exception)
 * }
**/
namespace detail
{

template <typename Functor>
struct FinallyGuard
{
    /**
     * @brief Construct a new Finally Guard object from an object
     *
     * @param f the input function pointer
     */
    explicit FinallyGuard(Functor f)
        : functor(std::move(f)), active(true) {}
    /**
     * @brief Construct a new Finally Guard object from a reference
     *
     * @param other the input FinallyGuard object
     */
    FinallyGuard(FinallyGuard&& other)
        : functor(std::move(other.functor)), active(other.active)
    {
        other.active = false;
    }

    /**
     * @brief The assign operator of FinallyGuard
     *
     * @return FinallyGuard&
     */
    FinallyGuard& operator=(FinallyGuard&&) = delete;

    /**
     * @brief Destroy the Finally Guard object
     *
     */
    ~FinallyGuard()
    {
        if (active)
        {
            functor();
        }
    }

    Functor functor;
    bool active;
};

} // namespace detail

/**
 * @brief Warpper of the detail::FinallyGuard function
 *
 * @tparam F the type of the function
 * @param f function pointer to be moved to the FinallyGuard
 * @return detail::FinallyGuard<typename std::decay<F>::type>
 */
template <typename F>
auto finally(F&& f) -> detail::FinallyGuard<typename std::decay<F>::type>
{
    return detail::FinallyGuard<typename std::decay<F>::type>(std::forward<F>(f));
}

} // end namespace util
} // end namespace core
} // end namespace dw

#endif