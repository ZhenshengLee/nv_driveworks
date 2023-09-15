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

#ifndef DW_CORE_UTILITY_HASH_HPP_
#define DW_CORE_UTILITY_HASH_HPP_

#include <type_traits>
#include <functional>

namespace dw
{
namespace core
{

/**
 * \defgroup hash_group Hash Group of Functions
 * @{
 */

/// Combines an existing hash with the hash of a new element
/// @param [out] seed The seed to hash with another element and its value will be overwritten.
/// @param [in] v The element to hash with the given seed.
template <class T>
inline void hashCombine(size_t& seed, const T& v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // taken from boost
}

/// Returns std::hash of the input value
/// @param v The input value
template <class T>
inline size_t multiHash(const T& v)
{
    return std::hash<T>()(v);
}

/// Function to hash multiple elements together.
/// std::hash will be used for each individual element.
/// @param v The first element to hash
/// @param rest The other elements to hash
template <class T, typename... Rest>
inline size_t multiHash(const T& v, Rest... rest)
{
    std::size_t seed = multiHash(rest...);
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // taken from boost
    return seed;
}

/**@}*/

/// Hash function for enums
template <typename TEnum>
struct EnumHash
{
    size_t operator()(const TEnum& e) const
    {
        using T = typename std::underlying_type<TEnum>::type;
        return std::hash<T>()(static_cast<T>(e));
    }
};

} // core
} // dw

// specialization of hash function of std::pair
namespace std
{

template <typename S, typename T>
struct hash<pair<S, T>>
{
    inline size_t operator()(const pair<S, T>& v) const
    {
        return dw::core::multiHash(v.first, v.second);
    }
};
}

#endif
