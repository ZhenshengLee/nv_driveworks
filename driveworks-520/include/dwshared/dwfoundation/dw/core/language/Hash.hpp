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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Combines an existing hash with the hash of a new element.
 * @param [in,out] seed The seed to hash with another element and its value will be overwritten.
 * @param [in] v The element to hash with the given seed.
*/
template <class T>
inline void hashCombine(size_t& seed, const T& v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // taken from boost
}

/**
 * @brief Returns std::hash of the input value
 * @param[in] v The input value
 * @return Hash value
*/
template <class T>
inline size_t multiHash(const T& v)
{
    return std::hash<T>()(v);
}

/**
 * @brief Function to hash multiple elements together.
 *        std::hash will be used for each individual element.
 * @param[in] v The first element to hash
 * @param[in] rest The other elements to hash
 * @return Hash value
*/
template <class T, typename... Rest>
inline size_t multiHash(const T& v, Rest... rest)
{
    std::size_t seed{multiHash(rest...)};
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // taken from boost
    return seed;
}

/**
 * @brief Define a hash template for handling std::pair
*/
template <typename S, typename T>
struct PairHash
{
    size_t operator()(const std::pair<S, T>& v) const
    {
        return dw::core::multiHash(v.first, v.second);
    }
};

/**
 * @brief Specialization of hashCombine for std::pair
 * @param[in,out] seed The seed to hash with another element and its value will be overwritten.
 * @param[in] v The std::pair element to hash with the given seed.
*/
template <class S, class T>
inline void hashCombine(size_t& seed, const std::pair<S, T>& v)
{
    seed ^= PairHash<S, T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); // taken from boost
}

/**
 * @brief Specialization of multiHash for std::pair
 * @param[in] v  The input std::pair value
 * @return Hash value
*/
template <typename S, typename T>
inline size_t multiHash(const std::pair<S, T>& v)
{
    return PairHash<S, T>{}(v);
}

/**
 * @brief Hash function for enums
*/
template <typename TEnum>
struct EnumHash
{
    // Operator () overload to return hash value of the enum
    size_t operator()(const TEnum& e) const
    {
        using T = typename std::underlying_type<TEnum>::type;
        return std::hash<T>()(static_cast<T>(e));
    }
};

/**@}*/

} // core
} // dw

#endif
