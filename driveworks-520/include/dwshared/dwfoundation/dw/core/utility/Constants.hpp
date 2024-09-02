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
// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_UTILITY_CONSTANTS_HPP_
#define DW_CORE_UTILITY_CONSTANTS_HPP_

#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>

#include <cstdint>

namespace dw
{
namespace core
{
namespace util
{

/**
 * \defgroup constants_group Symbolic names for AUTOSAR A5-1-1 compliance
 * @{
 */

constexpr float32_t HALF_F{0.5F}; ///< generic floating point    variable with value 0.5, for AUTOSAR A5-1-1 compliance

constexpr int8_t ZERO{0};        ///< generic signed    integer variable with value 0, for AUTOSAR A5-1-1 compliance
constexpr uint8_t ZERO_U{0U};    ///< generic unsigned  integer variable with value 0, for AUTOSAR A5-1-1 compliance
constexpr size_t ZERO_UL{0UL};   ///< generic unsigned long integer variable with value 0, for AUTOSAR A5-1-1 compliance
constexpr float32_t ZERO_F{0.F}; ///< generic floating point    variable with value 0, for AUTOSAR A5-1-1 compliance

constexpr int8_t ONE{1};        ///< generic signed    integer variable with value 1, for AUTOSAR A5-1-1 compliance
constexpr uint8_t ONE_U{1U};    ///< generic unsigned  integer variable with value 1, for AUTOSAR A5-1-1 compliance
constexpr size_t ONE_UL{1UL};   ///< generic unsigned long integer variable with value 1, for AUTOSAR A5-1-1 compliance
constexpr float32_t ONE_F{1.F}; ///< generic floating point    variable with value 1, for AUTOSAR A5-1-1 compliance

constexpr int8_t TWO{2};        ///< generic signed    integer variable with value 2, for AUTOSAR A5-1-1 compliance
constexpr uint8_t TWO_U{2U};    ///< generic unsigned  integer variable with value 2, for AUTOSAR A5-1-1 compliance
constexpr size_t TWO_UL{2UL};   ///< generic unsigned long integer variable with value 2, for AUTOSAR A5-1-1 compliance
constexpr float32_t TWO_F{2.F}; ///< generic floating point    variable with value 2, for AUTOSAR A5-1-1 compliance

constexpr int8_t THREE{3};        ///< generic signed    integer variable with value 3, for AUTOSAR A5-1-1 compliance
constexpr uint8_t THREE_U{3U};    ///< generic unsigned  integer variable with value 3, for AUTOSAR A5-1-1 compliance
constexpr size_t THREE_UL{3UL};   ///< generic unsigned long integer variable with value 3, for AUTOSAR A5-1-1 compliance
constexpr float32_t THREE_F{3.F}; ///< generic floating point    variable with value 3, for AUTOSAR A5-1-1 compliance

constexpr int8_t FOUR{4};        ///< generic signed    integer variable with value 4, for AUTOSAR A5-1-1 compliance
constexpr uint8_t FOUR_U{4U};    ///< generic unsigned  integer variable with value 4, for AUTOSAR A5-1-1 compliance
constexpr size_t FOUR_UL{4UL};   ///< generic unsigned long integer variable with value 4, for AUTOSAR A5-1-1 compliance
constexpr float32_t FOUR_F{4.F}; ///< generic floating point    variable with value 4, for AUTOSAR A5-1-1 compliance

} // namespace util
} // namespace core
} // namespace dw

#endif
