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
// SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_ENUMDESCRIPTOR_HPP_
#define DW_FRAMEWORK_ENUMDESCRIPTOR_HPP_

#include <dw/core/base/Status.h>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>

#include <dwcgf/Exception.hpp>

#include <array>
#include <utility>
#include <type_traits>

#define DW_ENUMERATOR_NAME_STRING_VIEW(NAME_STR) \
    dw::core::StringView { NAME_STR }
/**
 * @brief Syntactic sugar calling describeEnumerator().
 *
 * It avoid having to pass both the enumerator name and value explicitly.
 * This macro should only be used within dw::framework::EnumDescription::get()
 * where EnumT is defined.
 * The block scope must contain the user-defined string literal from
 * dw::core::operator""_sv
 */
#define DW_DESCRIBE_ENUMERATOR(NAME) describeEnumerator(DW_ENUMERATOR_NAME_STRING_VIEW(#NAME), EnumT::NAME)
/**
 * @brief Syntactic sugar calling describeEnumerator().
 *
 * It avoid having to pass both the enumerator name and value explicitly.
 * The block scope must contain the user-defined string literal from
 * dw::core::operator""_sv
 */
#define DW_DESCRIBE_C_ENUMERATOR(NAME) describeEnumerator(DW_ENUMERATOR_NAME_STRING_VIEW(#NAME), (NAME))

namespace dw
{
namespace framework
{

// API to declare the mapping of a concrete enum.

template <typename EnumT, size_t NumberOfEnumerators>
using EnumDescriptionReturnType = std::array<std::pair<dw::core::StringView, EnumT>, NumberOfEnumerators>;

template <typename EnumT>
struct EnumDescription
{
    static_assert(std::is_enum<EnumT>::value, "EnumT must be an enumeration type");

    /**
     * @brief Describe the enumerators.
     *
     * This method needs to be specialized for each EnumT by returning
     * describeEnumeratorCollection().
     *
     * @return The array of enumerators. The size of the array depends on the number of enumerators of EnumT.
     */
    // coverity[autosar_cpp14_a2_10_5_violation]
    static constexpr EnumDescriptionReturnType<EnumT, 0> get()
    {
        static_assert(sizeof(EnumT) == 0, "Needs specialization for specific enum type");
        constexpr const size_t numberOfEnumerators = 0;
        EnumDescriptionReturnType<EnumT, numberOfEnumerators> ret;
        return ret;
    }
};

// API needed to declare the enumerators of an enum.

/**
 * @brief Describe the enumerators.
 *
 * This function is used in specializations of EnumDescription::get() to
 * describe all enumerators.
 *
 * @param args Each argument describes an enumerator created by
 * describeEnumerator()
 * @return The array of enumerators
 */
template <typename EnumT, typename... Args>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeEnumeratorCollection(Args const&&... args) -> std::array<std::pair<dw::core::StringView, EnumT>, sizeof...(Args)>
{
    return std::array<std::pair<dw::core::StringView, EnumT>, sizeof...(Args)>{std::forward<const Args>(args)...};
}

/**
 * @brief Describe an enumerator.
 *
 * This function is used to create arguments for
 * describeEnumeratorCollection().
 * To avoid having to specify both explicitly - the enumerator name as well as
 * the value - the macro DW_DESCRIBE_ENUMERATOR() is used.
 *
 * @param[in] name The enumerator name
 * @param[in] value The enumerator value
 * @return The pair with the enumerator name and value
 */
template <typename EnumT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeEnumerator(dw::core::StringView&& name, EnumT value) -> std::pair<dw::core::StringView, EnumT>
{
    return std::make_pair(std::move(name), value);
}

/**
 * @brief Get the enumerator value based on the name.
 *
 * @param[in] name The enumerator name
 * @return The enumerator value
 * @throws Exception if the enumerator name is invalid
 */
template <typename EnumT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
auto mapEnumNameToValue(dw::core::StringView const& name) -> EnumT
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    const auto pairs = EnumDescription<EnumT>::get();
    for (const auto& pair : pairs)
    {
        if (pair.first == name)
        {
            return pair.second;
        }
    }
    // coverity[autosar_cpp14_a18_5_8_violation] FP: nvbugs/3498833
    FixedString<4096> validNames{};
    for (const auto& pair : pairs)
    {
        validNames << dw::core::StringView{" "} << pair.first;
    }
    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Invalid enumerator name '", name, "' for enum type ", typeid(EnumT).name(), ", valid names:", validNames);
}

/**
 * @brief Get the enumerator name based on the value.
 *
 * @param[in] value The enumerator value
 * @return The enumerator name
 * @throws Exception if the enumerator value is invalid
 */
template <typename EnumT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
dw::core::StringView mapEnumValueToName(EnumT value)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    const auto pairs = EnumDescription<EnumT>::get();
    for (const auto& pair : pairs)
    {
        if (pair.second == value)
        {
            return pair.first;
        }
    }
    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "The enumerator value is invalid: ", typeid(EnumT).name(), " ", static_cast<std::underlying_type_t<EnumT>>(value));
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ENUMDESCRIPTOR_HPP_
