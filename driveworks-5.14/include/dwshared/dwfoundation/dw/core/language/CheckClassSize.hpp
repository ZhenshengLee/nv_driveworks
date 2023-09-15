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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_CHECKCLASSSIZE_HPP_
#define DW_CORE_UTILITY_CHECKCLASSSIZE_HPP_

#include <cstddef> // std::size_t

namespace dw
{
namespace core
{

/**
 * \defgroup sizecheck_group CheckClassSize Group of Functions
 * @{
 */

/// Implements a static assert for classes to check if its data size is equal to a specified size. For example:
///    void MyClass::serialize(serialization::AnyArchive &archive)<br>
///    {<br>
///        dw::core::checkClassSize<decltype(*this), 104>();<br>
///        ...<br>
///    }<br>
template <typename T, std::size_t EXPECTEDSIZE, std::size_t ACTUALSIZE = sizeof(T)>
inline constexpr bool checkClassSize()
{
    static_assert(EXPECTEDSIZE == ACTUALSIZE, "The expected size of this class has changed.  This might"
                                              " require a change in other code.  Certain events might cause a false positive on"
                                              " this check such as adding a vptr or compiler changes.  Once you updated the code or have"
                                              " verified that it does not need to be changed, update the EXPECTEDSIZE template parameter with"
                                              " the value of ACTUALSIZE");
    return true;
};

/// Implements a static assert for classes to check if its data size is less than a specified size. For example:
///    void MyClass::serialize(serialization::AnyArchive &archive)<br>
///    {<br>
///        dw::core::checkClassSize<decltype(*this), 104>();<br>
///        ...<br>
///    }<br>
template <typename T, std::size_t EXPECTEDSIZE, std::size_t ACTUALSIZE = sizeof(T)>
inline constexpr bool checkClassSizeLessThan()
{
    static_assert(EXPECTEDSIZE >= ACTUALSIZE, "The size of this class is too big");

    return true;
};

/**@}*/

} // namespace core
} // namespace dw

#endif // DW_CORE_UTILITY_CHECKCLASSSIZE_HPP_
