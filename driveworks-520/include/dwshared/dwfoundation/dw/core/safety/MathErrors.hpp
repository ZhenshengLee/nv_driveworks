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

#ifndef DW_CORE_SAFETY_MATHERRORS_HPP_
#define DW_CORE_SAFETY_MATHERRORS_HPP_

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>

namespace dw
{
namespace core
{

// TODO(vsamokhvalov): this does not need a header, move to Safety.hpp
// Reset math error indicators before running a math command
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void resetMathErrors()
{ // clang-format off
#ifdef __CUDA_ARCH__
    // CUDA code: no math error reset
#else
    // CPU code: reset standard floating-point error variables

    if ((math_errhandling & MATH_ERRNO) != 0)
    {
        errno = 0;
    }
    if ((math_errhandling & MATH_ERREXCEPT) != 0 )
    {
        static_cast<void>(std::feclearexcept(FE_ALL_EXCEPT));
    }
#endif
} // clang-format on

} // namespace core
} // namespace dw

#endif // DWSHARED_CORE_SAFETY_MATHERRORS_HPP_
