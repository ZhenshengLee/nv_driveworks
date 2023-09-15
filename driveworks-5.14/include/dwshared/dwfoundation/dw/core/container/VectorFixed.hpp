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

#ifndef DW_CORE_VECTORFIXED_HPP_
#define DW_CORE_VECTORFIXED_HPP_

#include "impl/VectorFixed/StaticVectorFixed.hpp"
#include "impl/VectorFixed/HeapVectorFixed.hpp"

namespace dw
{
namespace core
{

/////////////////////////////////////////////
// Old class names using templates
// The name HeapVectorFixed is temporary.
// The final names will be StaticVectorFixed<T,N> and VectorFixed<T>.
// This will require a follow-up MR with a massive rename.
//
// TODO(danielh): remove these and replace names in existing code

/// Static VectorFixed
template <typename T, size_t C = 0>
class VectorFixed : public StaticVectorFixed<T, C>
{
public:
    /// VectorFixed capacity
    static constexpr size_t CAPACITY_AT_COMPILE_TIME = C;

    using Base = StaticVectorFixed<T, C>; ///< Type of base class

    using Base::Base; // import constructor from base class

    VectorFixed() = default;

    /// Copy constructor
    VectorFixed(const VectorFixed& other)
        : Base(other)
    {
    }

    /// Move constructor
    VectorFixed(VectorFixed&& other)
        : Base(std::move(other))
    {
    }

    using Base::operator=;

    /// Copy operator
    VectorFixed& operator=(const VectorFixed& other)
    {
        Base::operator=(other);
        return *this;
    }

    /// Move operator
    VectorFixed& operator=(VectorFixed&& other)
    {
        Base::operator=(std::move(other));
        return *this;
    }
};

/// Heap VectorFixed
template <typename T>
class VectorFixed<T, 0> : public HeapVectorFixed<T>
{
public:
    using Base = HeapVectorFixed<T>; ///< Type of base class

    using Base::Base; // import constructor from base class

    VectorFixed() = default;

    /// Copy constructor
    VectorFixed(const VectorFixed& other)
        : Base(other)
    {
    }

    /// Move constructor
    VectorFixed(VectorFixed&& other)
        : Base(std::move(other))
    {
    }

    using Base::operator=;

    /// Copy operator
    VectorFixed& operator=(const VectorFixed& other)
    {
        Base::operator=(other);
        return *this;
    }

    /// Move operator
    VectorFixed& operator=(VectorFixed&& other)
    {
        Base::operator=(std::move(other));
        return *this;
    }
};

} // namespace core
} // namespace dw

#endif
