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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CONTEXTPARAMETERPROVIDER_HPP_
#define DW_FRAMEWORK_CONTEXTPARAMETERPROVIDER_HPP_

#include <dwcgf/parameter/TypeBasedParameterProvider.hpp>
#include <dw/core/context/Context.h>

namespace dw
{
namespace framework
{

/// A parameter provider for the dwContextHandle_t.
class ContextParameterProvider : public ITypeBasedParameterProviderChild
{
protected:
    /// Copy constructor.
    ContextParameterProvider(ContextParameterProvider const&) = default;
    /// Move constructor.
    ContextParameterProvider(ContextParameterProvider&&) = default;
    /// Copy assignment operator.
    ContextParameterProvider& operator=(ContextParameterProvider const&) & = default;
    /// Move assignment operator.
    ContextParameterProvider& operator=(ContextParameterProvider&&) & = default;

public:
    /// Constructor.
    ContextParameterProvider(dwContextHandle_t const ctx) noexcept;

    /// Destructor.
    ~ContextParameterProvider() override = default;

    /// @see dw::framework::ITypeBasedParameterProviderChild::registerAt()
    void registerAt(TypeBasedParameterProvider& provider) const override;

protected:
    /// Handler function to retrieve a dwContextHandle_t.
    bool getContextHandle(void* const out) const noexcept;

private:
    // The context handle
    dwContextHandle_t const m_ctx;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CONTEXTPARAMETERPROVIDER_HPP_
