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

#ifndef DW_FRAMEWORK_PASSCOLLECTIONDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PASSCOLLECTIONDESCRIPTOR_HPP_

#include <dwcgf/pass/PassDescriptor.hpp>

#include <dw/core/container/VectorFixed.hpp>

namespace dw
{
namespace framework
{

/// The description of a pass.
class PassDescriptor
{
public:
    /// Constructor.
    PassDescriptor(dw::core::StringView const& name, dwProcessorType const processorType, const bool hasExplicitDependencies, size_t const dependencies);

    /// Get the pass name.
    dw::core::StringView const& getName() const noexcept;

    /// Get the processor type running on.
    dwProcessorType getProcessorType() const noexcept;

    /// Check if the pass specifies explicit dependencies.
    bool hasExplicitDependencies() const noexcept;

    /// Get number of dependencies.
    size_t getNumberOfDependencies() const;

    /// Get the dependent on pass name by index.
    dw::core::StringView const& getDependency(size_t const index) const;

    /// Add another pass name as a dependency.
    void addDependency(dw::core::StringView const& dependency);

private:
    /// The pass name.
    dw::core::StringView m_name;
    /// The processor type this pass runs on.
    dwProcessorType m_processorType;
    /// If the pass has explicit dependencies.
    bool m_hasExplicitDependencies;
    /// The dependencies.
    dw::core::VectorFixed<dw::core::StringView> m_dependencies;
};

/// A collection of pass descriptors.
class PassCollectionDescriptor
{
public:
    /// Constructor.
    explicit PassCollectionDescriptor(size_t const capacity);

    /// Get the number of pass descriptors.
    size_t getSize() const;

    /// Get the pass descriptor for the passed index.
    const PassDescriptor& getDescriptor(size_t const index) const;

    /**
     * Get the index of the pass descriptor with the passed name.
     *
     * @throws Exception if a pass descriptor with the passed name isn't in the collection
     */
    size_t getIndex(dw::core::StringView const& identifier) const;

    /**
     * Get the pass descriptor with the passed name.
     *
     * @throws Exception if a pass descriptor with the passed name isn't in the collection
     */
    const PassDescriptor& getDescriptor(dw::core::StringView const& identifier) const;

    /// Check if the passed index is valid, which mean is within [0, size()).
    bool isValid(size_t const index) const noexcept;

    /// Check if the passed pass name matches a pass descriptor in the collection.
    bool isValid(dw::core::StringView const& identifier) const noexcept;

    /**
     * Add a pass descriptor to the collection.
     *
     * @throws Exception if the same pass descriptor is already in the collection or the collection is already at capacity
     */
    void addDescriptor(const PassDescriptor& descriptor);

private:
    /// The pass descriptors
    dw::core::VectorFixed<PassDescriptor> m_descriptors;
};

namespace detail
{

/// Terminate recursion to add pass descriptors for each described pass.
template <
    typename NodeT, size_t Index,
    typename std::enable_if_t<Index == passSize<NodeT>(), void>* = nullptr>
void addPassDescriptor(PassCollectionDescriptor& collection)
{
    (void)collection;
}

/// Recursion to add pass descriptors for each described pass.
template <
    typename NodeT, size_t Index,
    typename std::enable_if_t<Index<passSize<NodeT>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    void addPassDescriptor(PassCollectionDescriptor& collection)
{
    constexpr auto dependencies = passDependencies<NodeT, Index>();
    PassDescriptor descriptor(passName<NodeT, Index>(), passProcessorType<NodeT, Index>(), hasPassDependencies<NodeT, Index>(), dependencies.size());
    for (size_t i = 0; i < dependencies.size(); ++i)
    {
        descriptor.addDependency(dependencies[i]);
    }
    collection.addDescriptor(descriptor);

    addPassDescriptor<NodeT, Index + 1>(collection);
}

} // namespace detail

/// Create a pass collection descriptor for a give node.
template <typename NodeT>
static PassCollectionDescriptor createPassCollectionDescriptor()
{
    PassCollectionDescriptor collection(passSize<NodeT>());
    detail::addPassDescriptor<NodeT, 0>(collection);
    return collection;
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PASSCOLLECTIONDESCRIPTOR_HPP_
