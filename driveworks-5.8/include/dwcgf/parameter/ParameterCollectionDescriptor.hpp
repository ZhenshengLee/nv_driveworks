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

#ifndef DW_FRAMEWORK_PARAMETERCOLLECTIONDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PARAMETERCOLLECTIONDESCRIPTOR_HPP_

#include <dwcgf/parameter/ParameterDescriptor.hpp>

#include <dw/core/container/VectorFixed.hpp>

#include <modern-json/json.hpp>

#include <memory>

namespace dw
{
namespace framework
{

/// The description of a parameter.
class ParameterDescriptor
{
public:
    /// Constructor.
    ParameterDescriptor(
        dw::core::StringView const& name,
        dw::core::StringView const& typeName,
        const bool isIndex,
        size_t const arraySize) noexcept;
    /// Destructor
    virtual ~ParameterDescriptor() = default;
    /// Copy constructor.
    ParameterDescriptor(ParameterDescriptor const&) = delete;
    /// Move constructor.
    ParameterDescriptor(ParameterDescriptor&&) = delete;
    /// Copy assignment operator.
    ParameterDescriptor& operator=(ParameterDescriptor const&) & = delete;
    /// Move assignment operator.
    ParameterDescriptor& operator=(ParameterDescriptor&&) & = delete;
    /// Get the parameter name, can be empty for unnamed parameters.
    dw::core::StringView const& getName() const noexcept;
    /// Get the C++ type name of the parameter.
    dw::core::StringView const& getTypeName() const noexcept;
    /// Check if parameter is an index.
    bool isIndex() const noexcept;
    /// Get the array size, 0 for non-array parameters.
    size_t getArraySize() const noexcept;
    /// Add the default value to the passed JSON object (only used by ParameterDescriptorWithDefault()).
    virtual void addDefault(nlohmann::ordered_json& object) const noexcept;

private:
    /// The name, can be empty for unnamed parameters.
    dw::core::StringView m_name;
    /// The C++ type name.
    dw::core::StringView m_typeName;
    /// If parameter is an index.
    bool m_isIndex;
    /// The array size, 0 for non-array parameters.
    size_t m_arraySize;
};

namespace detail
{

/// Convert non-enum value to JSON.
template <
    typename DefaultType,
    std::enable_if_t<
        !std::is_enum<DefaultType>::value>* = nullptr>
auto getDefaultValueForJson(DefaultType defaultValue)
{
    return nlohmann::json(defaultValue);
}

/// Convert enum value to JSON by mapping the numeric value to the enumerator string.
template <
    typename DefaultType,
    std::enable_if_t<
        std::is_enum<DefaultType>::value>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
auto getDefaultValueForJson(DefaultType defaultValue)
{
    StringView name = mapEnumValueToName<DefaultType>(defaultValue);
    std::string nameStr{name.data(), name.size()};
    return nlohmann::json(nameStr);
}

/// Convert array of enum values to JSON by mapping the numeric values to the enumerator strings.
template <
    typename DefaultType, size_t N,
    std::enable_if_t<
        std::is_enum<DefaultType>::value>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
auto getDefaultValueForJson(const std::array<DefaultType, N>& defaultValues)
{
    std::array<std::string, N> nameStrings;
    for (size_t i = 0; i < N; ++i)
    {
        StringView name = mapEnumValueToName<DefaultType>(defaultValues[i]);
        std::string nameStr{name.data(), name.size()};
        nameStrings[i] = nameStr;
    }
    return nlohmann::json(nameStrings);
}

} // namespace detail

/// The description of a parameter with a default value.
// static_assert needed covering all possible type accepted by nlohmann::json()
// beside enums and std::arrays of such types
// coverity[autosar_cpp14_a14_1_1_violation]
template <typename DefaultType>
class ParameterDescriptorWithDefault : public ParameterDescriptor
{

public:
    /// Constructor.
    ParameterDescriptorWithDefault(
        dw::core::StringView name,
        dw::core::StringView typeName,
        bool isIndex,
        size_t arraySize,
        DefaultType defaultValue)
        : ParameterDescriptor(name, typeName, isIndex, arraySize)
        , m_defaultValue(defaultValue)
    {
    }
    /// Destructor
    ~ParameterDescriptorWithDefault() override = default;
    /// Copy constructor.
    ParameterDescriptorWithDefault(ParameterDescriptorWithDefault const&) = delete;
    /// Move constructor.
    ParameterDescriptorWithDefault(ParameterDescriptorWithDefault&&) = delete;
    /// Copy assignment operator.
    auto operator=(ParameterDescriptorWithDefault const&) -> ParameterDescriptorWithDefault& = delete;
    /// Move assignment operator.
    auto operator=(ParameterDescriptorWithDefault &&) -> ParameterDescriptorWithDefault& = delete;
    /// Add the default value to the passed JSON object.
    void addDefault(nlohmann::ordered_json& object) const noexcept override
    {
        object["default"] = detail::getDefaultValueForJson(m_defaultValue);
    }

private:
    /// The default value.
    DefaultType m_defaultValue;
};

/// A collection of parameter descriptors.
class ParameterCollectionDescriptor
{
public:
    /// Constructor.
    explicit ParameterCollectionDescriptor(size_t const capacity);

    /// Get the number of parameter descriptors.
    size_t getSize() const;

    /// Get the parameter descriptor for the passed index.
    const std::shared_ptr<const ParameterDescriptor>& getDescriptor(size_t const index) const;

    /**
     * Get the index of the parameter descriptor with the passed name.
     *
     * @throws Exception if a parameter descriptor with the passed name isn't in the collection
     */
    size_t getIndex(dw::core::StringView const& identifier) const;

    /**
     * Get the parameter descriptor with the passed name.
     *
     * @throws Exception if a parameter descriptor with the passed name isn't in the collection
     */
    const std::shared_ptr<const ParameterDescriptor>& getDescriptor(dw::core::StringView const& identifier) const;

    /// Check if the passed index is valid, which mean is within [0, size()).
    bool isValid(size_t const index) const;

    /// Check if the passed parameter name matches a parameter descriptor in the collection.
    bool isValid(dw::core::StringView const& identifier) const noexcept;

    /**
     * Add a parameter descriptor to the collection.
     *
     * @throws Exception if the same parameter descriptor is already in the collection or the collection is already at capacity
     */
    void addDescriptor(const std::shared_ptr<const ParameterDescriptor>& descriptor);

    /// Terminate recursion to add parameter descriptors for each node constructor argument to the collection.
    template <
        typename NodeT, size_t ConstructorArgumentIndex,
        typename std::enable_if_t<ConstructorArgumentIndex == std::tuple_size<decltype(describeParameters<NodeT>())>::value, void>* = nullptr>
    void addDescriptors()
    {
    }

    /// Recursion to add parameter descriptors for each node constructor argument to the collection.
    template <
        typename NodeT, size_t ConstructorArgumentIndex,
        typename std::enable_if_t<ConstructorArgumentIndex<std::tuple_size<decltype(describeParameters<NodeT>())>::value, void>* = nullptr> void addDescriptors()
    {
        auto t         = std::get<ConstructorArgumentIndex>(describeParameters<NodeT>());
        const auto& t2 = std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(t);
        // all parameters of this constructor argument
        addDescriptors<0>(t2);
        // next constructor argument
        addDescriptors<NodeT, ConstructorArgumentIndex + 1>();
    }

private:
    /// Terminate recursion to add parameter descriptors for each described parameter.
    template <
        size_t ParameterIndex, typename ParamsT,
        typename std::enable_if_t<ParameterIndex == dw::core::tuple_size<ParamsT>::value, void>* = nullptr>
    void addDescriptors(const ParamsT& params)
    {
        static_cast<void>(params);
    }

    /// Recursion to add parameter descriptors for each described parameter.
    template <
        size_t ParameterIndex, typename ParamsT,
        typename std::enable_if_t<ParameterIndex<dw::core::tuple_size<ParamsT>::value, void>* = nullptr> void addDescriptors(const ParamsT& params)
    {
        const auto& p = dw::core::get<ParameterIndex>(params);
        addDescriptor(p);
        addDescriptors<ParameterIndex + 1>(params);
    }

    /// Add a parameter descriptor if the parameter doesn't have a default value.
    template <
        typename ParamT,
        typename std::enable_if_t<std::tuple_size<ParamT>::value <= PARAMETER_DEFAULT_VALUE, void>* = nullptr>
    void addDescriptor(const ParamT& param)
    {
        const auto d = std::make_shared<const ParameterDescriptor>(
            std::get<PARAMETER_NAME>(param),
            std::get<PARAMETER_TYPE_NAME>(param),
            std::get<PARAMETER_IS_INDEX>(param),
            std::get<PARAMETER_ARRAY_SIZE>(param));
        addDescriptor(d);
    }

    /// Add a parameter descriptor if the parameter has a default value.
    template <
        typename ParamT,
        typename std::enable_if_t<PARAMETER_DEFAULT_VALUE<std::tuple_size<ParamT>::value, void>* = nullptr> void addDescriptor(const ParamT& param)
    {
        using DefaultT = typename std::tuple_element_t<PARAMETER_DEFAULT_VALUE, ParamT>;
        const auto d   = std::make_shared<const ParameterDescriptorWithDefault<DefaultT>>(
            std::get<PARAMETER_NAME>(param),
            std::get<PARAMETER_TYPE_NAME>(param),
            std::get<PARAMETER_IS_INDEX>(param),
            std::get<PARAMETER_ARRAY_SIZE>(param),
            std::get<PARAMETER_DEFAULT_VALUE>(param));
        addDescriptor(d);
    }

    /// The parameter descriptors
    dw::core::VectorFixed<std::shared_ptr<const ParameterDescriptor>> m_descriptors;
};

/// Create a parameter collection descriptor for a give node.
template <typename NodeT>
static ParameterCollectionDescriptor createParameterCollectionDescriptor()
{
    ParameterCollectionDescriptor d(parameterSize<NodeT>());
    d.addDescriptors<NodeT, 0>();
    return d;
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PARAMETERCOLLECTIONDESCRIPTOR_HPP_
