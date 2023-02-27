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

#ifndef DW_FRAMEWORK_PARAMETERDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PARAMETERDESCRIPTOR_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>
#include <dwcgf/parameter/ParameterProvider.hpp>
#include <dw/core/container/StringView.hpp>
#include <dw/core/language/Tuple.hpp>

namespace dw
{
namespace framework
{

// Indices within the tuple describing constructor arguments
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_CONSTRUCTOR_ARGUMENT_TYPE{0U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR{1U};

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
constexpr std::tuple<> describeConstructorArguments()
{
    return std::make_tuple();
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T,
    typename Arg1>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1)
{
    return std::make_tuple(
        std::make_tuple(
            static_cast<Argument1T*>(nullptr),
            std::forward<const Arg1>(arg1)));
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T, typename Argument2T,
    typename Arg1, typename Arg2>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2)
{
    return std::make_tuple(
        std::make_tuple(
            static_cast<Argument1T*>(nullptr),
            std::forward<const Arg1>(arg1)),
        std::make_tuple(
            static_cast<Argument2T*>(nullptr),
            std::forward<const Arg2>(arg2)));
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T, typename Argument2T, typename Argument3T,
    typename Arg1, typename Arg2, typename Arg3>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2, const Arg3&& arg3)
{
    return std::make_tuple(
        std::make_tuple(
            static_cast<Argument1T*>(nullptr),
            std::forward<const Arg1>(arg1)),
        std::make_tuple(
            static_cast<Argument2T*>(nullptr),
            std::forward<const Arg2>(arg2)),
        std::make_tuple(
            static_cast<Argument3T*>(nullptr),
            std::forward<const Arg3>(arg3)));
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T, typename Argument2T, typename Argument3T, typename Argument4T,
    typename Arg1, typename Arg2, typename Arg3, typename Arg4>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2, const Arg3&& arg3, const Arg4&& arg4)
{
    return std::make_tuple(
        std::make_tuple(
            static_cast<Argument1T*>(nullptr),
            std::forward<const Arg1>(arg1)),
        std::make_tuple(
            static_cast<Argument2T*>(nullptr),
            std::forward<const Arg2>(arg2)),
        std::make_tuple(
            static_cast<Argument3T*>(nullptr),
            std::forward<const Arg3>(arg3)),
        std::make_tuple(
            static_cast<Argument4T*>(nullptr),
            std::forward<const Arg4>(arg4)));
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T, typename Argument2T, typename Argument3T, typename Argument4T, typename Argument5T,
    typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2, const Arg3&& arg3, const Arg4&& arg4, const Arg5&& arg5)
{
    return std::make_tuple(
        std::make_tuple(
            static_cast<Argument1T*>(nullptr),
            std::forward<const Arg1>(arg1)),
        std::make_tuple(
            static_cast<Argument2T*>(nullptr),
            std::forward<const Arg2>(arg2)),
        std::make_tuple(
            static_cast<Argument3T*>(nullptr),
            std::forward<const Arg3>(arg3)),
        std::make_tuple(
            static_cast<Argument4T*>(nullptr),
            std::forward<const Arg4>(arg4)),
        std::make_tuple(
            static_cast<Argument5T*>(nullptr),
            std::forward<const Arg5>(arg5)));
}

/**
 * Describe a specific constructor argument of a node.
 *
 * The function is used to create the arguments for describeConstructorArguments().
 * Each argument of the function is created by one of the describe parameter functions below.
 */
template <typename... Args>
constexpr auto describeConstructorArgument(const Args&&... args)
{
    return dw::core::make_tuple(
        std::forward<const Args>(args)...);
}

// Indices within the tuple describing parameters
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_TYPE_NAME{0U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_NAME{1U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_TYPE{2U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_SEMANTIC_TYPE{3U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_IS_INDEX{4U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_ARRAY_SIZE{5U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_MEMBER_PTRS{6U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PARAMETER_DEFAULT_VALUE{7U};

} // namespace framework
} // namespace dw

#define DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME_STR) TYPE_NAME_STR##_sv
#define DW_DESCRIBE_PARAMETER(TYPE_NAME, args...) dw::framework::describeParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe a parameter.
 *
 * The value is stored within the constructor argument where this parameter is described in (see describeConstructorArgument()).
 * When no member pointers are provided the value is stored directly in the constructor argument.
 * When one member pointer is provided the value is stored in that member of the constructor argument.
 * When more than one member pointer is provided the value is stored in the recursive member of the constructor argument.
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, const MemberPointers&&... memberPointers)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<S*>(nullptr),
        false,
        static_cast<size_t>(0),
        std::move(
            std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers)...)));
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ABSTRACT_PARAMETER(TYPE_NAME, args...) dw::framework::describeAbstractParameter<TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(TYPE_NAME, args...) dw::framework::describeAbstractArrayParameter<TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an abstract parameter.
 *
 * The value isn't stored anywhere automatically but the nodes create() function should retrieve the parameter from the provider and use custom logic to use the value.
 */
template <typename T>
constexpr auto describeAbstractParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<T*>(nullptr),
        false,
        static_cast<size_t>(0));
}

/**
 * Describe an abstract array parameter.
 *
 * The values aren't stored anywhere automatically but the nodes create() function should retrieve the parameters from the provider and use custom logic to use the values.
 */
template <typename T>
constexpr auto describeAbstractArrayParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, size_t arraySize)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<T*>(nullptr),
        false,
        arraySize);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_INDEX_PARAMETER(TYPE_NAME, args...) dw::framework::describeIndexParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeIndexParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an index parameter.
 *
 * The value represent an index which is used to retrieve a specific item from an array parameter identified by the semantic type.
 * Regarding the member pointers see describeParameter().
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeIndexParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, const MemberPointers&&... memberPointers)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<S*>(nullptr),
        true,
        static_cast<size_t>(0),
        std::move(
            std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers)...)));
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ARRAY_PARAMETER(TYPE_NAME, args...) dw::framework::describeArrayParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an array parameter.
 *
 * Regarding the member pointers see describeParameter().
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeArrayParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, size_t arraySize, const MemberPointers&&... memberPointers)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<S*>(nullptr),
        false,
        arraySize,
        std::move(
            std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers)...)));
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_UNNAMED_PARAMETER(TYPE_NAME, args...) dw::framework::describeUnnamedParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeUnnamedParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an unnamed parameter.
 *
 * The value isn't identified by a name but solely by the semantic type.
 * Regarding the member pointers see describeParameter().
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeUnnamedParameter(
    dw::core::StringView const&& typeName, const MemberPointers&&... memberPointers)
{
    return describeParameter<T, S>(
        std::move(typeName),
        ""_sv,
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER(TYPE_NAME, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an unnamed array parameter.
 *
 * The value isn't identified by a name but solely by the semantic type.
 * Regarding the member pointers see describeParameter().
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeUnnamedArrayParameter(
    dw::core::StringView const&& typeName, size_t arraySize, const MemberPointers&&... memberPointers)
{
    return describeArrayParameter<T, S>(
        std::move(typeName),
        ""_sv,
        arraySize,
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_PARAMETER_WITH_DEFAULT(TYPE_NAME, args...) dw::framework::describeParameterWithDefault<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe a parameter with a default value.
 *
 * See describeParameter().
 */
template <typename T, typename S, typename... MemberPointers>
constexpr auto describeParameterWithDefault(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, T defaultValue, const MemberPointers&&... memberPointers)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<S*>(nullptr),
        false,
        static_cast<size_t>(0),
        std::move(
            std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers)...)),
        std::move(defaultValue));
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ARRAY_PARAMETER_WITH_DEFAULT(TYPE_NAME, args...) dw::framework::describeArrayParameterWithDefault<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an array parameter with a default value.
 *
 * See describeArrayParameter().
 */
template <typename T, typename S, size_t ArraySize, typename... MemberPointers>
constexpr auto describeArrayParameterWithDefault(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, std::array<T, ArraySize> defaultValue, const MemberPointers&&... memberPointers)
{
    return std::make_tuple(
        std::move(typeName),
        std::move(parameterName),
        static_cast<T*>(nullptr),
        static_cast<S*>(nullptr),
        false,
        ArraySize,
        std::move(
            std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers)...)),
        std::move(defaultValue));
}

/// Get described parameters for the passed node.
template <typename NodeT>
constexpr auto describeParameters()
{
    return NodeT::describeParameters();
}

namespace detail
{

/// Populate parameter with default value (which is a no op for parameters which don't specify a default value).
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<std::tuple_size<Param>() <= PARAMETER_DEFAULT_VALUE, void>* = nullptr>
void populateDefault(const Param& param, MemberPtr& memberPtr)
{
    static_cast<void>(param);
    static_cast<void>(memberPtr);
}

/// Populate parameter with default value.
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<PARAMETER_DEFAULT_VALUE<std::tuple_size<Param>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    void populateDefault(const Param& param, MemberPtr& memberPtr)
{
    auto defaultValue = std::get<PARAMETER_DEFAULT_VALUE>(param);
    memberPtr         = defaultValue;
}

/// Populate array parameter with default value (which is a no op for parameters which don't specify a default value).
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<std::tuple_size<Param>() <= PARAMETER_DEFAULT_VALUE, void>* = nullptr>
void populateArrayDefault(const Param& param, MemberPtr& memberPtr, size_t index)
{
    static_cast<void>(param);
    static_cast<void>(memberPtr);
    static_cast<void>(index);
}

/// Populate array parameter with default value.
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<PARAMETER_DEFAULT_VALUE<std::tuple_size<Param>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    void populateArrayDefault(const Param& param, MemberPtr& memberPtr, size_t index)
{
    auto defaultValue = std::get<PARAMETER_DEFAULT_VALUE>(param);
    memberPtr[index]  = defaultValue[index];
}

/// Terminate recursion to get the constructor argument where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<std::tuple_size<MemberPtrs>() == 0, void>* = nullptr>
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    static_cast<void>(memberPtrs);
    return arg;
}

/// Terminate recursion to get the member pointer where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<MemberIndex + 1 == std::tuple_size<MemberPtrs>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    auto member = std::get<MemberIndex>(memberPtrs);
    return arg.*member;
}

/// Recursion to get the member pointer where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<MemberIndex + 1 < std::tuple_size<MemberPtrs>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    auto& member = std::get<MemberIndex>(memberPtrs);
    auto& m      = arg.*member;
    return getMemberPtr<MemberIndex + 1>(m, memberPtrs);
}

/// Get the number of parameters for a specific constructor argument.
template <typename NodeT, size_t ConstructorParameterIndex>
constexpr size_t constructorArgumentParameterSize()
{
    return dw::core::tuple_size<
        std::tuple_element_t<
            PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR,
            std::tuple_element_t<
                ConstructorParameterIndex,
                decltype(describeParameters<NodeT>())>>>();
}

/// Terminate recursion to populate each constructor argument using values from the parameter provider.
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<ParamIndex == constructorArgumentParameterSize<NodeT, ConstructorParameterIndex>(), void>* = nullptr>
void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg)
{
    static_cast<void>(arg);
    static_cast<void>(provider);
}

/// Check if a parameter is an index.
template <typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex>
constexpr bool isIndexParameter()
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));
    return std::get<PARAMETER_IS_INDEX>(param);
}

/// Check if a parameter is an array.
template <typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex>
constexpr bool isArrayParameter()
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));
    constexpr size_t arraySize = std::get<
        PARAMETER_ARRAY_SIZE>(param);
    return arraySize > 0;
}

/// Check if a parameter is abstract.
template <typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex>
constexpr bool isAbstractParameter()
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));
    return std::tuple_size<decltype(param)>() <= PARAMETER_MEMBER_PTRS;
}

/// Populate an abstract parameter which is a no op.
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<isAbstractParameter<NodeT, ConstructorParameterIndex, ParamIndex>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameter(const ParameterProvider& provider, ArgType& arg)
{
    static_cast<void>(arg);
    static_cast<void>(provider);
}

/**
 * Populate a non-array non-index pararmeter using value from the parameter provider.
 *
 * If the parameter value isn't available, fall back to use the default value if available.
 */
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<!isAbstractParameter<NodeT, ConstructorParameterIndex, ParamIndex>() && !isArrayParameter<NodeT, ConstructorParameterIndex, ParamIndex>() && !isIndexParameter<NodeT, ConstructorParameterIndex, ParamIndex>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameter(const ParameterProvider& provider, ArgType& arg)
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));

    using DataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_TYPE, decltype(param)>>;
    using SemanticDataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_SEMANTIC_TYPE, decltype(param)>>;

    DataType& memberPtr = getMemberPtr<0>(arg, std::get<PARAMETER_MEMBER_PTRS>(param));
    bool hadParameter   = provider.getOptional<SemanticDataType, DataType>(std::get<PARAMETER_NAME>(param), &memberPtr);
    if (!hadParameter)
    {
        populateDefault(param, memberPtr);
    }
}

/**
 * Populate a non-array index pararmeter using value from the parameter provider.
 *
 * If the parameter value or index within the array isn't available, fall back to use the default value if available.
 */
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<!isAbstractParameter<NodeT, ConstructorParameterIndex, ParamIndex>() && !isArrayParameter<NodeT, ConstructorParameterIndex, ParamIndex>() && isIndexParameter<NodeT, ConstructorParameterIndex, ParamIndex>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameter(const ParameterProvider& provider, ArgType& arg)
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));

    using DataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_TYPE, decltype(param)>>;
    using SemanticDataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_SEMANTIC_TYPE, decltype(param)>>;
    DataType& memberPtr = getMemberPtr<0>(arg, std::get<PARAMETER_MEMBER_PTRS>(param));

    size_t index      = static_cast<size_t>(-1);
    bool hadParameter = provider.getOptional<size_t, size_t>(std::get<PARAMETER_NAME>(param), &index);
    if (hadParameter)
    {
        hadParameter = provider.getOptional<SemanticDataType, DataType>("", index, &memberPtr);
    }
    if (!hadParameter)
    {
        populateDefault(param, memberPtr);
    }
}

/**
 * Populate an array pararmeter using value from the parameter provider.
 *
 * If the parameter value for each index within the array isn't available, fall back to use the default value if available.
 */
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<!isAbstractParameter<NodeT, ConstructorParameterIndex, ParamIndex>() && isArrayParameter<NodeT, ConstructorParameterIndex, ParamIndex>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameter(const ParameterProvider& provider, ArgType& arg)
{
    constexpr auto param = dw::core::get<ParamIndex>(
        std::get<PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>())));

    using DataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_TYPE, decltype(param)>>;
    using SemanticDataType = std::remove_pointer_t<
        std::tuple_element_t<
            PARAMETER_SEMANTIC_TYPE, decltype(param)>>;

    constexpr size_t arraySize = std::get<PARAMETER_ARRAY_SIZE>(
        param);

    DataType(&memberPtr)[arraySize] = getMemberPtr<0>(arg, std::get<PARAMETER_MEMBER_PTRS>(param));
    for (size_t i = 0; i < arraySize; ++i)
    {
        bool hadParameter = provider.getOptional<SemanticDataType, DataType>(std::get<PARAMETER_NAME>(param), i, &memberPtr[i]);
        if (!hadParameter)
        {
            populateArrayDefault(param, memberPtr, i);
        }
    }
}

/// Recursion to populate each constructor argument using values from the parameter provider.
template <
    typename NodeT, size_t ConstructorParameterIndex, size_t ParamIndex, typename ArgType,
    typename std::enable_if_t<ParamIndex<constructorArgumentParameterSize<NodeT, ConstructorParameterIndex>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg)
{
    populateParameter<NodeT, ConstructorParameterIndex, ParamIndex>(provider, arg);

    populateParametersRecursion<NodeT, ConstructorParameterIndex, ParamIndex + 1>(provider, arg);
}

/**
 * Populate the constructor arguments using values from the parameter provider.
 *
 * @throws Exception if retrieving any parameter value fails
 */
template <typename NodeT, size_t ConstructorParameterIndex, typename ArgType>
// TODO(dwplc): FP -- The other populateParameters() functions are defined in the parent namespace
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(const ParameterProvider& provider, ArgType& arg)
{
    try
    {
        populateParametersRecursion<NodeT, ConstructorParameterIndex, 0>(provider, arg);
    }
    catch (Exception& e)
    {
        throw Exception(e.status(), "Exception while populating parameters of mangled node type ", typeid(NodeT).name(), ": ", e.messageStr());
    }
}

} // namespace detail

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 1, void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    auto& arg0 = std::get<0>(constructorArguments);
    detail::populateParameters<NodeT, 0>(provider, arg0);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 2, void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    auto& arg0 = std::get<0>(constructorArguments);
    detail::populateParameters<NodeT, 0>(provider, arg0);
    auto& arg1 = std::get<1>(constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, arg1);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 3, void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    auto& arg0 = std::get<0>(constructorArguments);
    detail::populateParameters<NodeT, 0>(provider, arg0);
    auto& arg1 = std::get<1>(constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, arg1);
    auto& arg2 = std::get<2>(constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, arg2);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 4, void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    auto& arg0 = std::get<0>(constructorArguments);
    detail::populateParameters<NodeT, 0>(provider, arg0);
    auto& arg1 = std::get<1>(constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, arg1);
    auto& arg2 = std::get<2>(constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, arg2);
    auto& arg3 = std::get<3>(constructorArguments);
    detail::populateParameters<NodeT, 3>(provider, arg3);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 5, void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    auto& arg0 = std::get<0>(constructorArguments);
    detail::populateParameters<NodeT, 0>(provider, arg0);
    auto& arg1 = std::get<1>(constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, arg1);
    auto& arg2 = std::get<2>(constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, arg2);
    auto& arg3 = std::get<3>(constructorArguments);
    detail::populateParameters<NodeT, 3>(provider, arg3);
    auto& arg4 = std::get<4>(constructorArguments);
    detail::populateParameters<NodeT, 4>(provider, arg4);
}

namespace detail
{

// Get the Number of constructor arguments of the passed node.
template <typename NodeT>
constexpr std::size_t parameterConstructorArgumentSize()
{
    return std::tuple_size<decltype(describeParameters<NodeT>())>::value;
}

/// Terminate recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex == parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
auto createConstructorArguments()
{
    return std::make_tuple();
}

/// Recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex<parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    auto createConstructorArguments()
{
    using NodeParams = decltype(describeParameters<NodeT>());

    using ConstructorParameter = std::tuple_element_t<ConstructorArgumentIndex, NodeParams>;
    using ArgType              = std::remove_pointer_t<
        typename std::tuple_element_t<PARAMETER_CONSTRUCTOR_ARGUMENT_TYPE, ConstructorParameter>>;
    ArgType arg{};

    return std::tuple_cat(std::make_tuple(arg), createConstructorArguments<NodeT, ConstructorArgumentIndex + 1>());
}

} // namespace detail

/// Create a tuple of constructor argument needed by the constructor of the passed node.
template <typename NodeT>
auto createConstructorArguments()
{
    return detail::createConstructorArguments<NodeT, 0>();
}

namespace detail
{

/// Instantiate a node using the passed tuple of constructor arguments.
template <class T, class Tuple, size_t... Is>
// TODO(dwplc): FP -- The other parameterSize() functions are defined in a namespace
// coverity[autosar_cpp14_a2_10_5_violation]
auto makeUniqueFromTuple(const Tuple&& tuple, std::index_sequence<Is...>) -> std::unique_ptr<T>
{
    return std::unique_ptr<T>(new T{std::get<Is>(std::move(tuple))...});
}
}

/// Instantiate a node using the passed constructor arguments.
template <typename NodeT, class ConstructorArguments>
auto makeUniqueFromTuple(const ConstructorArguments&& constructorArguments) -> std::unique_ptr<NodeT>
{
    return detail::makeUniqueFromTuple<NodeT>(
        std::move(constructorArguments),
        std::make_index_sequence<std::tuple_size<std::decay_t<ConstructorArguments>>::value>{});
}

/**
 * Instantiate a node using parameters from the passed provider.
 *
 * Syntactic sugar chaining the following calls:
 * - createConstructorArguments()
 * - populateParameters()
 * - makeUniqueFromTuple()
 */
template <typename NodeT>
auto create(const ParameterProvider& provider) -> std::unique_ptr<NodeT>
{
    auto constructorArguments = createConstructorArguments<NodeT>();
    populateParameters<NodeT>(constructorArguments, provider);
    return makeUniqueFromTuple<NodeT>(std::move(constructorArguments));
}

// Number of parameters (sum across all constructor arguments)
namespace detail
{

/// Terminate recursion to count the number of parameters for each constructor argument of a given node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex == parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
// TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr std::size_t parameterSize()
{
    return 0;
}

/// Recursion to count the number of parameters for each constructor argument of a given node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex<parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    constexpr std::size_t parameterSize()
{
    return constructorArgumentParameterSize<NodeT, ConstructorArgumentIndex>() + parameterSize<NodeT, ConstructorArgumentIndex + 1>();
}

} // namespace detail

/// Get the number of parameters for a given node.
// TODO(dwplc): FP -- The other parameterSize() functions are defined in a namespace
// coverity[autosar_cpp14_a2_10_5_violation]
template <typename NodeT>
constexpr std::size_t parameterSize()
{
    return detail::parameterSize<NodeT, 0>();
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PARAMETERDESCRIPTOR_HPP_
