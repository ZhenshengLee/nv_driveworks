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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/language/Optional.hpp>
#include <dwshared/dwfoundation/dw/core/language/Tuple.hpp>

namespace dw
{
namespace framework
{

template <typename ConstructorArgumentT, typename ParameterDescriptorsT>
struct ConstructorArgumentDescriptorT
{
    static_assert(std::is_constructible<ConstructorArgumentT>::value, "ConstructorArgumentT must be constructible");

    // coverity[autosar_cpp14_a0_1_6_violation]
    // coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
    using ConstructorArgumentType  = ConstructorArgumentT;
    using ParameterDescriptorsType = ParameterDescriptorsT;
    ParameterDescriptorsType parameterDescriptors;

    constexpr ConstructorArgumentDescriptorT(ParameterDescriptorsType&& parameterDescriptors_)
        : parameterDescriptors(std::move(parameterDescriptors_))
    {
    }
};

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * It describes a node constructor with no arguments.
 */
constexpr std::tuple<> describeConstructorArguments()
{
    return std::make_tuple();
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * It describes a node constructor with one argument.
 * Each argument of the function is created by describeConstructorArgument().
 */
template <typename Arg1>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeConstructorArguments(Arg1&& arg1) -> std::tuple<Arg1>
{
    return std::make_tuple(std::forward<Arg1>(arg1));
}

/**
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * It recursively describes a node constructor with more than one arguments.
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Arg1, typename... ArgRest,
    typename std::enable_if_t<sizeof...(ArgRest) != 0>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeConstructorArguments(Arg1&& arg1, ArgRest&&... argRest)
{
    return std::tuple_cat(
        describeConstructorArguments<Arg1>(std::forward<Arg1>(arg1)),
        describeConstructorArguments<ArgRest...>(std::forward<ArgRest>(argRest)...));
}

/**
 * Describe a specific constructor argument of a node.
 *
 * The function is used to create the arguments for describeConstructorArguments().
 * Each argument of the function is created by one of the describe parameter functions below.
 */
template <typename ConstructorArgumentT, typename... Args>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeConstructorArgument(const Args&&... args) -> ConstructorArgumentDescriptorT<ConstructorArgumentT, dw::core::Tuple<Args...>>
{
    return ConstructorArgumentDescriptorT<ConstructorArgumentT, dw::core::Tuple<Args...>>(
        dw::core::make_tuple<const Args...>(std::forward<const Args>(args)...));
}

template <typename Type_, typename SemanticType_, bool IsIndex, size_t ArraySize, bool IsAbstract, bool HasDefault, typename... MemberPointers>
struct ParameterDescriptorT
{
    static_assert(std::is_constructible<Type_>::value, "Type_ must be constructible");

    dw::core::StringView typeName;
    dw::core::StringView parameterName;
    // coverity[autosar_cpp14_a0_1_6_violation]
    using Type = Type_;
    // coverity[autosar_cpp14_a0_1_6_violation]
    // coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
    using SemanticType = SemanticType_;
    static constexpr bool IS_INDEX{IsIndex};
    static constexpr size_t ARRAY_SIZE{ArraySize};
    static constexpr bool IS_ABSTRACT{IsAbstract};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr bool HAS_DEFAULT{HasDefault};
    std::tuple<const MemberPointers...> memberPointers;

    constexpr ParameterDescriptorT(dw::core::StringView&& typeName_, dw::core::StringView&& parameterName_, const MemberPointers&&... memberPointers_)
        : typeName{std::move(typeName_)}
        , parameterName{std::move(parameterName_)}
        , memberPointers(std::make_tuple<const MemberPointers...>(std::forward<const MemberPointers>(memberPointers_)...))
    {
    }
};

template <typename Type_, typename SemanticType_, bool IsIndex, size_t ArraySize, bool IsAbstract, bool HasDefault, typename... MemberPointers>
constexpr bool ParameterDescriptorT<Type_, SemanticType_, IsIndex, ArraySize, IsAbstract, HasDefault, MemberPointers...>::IS_INDEX;

template <typename Type_, typename SemanticType_, bool IsIndex, size_t ArraySize, bool IsAbstract, bool HasDefault, typename... MemberPointers>
constexpr size_t ParameterDescriptorT<Type_, SemanticType_, IsIndex, ArraySize, IsAbstract, HasDefault, MemberPointers...>::ARRAY_SIZE;

template <typename Type_, typename SemanticType_, bool IsIndex, size_t ArraySize, bool IsAbstract, bool HasDefault, typename... MemberPointers>
constexpr bool ParameterDescriptorT<Type_, SemanticType_, IsIndex, ArraySize, IsAbstract, HasDefault, MemberPointers...>::HAS_DEFAULT;

template <typename Type, typename SemanticType, bool IsIndex, size_t ArraySize, bool IsAbstract, typename DefaultType, typename... MemberPointers>
struct ParameterDescriptorWithDefaultT : public ParameterDescriptorT<Type, SemanticType, IsIndex, ArraySize, IsAbstract, true, MemberPointers...>
{
    static_assert(std::is_trivially_move_constructible<DefaultType>(), "DefaultType must be trivially move constructible");

    DefaultType defaultValue;

    constexpr ParameterDescriptorWithDefaultT(dw::core::StringView&& typeName_, dw::core::StringView&& parameterName_, DefaultType&& defaultValue_, const MemberPointers&&... memberPointers_)
        : ParameterDescriptorT<Type, SemanticType, IsIndex, ArraySize, IsAbstract, true, MemberPointers...>{std::move(typeName_), std::move(parameterName_), std::forward<const MemberPointers>(memberPointers_)...}
        , defaultValue{std::move(defaultValue_)}
    {
    }
};

} // namespace framework
} // namespace dw

#define DW_PARAMETER_TYPE_NAME_STRING_VIEW_IMPL(TYPE_NAME_STR) \
    dw::core::StringView { TYPE_NAME_STR }
#define DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME) DW_PARAMETER_TYPE_NAME_STRING_VIEW_IMPL(#TYPE_NAME)
#define DW_DESCRIBE_PARAMETER(TYPE_NAME, args...) dw::framework::describeParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)
#define DW_DESCRIBE_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeParameter(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName, const MemberPointers&&... memberPointers) -> ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>
{
    return ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ABSTRACT_PARAMETER(TYPE_NAME, args...) dw::framework::describeAbstractParameter<TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)
#define DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeAbstractArrayParameter<TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), PARAM_NAME, ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeAbstractParameter(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName)
{
    return ParameterDescriptorT<T, T, false, 0, true, false>(
        std::move(typeName),
        std::move(parameterName));
}

/**
 * Describe an abstract array parameter.
 *
 * The values aren't stored anywhere automatically but the nodes create() function should retrieve the parameters from the provider and use custom logic to use the values.
 */
template <typename T, size_t ArraySize>
constexpr auto describeAbstractArrayParameter(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName)
{
    return ParameterDescriptorT<T, T, false, ArraySize, true, false>(
        std::move(typeName),
        std::move(parameterName));
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_INDEX_PARAMETER(TYPE_NAME, args...) dw::framework::describeIndexParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)
#define DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeIndexParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeIndexParameter(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName, const MemberPointers&&... memberPointers)
{
    return ParameterDescriptorT<T, S, true, 0, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ARRAY_PARAMETER(TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeArrayParameter<TYPE_NAME, TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), PARAM_NAME, ##args)
#define DW_DESCRIBE_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), PARAM_NAME, ##args)

namespace dw
{
namespace framework
{

/**
 * Describe an array parameter.
 *
 * Regarding the member pointers see describeParameter().
 */
template <typename T, typename S, size_t ArraySize, typename... MemberPointers>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeArrayParameter(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName, const MemberPointers&&... memberPointers)
{
    return ParameterDescriptorT<T, S, false, ArraySize, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_UNNAMED_PARAMETER(TYPE_NAME, args...) dw::framework::describeUnnamedParameter<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)
#define DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, args...) dw::framework::describeUnnamedParameter<TYPE_NAME, SEMANTIC_TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeUnnamedParameter(
    dw::core::StringView&& typeName, const MemberPointers&&... memberPointers) -> ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>
{
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3949905
    return describeParameter<T, S, MemberPointers...>(
        std::move(typeName),
        dw::core::StringView{""},
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER(TYPE_NAME, ARRAY_SIZE, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)
#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
template <typename T, typename S, size_t ArraySize, typename... MemberPointers>
constexpr auto describeUnnamedArrayParameter(
    dw::core::StringView&& typeName, const MemberPointers&&... memberPointers)
{
    return describeArrayParameter<T, S, ArraySize>(
        std::move(typeName),
        ""_sv,
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_PARAMETER_WITH_DEFAULT(TYPE_NAME, args...) dw::framework::describeParameterWithDefault<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeParameterWithDefault(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName, T&& defaultValue, const MemberPointers&&... memberPointers) -> ParameterDescriptorWithDefaultT<T, S, false, 0, false, T, MemberPointers...>
{
    return ParameterDescriptorWithDefaultT<T, S, false, 0, false, T, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::move(defaultValue),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ARRAY_PARAMETER_WITH_DEFAULT(TYPE_NAME, args...) dw::framework::describeArrayParameterWithDefault<TYPE_NAME, TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(TYPE_NAME), ##args)

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeArrayParameterWithDefault(
    dw::core::StringView&& typeName, dw::core::StringView&& parameterName, std::array<T, ArraySize> defaultValue, const MemberPointers&&... memberPointers) -> ParameterDescriptorWithDefaultT<T, S, false, ArraySize, false, std::array<T, ArraySize>, MemberPointers...>
{
    return ParameterDescriptorWithDefaultT<T, S, false, ArraySize, false, std::array<T, ArraySize>, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::move(defaultValue),
        std::forward<const MemberPointers>(memberPointers)...);
}

/// Get described parameters for the passed node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeNodeParameters()
{
    return NodeT::describeParameters();
}

namespace detail
{

/// Populate parameter with default value (which is a no op for parameters which don't specify a default value).
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<!Param::HAS_DEFAULT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
bool populateDefault(const Param& param, MemberPtr& memberPtr)
{
    static_cast<void>(param);
    static_cast<void>(memberPtr);
    return false;
}

/// Populate parameter with default value.
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<Param::HAS_DEFAULT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
bool populateDefault(const Param& param, MemberPtr& memberPtr)
{
    memberPtr = param.defaultValue;
    return true;
}

/// Populate array parameter with default value (which is a no op for parameters which don't specify a default value).
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<!Param::HAS_DEFAULT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateArrayDefault(const Param& param, MemberPtr& memberPtr, size_t index)
{
    static_cast<void>(param);
    static_cast<void>(memberPtr);
    static_cast<void>(index);
}

/// Populate array parameter with default value.
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<Param::HAS_DEFAULT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateArrayDefault(const Param& param, MemberPtr& memberPtr, size_t index)
{
    memberPtr = param.defaultValue.at(index);
}

/// Terminate recursion to get the constructor argument where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<std::tuple_size<MemberPtrs>() == 0, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
auto getMemberPtr(ArgType& arg, MemberPtrs memberPtrs) -> ArgType&
{
    static_cast<void>(memberPtrs);
    return arg;
}

/// Terminate recursion to get the member pointer where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<MemberIndex + 1 == std::tuple_size<MemberPtrs>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto member = std::get<MemberIndex>(memberPtrs);
    return arg.*member;
}

/// Recursion to get the member pointer where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<MemberIndex + 1 < std::tuple_size<MemberPtrs>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto& member = std::get<MemberIndex>(memberPtrs);
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto& m = arg.*member;
    return getMemberPtr<MemberIndex + 1>(m, memberPtrs);
}

/// Get the number of parameters for a specific constructor argument.
template <typename NodeT, size_t ConstructorParameterIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t constructorArgumentParameterSize()
{
    using ConstructorArgumentDescriptor = std::tuple_element_t<ConstructorParameterIndex, decltype(describeNodeParameters<NodeT>())>;
    return dw::core::tuple_size<typename ConstructorArgumentDescriptor::ParameterDescriptorsType>::value;
}

/// Check if a parameter is an index.
template <typename ParamT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool isIndexParameter()
{
    return ParamT::IS_INDEX;
}

/// Check if a parameter is an array.
template <typename ParamT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool isArrayParameter()
{
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return ParamT::ARRAY_SIZE > 0U;
}

/// Check if a parameter is a abstract.
template <typename ParamT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool isAbstractParameter()
{
    return ParamT::IS_ABSTRACT;
}

/// Populate an abstract parameter which is a no op.
template <
    typename ArgType, typename ParamT,
    typename std::enable_if_t<isAbstractParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    static_cast<void>(provider);
    static_cast<void>(arg);
    static_cast<void>(param);
}

/**
 * Populate a non-array non-index parameter using value from the parameter provider.
 *
 * If the parameter value isn't available, fall back to use the default value if available.
 *
 * @throws Exception if retrieving a required parameter value fails
 */
template <
    typename ArgType, typename ParamT,
    typename std::enable_if_t<!isAbstractParameter<ParamT>() && !isArrayParameter<ParamT>() && !isIndexParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    using DataType         = typename ParamT::Type;
    using SemanticDataType = typename ParamT::SemanticType;
    DataType& memberPtr{getMemberPtr<0>(arg, param.memberPointers)};
    bool hadParameter{provider.getOptional<SemanticDataType, DataType>(param.parameterName, memberPtr)};
    if (!hadParameter)
    {
        bool setDefault{populateDefault(param, memberPtr)};
        if (!setDefault)
        {
            throw ExceptionWithStatus(DW_FAILURE, "Failed to get required parameter with name '", param.parameterName, "', T=", typeid(DataType).name(), ", S=", typeid(SemanticDataType).name());
        }
    }
}

/**
 * Populate a non-array index parameter using value from the parameter provider.
 *
 * If the parameter value or index within the array isn't available, fall back to use the default value if available.
 *
 * @throws Exception if retrieving a required parameter value fails
 */
template <
    typename ArgType, typename ParamT,
    typename std::enable_if_t<!isAbstractParameter<ParamT>() && !isArrayParameter<ParamT>() && isIndexParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    using DataType         = typename ParamT::Type;
    using SemanticDataType = typename ParamT::SemanticType;
    DataType& memberPtr{getMemberPtr<0>(arg, param.memberPointers)};

    size_t index{std::numeric_limits<size_t>::max()};
    bool hadParameter{provider.getOptional<size_t, size_t>(param.parameterName, index)};
    if (hadParameter)
    {
        hadParameter = provider.getOptionalWithIndex<SemanticDataType, DataType>("", index, memberPtr);
    }
    if (!hadParameter)
    {
        bool setDefault{populateDefault(param, memberPtr)};
        if (!setDefault)
        {
            throw ExceptionWithStatus(DW_FAILURE, "Failed to get required parameter with name '", param.parameterName, "', index=", index, ", T=", typeid(DataType).name(), ", S=", typeid(SemanticDataType).name());
        }
    }
}

/**
 * Populate an array parameter using value from the parameter provider.
 *
 * If the parameter value for each index within the array isn't available, fall back to use the default value if available.
 */
template <
    typename ArgType, typename ParamT,
    typename std::enable_if_t<!isAbstractParameter<ParamT>() && isArrayParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    using DataType         = typename ParamT::Type;
    using SemanticDataType = typename ParamT::SemanticType;

    constexpr size_t arraySize{ParamT::ARRAY_SIZE};

    DataType(&memberPtr)[arraySize]{getMemberPtr<0>(arg, param.memberPointers)};
    for (size_t i{0U}; i < arraySize; ++i)
    {
        bool hadParameter{provider.getOptionalWithIndex<SemanticDataType, DataType>(param.parameterName, i, memberPtr[i])};
        if (!hadParameter)
        {
            populateArrayDefault(param, memberPtr[i], i);
        }
    }
}

// LCOV_EXCL_START end of template recursion is never reached
/// Terminate template recursion for empty parameter type lists.
template <typename ArgType>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg, const dw::core::Tuple<>& t)
{
    static_cast<void>(provider);
    static_cast<void>(arg);
    static_cast<void>(t);
}
// LCOV_EXCL_STOP

/// Terminate template recursion to populate each constructor argument using values from the parameter provider.
template <typename ArgType, typename THead>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg, const dw::core::Tuple<THead>& t)
{
    populateParameter(provider, arg, t.m_head);
}

/// Template recursion to populate each constructor argument using values from the parameter provider.
template <
    typename ArgType, typename THead, typename... TTail,
    typename std::enable_if_t<sizeof...(TTail) != 0>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg, const dw::core::Tuple<THead, TTail...>& t)
{
    populateParameter(provider, arg, t.m_head);
    populateParametersRecursion(provider, arg, t.m_tail);
}

/**
 * Populate the constructor arguments using values from the parameter provider.
 *
 * @throws Exception if retrieving any parameter value fails
 */
template <typename NodeT, size_t ConstructorParameterIndex, typename ConstructorArgsType>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(const ParameterProvider& provider, ConstructorArgsType& constructorArguments)
{
    try
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto& arg = std::get<ConstructorParameterIndex>(constructorArguments);
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        constexpr auto params = std::get<ConstructorParameterIndex>(describeNodeParameters<NodeT>()).parameterDescriptors;
        // coverity[cert_err59_cpp_violation] RFD Pending: TID-2587
        populateParametersRecursion(provider, arg, params);
    }
    catch (ExceptionWithStatus& e)
    {
        throw ExceptionWithStatus(e.statusCode(), "Exception while populating parameters of mangled node type ", typeid(NodeT).name(), ": ", e.message());
    }
}

// Return value needed by populateParametersForEachConstructorArgument() for the forEach variable in C++14
template <typename NodeT, size_t ConstructorParameterIndex, typename ConstructorArgsType>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void* populateParametersWithNonVoidReturnType(const ParameterProvider& provider, ConstructorArgsType& constructorArguments)
{
    populateParameters<NodeT, ConstructorParameterIndex>(provider, constructorArguments);
    return nullptr;
}

template <
    typename NodeT, typename ConstructorArguments, size_t... Is>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParametersForEachConstructorArgument(const ParameterProvider& provider, ConstructorArguments& constructorArguments, std::integer_sequence<size_t, Is...>)
{
    void* forEach[]{
        populateParametersWithNonVoidReturnType<NodeT, Is>(provider, constructorArguments)...};
    static_cast<void>(forEach);
}

} // namespace detail

/// Populate the constructor arguments using values from the parameter provider.
template <typename NodeT, typename... Ts>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(std::tuple<Ts...>& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParametersForEachConstructorArgument<NodeT>(provider, constructorArguments, std::make_index_sequence<sizeof...(Ts)>());
}

namespace detail
{

// Get the Number of constructor arguments of the passed node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t parameterConstructorArgumentSize()
{
    return std::tuple_size<decltype(describeNodeParameters<NodeT>())>::value;
}

/// Terminate recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex == parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
std::tuple<> createConstructorArguments()
{
    return std::make_tuple();
}

/// Recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex<parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    auto createConstructorArguments()
{
    using NodeParams = decltype(describeNodeParameters<NodeT>());

    using ConstructorArgumentDescriptor = std::tuple_element_t<ConstructorArgumentIndex, NodeParams>;
    typename ConstructorArgumentDescriptor::ConstructorArgumentType arg{};

    return std::tuple_cat(std::make_tuple(arg), createConstructorArguments<NodeT, ConstructorArgumentIndex + 1>());
}

} // namespace detail

/// Create a tuple of constructor argument needed by the constructor of the passed node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
auto createConstructorArguments()
{
    return detail::createConstructorArguments<NodeT, 0>();
}

namespace detail
{

/// Instantiate a node using the passed tuple of constructor arguments.
template <class T, class Tuple, size_t... Is>
// coverity[autosar_cpp14_a2_10_5_violation]
auto makeUniqueFromTuple(Tuple&& tuple, std::index_sequence<Is...>) -> std::unique_ptr<T>
{
    return std::make_unique<T>(std::get<Is>(tuple)...);
}
}

/// Instantiate a node using the passed constructor arguments.
template <typename NodeT, class ConstructorArguments>
// coverity[autosar_cpp14_a2_10_5_violation]
auto makeUniqueFromTuple(ConstructorArguments&& constructorArguments) -> std::unique_ptr<NodeT>
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
auto createNode(const ParameterProvider& provider) -> std::unique_ptr<NodeT>
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t parameterSize()
{
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 0U;
}

/// Recursion to count the number of parameters for each constructor argument of a given node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex<parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t parameterSize()
{
    return constructorArgumentParameterSize<NodeT, ConstructorArgumentIndex>() + parameterSize<NodeT, ConstructorArgumentIndex + 1>();
}

} // namespace detail

/// Get the number of parameters for a given node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr std::size_t parameterSize()
{
    return detail::parameterSize<NodeT, 0>();
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PARAMETERDESCRIPTOR_HPP_
