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

// Indices within the tuple describing constructor arguments
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr const size_t PARAMETER_CONSTRUCTOR_ARGUMENT_TYPE{0U};
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr const size_t PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR{1U};

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
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2) -> std::tuple<std::tuple<Argument1T*, Arg1>, std::tuple<Argument2T*, Arg2>>
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
 * Describe the constructor arguments of a node.
 *
 * The function is used to implement NodeConcept::describeParameters().
 * Each argument of the function is created by describeConstructorArgument().
 */
template <
    typename Argument1T, typename Argument2T, typename Argument3T, typename Argument4T, typename Argument5T, typename Argument6T,
    typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describeConstructorArguments(const Arg1&& arg1, const Arg2&& arg2, const Arg3&& arg3, const Arg4&& arg4, const Arg5&& arg5, const Arg6&& arg6)
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
            std::forward<const Arg5>(arg5)),
        std::make_tuple(
            static_cast<Argument6T*>(nullptr),
            std::forward<const Arg6>(arg6)));
}

/**
 * Describe a specific constructor argument of a node.
 *
 * The function is used to create the arguments for describeConstructorArguments().
 * Each argument of the function is created by one of the describe parameter functions below.
 */
template <typename... Args>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeConstructorArgument(const Args&&... args) -> dw::core::Tuple<Args...>
{
    return dw::core::make_tuple(
        std::forward<const Args>(args)...);
}

// coverity[autosar_cpp14_a14_1_1_violation]
template <typename Type_, typename SemanticType_, bool IsIndex, size_t ArraySize, bool IsAbstract, bool HasDefault, typename... MemberPointers>
struct ParameterDescriptorT
{
    dw::core::StringView typeName;
    dw::core::StringView parameterName;
    // coverity[autosar_cpp14_a0_1_6_violation]
    using Type = Type_;
    // coverity[autosar_cpp14_a0_1_6_violation]
    using SemanticType = SemanticType_;
    static constexpr bool IS_INDEX{IsIndex};
    static constexpr size_t ARRAY_SIZE{ArraySize};
    static constexpr bool IS_ABSTRACT{IsAbstract};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr bool HAS_DEFAULT{HasDefault};
    std::tuple<const MemberPointers...> memberPointers;

    constexpr ParameterDescriptorT(dw::core::StringView const&& typeName_, dw::core::StringView const&& parameterName_, const MemberPointers&&... memberPointers_)
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

// coverity[autosar_cpp14_a14_1_1_violation]
template <typename Type, typename SemanticType, bool IsIndex, size_t ArraySize, bool IsAbstract, typename DefaultType, typename... MemberPointers>
struct ParameterDescriptorWithDefaultT : public ParameterDescriptorT<Type, SemanticType, IsIndex, ArraySize, IsAbstract, true, MemberPointers...>
{
    DefaultType defaultValue;

    constexpr ParameterDescriptorWithDefaultT(dw::core::StringView const&& typeName_, dw::core::StringView const&& parameterName_, const DefaultType&& defaultValue_, const MemberPointers&&... memberPointers_)
        : ParameterDescriptorT<Type, SemanticType, IsIndex, ArraySize, IsAbstract, true, MemberPointers...>{std::move(typeName_), std::move(parameterName_), std::forward<const MemberPointers>(memberPointers_)...}
        , defaultValue{std::move(defaultValue_)}
    {
    }
};

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, const MemberPointers&&... memberPointers) -> ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>
{
    return ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ABSTRACT_PARAMETER(TYPE_NAME, args...) dw::framework::describeAbstractParameter<TYPE_NAME>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeAbstractArrayParameter<TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), PARAM_NAME, ##args)

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
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName)
{
    return ParameterDescriptorT<T, T, false, ArraySize, true, false>(
        std::move(typeName),
        std::move(parameterName));
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
    return ParameterDescriptorT<T, S, true, 0, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_ARRAY_PARAMETER(TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeArrayParameter<TYPE_NAME, TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), PARAM_NAME, ##args)
#define DW_DESCRIBE_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, PARAM_NAME, ARRAY_SIZE, args...) dw::framework::describeArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), PARAM_NAME, ##args)

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
constexpr auto describeArrayParameter(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, const MemberPointers&&... memberPointers)
{
    return ParameterDescriptorT<T, S, false, ArraySize, false, false, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::forward<const MemberPointers>(memberPointers)...);
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeUnnamedParameter(
    dw::core::StringView const&& typeName, const MemberPointers&&... memberPointers) -> ParameterDescriptorT<T, S, false, 0, false, false, MemberPointers...>
{
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_a5_2_2_violation]
    return describeParameter<T, S, MemberPointers...>(
        std::move(typeName),
        // coverity[autosar_cpp14_a5_1_1_violation]
        ""_sv,
        std::forward<const MemberPointers>(memberPointers)...);
}

} // namespace framework
} // namespace dw

#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER(TYPE_NAME, ARRAY_SIZE, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)
#define DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE, args...) dw::framework::describeUnnamedArrayParameter<TYPE_NAME, SEMANTIC_TYPE_NAME, ARRAY_SIZE>(DW_PARAMETER_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

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
    dw::core::StringView const&& typeName, const MemberPointers&&... memberPointers)
{
    return describeArrayParameter<T, S, ArraySize>(
        std::move(typeName),
        ""_sv,
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describeParameterWithDefault(
    dw::core::StringView const&& typeName, dw::core::StringView const&& parameterName, T defaultValue, const MemberPointers&&... memberPointers) -> ParameterDescriptorWithDefaultT<T, S, false, 0, false, T, MemberPointers...>
{
    return ParameterDescriptorWithDefaultT<T, S, false, 0, false, T, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::move(defaultValue),
        std::forward<const MemberPointers>(memberPointers)...);
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
    return ParameterDescriptorWithDefaultT<T, S, false, ArraySize, false, std::array<T, ArraySize>, MemberPointers...>(
        std::move(typeName),
        std::move(parameterName),
        std::move(defaultValue),
        std::forward<const MemberPointers>(memberPointers)...);
}

/// Get described parameters for the passed node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describeParameters()
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
void populateDefault(const Param& param, MemberPtr& memberPtr)
{
    static_cast<void>(param);
    static_cast<void>(memberPtr);
}

/// Populate parameter with default value.
template <
    typename Param, typename MemberPtr,
    typename std::enable_if_t<Param::HAS_DEFAULT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateDefault(const Param& param, MemberPtr& memberPtr)
{
    memberPtr = param.defaultValue;
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
    memberPtr[index] = param.defaultValue[index];
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
// coverity[autosar_cpp14_a7_1_5_violation]
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    // coverity[autosar_cpp14_a8_5_2_violation]
    auto member = std::get<MemberIndex>(memberPtrs);
    return arg.*member;
}

/// Recursion to get the member pointer where a specific parameter value should be stored in.
template <
    size_t MemberIndex, typename ArgType, typename MemberPtrs,
    typename std::enable_if_t<MemberIndex + 1 < std::tuple_size<MemberPtrs>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
auto& getMemberPtr(ArgType& arg, MemberPtrs memberPtrs)
{
    auto& member = std::get<MemberIndex>(memberPtrs);
    auto& m      = arg.*member;
    return getMemberPtr<MemberIndex + 1>(m, memberPtrs);
}

/// Get the number of parameters for a specific constructor argument.
template <typename NodeT, size_t ConstructorParameterIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t constructorArgumentParameterSize()
{
    // coverity[autosar_cpp14_a13_5_3_violation]
    return dw::core::tuple_size<
        std::tuple_element_t<
            PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR,
            std::tuple_element_t<
                ConstructorParameterIndex,
                decltype(describeParameters<NodeT>())>>>();
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
 * Populate a non-array non-index pararmeter using value from the parameter provider.
 *
 * If the parameter value isn't available, fall back to use the default value if available.
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
    bool hadParameter{provider.getOptional<SemanticDataType, DataType>(param.parameterName, &memberPtr)};
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
    typename ArgType, typename ParamT,
    typename std::enable_if_t<!isAbstractParameter<ParamT>() && !isArrayParameter<ParamT>() && isIndexParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    using DataType         = typename ParamT::Type;
    using SemanticDataType = typename ParamT::SemanticType;
    DataType& memberPtr    = getMemberPtr<0>(arg, param.memberPointers);

    size_t index      = static_cast<size_t>(-1);
    bool hadParameter = provider.getOptional<size_t, size_t>(param.parameterName, &index);
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
    typename ArgType, typename ParamT,
    typename std::enable_if_t<!isAbstractParameter<ParamT>() && isArrayParameter<ParamT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameter(const ParameterProvider& provider, ArgType& arg, const ParamT& param)
{
    using DataType         = typename ParamT::Type;
    using SemanticDataType = typename ParamT::SemanticType;

    constexpr size_t arraySize = ParamT::ARRAY_SIZE;

    DataType(&memberPtr)[arraySize] = getMemberPtr<0>(arg, param.memberPointers);
    for (size_t i = 0; i < arraySize; ++i)
    {
        bool hadParameter = provider.getOptional<SemanticDataType, DataType>(param.parameterName, i, &memberPtr[i]);
        if (!hadParameter)
        {
            populateArrayDefault(param, memberPtr, i);
        }
    }
}

/// Terminate template recursion for empty parameter type lists.
template <typename ArgType>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParametersRecursion(const ParameterProvider& provider, ArgType& arg, const dw::core::Tuple<>& t)
{
    static_cast<void>(provider);
    static_cast<void>(arg);
    static_cast<void>(t);
}

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
// TODO(dwplc): FP -- The other populateParameters() functions are defined in the parent namespace
// coverity[autosar_cpp14_a2_10_5_violation]
void populateParameters(const ParameterProvider& provider, ConstructorArgsType& constructorArguments)
{
    try
    {
        // coverity[autosar_cpp14_a8_5_2_violation]
        auto& arg = std::get<ConstructorParameterIndex>(constructorArguments);
        // coverity[autosar_cpp14_a8_5_2_violation]
        constexpr auto params = std::get<dw::framework::PARAMETER_CONSTRUCTOR_ARGUMENT_DESCRIPTOR>(
            std::get<ConstructorParameterIndex>(describeParameters<NodeT>()));
        // coverity[cert_err59_cpp_violation]
        populateParametersRecursion(provider, arg, params);
    }
    catch (ExceptionWithStatus& e)
    {
        throw ExceptionWithStatus(e.statusCode(), "Exception while populating parameters of mangled node type ", typeid(NodeT).name(), ": ", e.message());
    }
}

} // namespace detail

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 1, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 2, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, constructorArguments);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 3, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, constructorArguments);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 4, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, constructorArguments);
    detail::populateParameters<NodeT, 3>(provider, constructorArguments);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 5, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, constructorArguments);
    detail::populateParameters<NodeT, 3>(provider, constructorArguments);
    detail::populateParameters<NodeT, 4>(provider, constructorArguments);
}

/// Populate the constructor arguments using values from the parameter provider.
template <
    typename NodeT,
    class ConstructorArguments,
    typename std::enable_if_t<std::tuple_size<ConstructorArguments>() == 6, void>* = nullptr>
// Output parameter is needed to populate member of arbitrary struct
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void populateParameters(ConstructorArguments& constructorArguments, const ParameterProvider& provider)
{
    detail::populateParameters<NodeT, 0>(provider, constructorArguments);
    detail::populateParameters<NodeT, 1>(provider, constructorArguments);
    detail::populateParameters<NodeT, 2>(provider, constructorArguments);
    detail::populateParameters<NodeT, 3>(provider, constructorArguments);
    detail::populateParameters<NodeT, 4>(provider, constructorArguments);
    detail::populateParameters<NodeT, 5>(provider, constructorArguments);
}

namespace detail
{

// Get the Number of constructor arguments of the passed node.
template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t parameterConstructorArgumentSize()
{
    return std::tuple_size<decltype(describeParameters<NodeT>())>::value;
}

/// Terminate recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex == parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation]
auto createConstructorArguments()
{
    return std::make_tuple();
}

/// Recursion to create a tuple of constructor argument needed by the constructor of the passed node.
template <
    typename NodeT, size_t ConstructorArgumentIndex,
    typename std::enable_if_t<ConstructorArgumentIndex<parameterConstructorArgumentSize<NodeT>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    // coverity[autosar_cpp14_a7_1_5_violation]
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation]
auto createConstructorArguments()
{
    return detail::createConstructorArguments<NodeT, 0>();
}

namespace detail
{

/// Instantiate a node using the passed tuple of constructor arguments.
template <class T, class Tuple, size_t... Is>
// coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
auto makeUniqueFromTuple(const Tuple&& tuple, std::index_sequence<Is...>) -> std::unique_ptr<T>
{
    // coverity[use_after_move]
    return std::make_unique<T>(std::get<Is>(std::move(tuple))...);
}
}

/// Instantiate a node using the passed constructor arguments.
template <typename NodeT, class ConstructorArguments>
// coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
auto create(const ParameterProvider& provider) -> std::unique_ptr<NodeT>
{
    // coverity[autosar_cpp14_a8_5_2_violation]
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
// coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
constexpr std::size_t parameterSize()
{
    return detail::parameterSize<NodeT, 0>();
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PARAMETERDESCRIPTOR_HPP_
