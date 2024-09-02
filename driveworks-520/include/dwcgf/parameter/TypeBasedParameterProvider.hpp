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

#ifndef DW_FRAMEWORK_TYPEBASEDPARAMETERPROVIDER_HPP_
#define DW_FRAMEWORK_TYPEBASEDPARAMETERPROVIDER_HPP_

#include <dwcgf/parameter/ParameterProvider.hpp>

#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwcgf/Exception.hpp>

#include <typeindex>
#include <utility>

namespace dw
{
namespace framework
{

class TypeBasedParameterProvider;

/// The additional interface a ParameterProvider can implement so it can be registered at a TypeBasedParameterProvider.
class ITypeBasedParameterProviderChild
{
protected:
    /// Copy constructor.
    ITypeBasedParameterProviderChild(ITypeBasedParameterProviderChild const&) = default;
    /// Move constructor.
    ITypeBasedParameterProviderChild(ITypeBasedParameterProviderChild&&) = default;
    /// Copy assignment operator.
    ITypeBasedParameterProviderChild& operator=(ITypeBasedParameterProviderChild const&) & = default;
    /// Move assignment operator.
    ITypeBasedParameterProviderChild& operator=(ITypeBasedParameterProviderChild&&) & = default;

public:
    /// Default constructor.
    ITypeBasedParameterProviderChild() = default;
    /// Destructor.
    virtual ~ITypeBasedParameterProviderChild() = default;
    /// Register handlers implemented in this parameter provider at the passed type base parameter provider.
    virtual void registerAt(TypeBasedParameterProvider& provider) const = 0;
};

/// A parameter provider which dispatches the retrieval of the parameter value to registered handlers which are selected based on the name, size, and optional array index of the parameter.
class TypeBasedParameterProvider : public ParameterProvider
{
protected:
    /// Copy constructor.
    TypeBasedParameterProvider(TypeBasedParameterProvider const&) = default;
    /// Move constructor.
    TypeBasedParameterProvider(TypeBasedParameterProvider&&) = default;
    /// Copy assignment operator.
    TypeBasedParameterProvider& operator=(TypeBasedParameterProvider const&) & = default;
    /// Move assignment operator.
    TypeBasedParameterProvider& operator=(TypeBasedParameterProvider&&) & = default;

public:
    /// Default constructor.
    TypeBasedParameterProvider() = default;
    /// Destructor.
    ~TypeBasedParameterProvider() override = default;

    /// @see dw::framework::ParameterProvider::getImpl(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, void*)
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    bool getImpl(
        ParameterProvider const* const parentProvider,
        dw::core::StringView const& key,
        const std::type_info& semanticTypeInfo,
        const std::type_info& dataTypeInfo,
        void* out) const override;

    /// @see dw::framework::ParameterProvider::getImplWithIndex(ParameterProvider const* const, dw::core::StringView const&, size_t const, const std::type_info&, const std::type_info&, void*)
    bool getImplWithIndex(
        ParameterProvider const* const parentProvider,
        dw::core::StringView const& key,
        size_t const index,
        const std::type_info& semanticTypeInfo,
        const std::type_info& dataTypeInfo,
        void* out) const override;

    /// @name Parameter providers selected by data type.
    /// @{
    /// Register the passed parameter provider for the template type T.
    template <typename T>
    void registerProvider(const ParameterProvider* provider)
    {
        // coverity[cert_exp60_cpp_violation] FP: nvbugs/3163193
        registerProvider(typeid(T), provider);
    }

    /// Get the provider for the template type T.
    template <typename T>
    const ParameterProvider* getProvider() const
    {
        return getProvider(typeid(T));
    }

protected:
    /// Register the parameter provider.
    void registerProvider(const std::type_info& typeInfo, const ParameterProvider* provider);

    /// Get the parameter provider.
    const ParameterProvider* getProvider(const std::type_info& typeInfo) const;

private:
    /// Registered parameter providers identified by the data type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<const std::type_info&, const ParameterProvider*>, 256> m_providers;
    /// @}

public:
    /// @name Parameter provider children selected by data type.
    /// @{
    /// Register the passed parameter provider child for the template type T.
    template <typename T>
    void registerProviderChild(const T& providerChild)
    {
        // coverity[cert_exp60_cpp_violation] FP: nvbugs/3163193
        registerProviderChild(typeid(T), providerChild);
    }

    /// Get the provider child for the template type T.
    template <typename T>
    const ITypeBasedParameterProviderChild* getProviderChild() const
    {
        return getProviderChild(typeid(T));
    }

    /// Register the parameter provider child .
    void registerProviderChild(const std::type_info& typeInfo, const ITypeBasedParameterProviderChild& providerChild);

protected:
    /// Get the parameter provider child.
    const ITypeBasedParameterProviderChild* getProviderChild(const std::type_info& typeInfo) const;

private:
    /// Registered parameter provider children identified by the data type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<const std::type_info&, const ITypeBasedParameterProviderChild&>, 256> m_providerChildren;
    /// @}

public:
    /// @name Handler functions retrieving parameters by name.
    /// @{
    /// Function signature of this handler type.
    using ParameterByNameHandler = std::function<bool(dw::core::StringView const&, void*)>;

    /// Register the handler function for the template type T.
    template <typename T>
    void registerByNameHandler(ParameterByNameHandler&& handler)
    {
        registerByNameHandler<T, T>(std::move(handler));
    }

    /// Register the handler function for the template type T and the semantic template type S.
    template <typename T, typename S>
    void registerByNameHandler(ParameterByNameHandler&& handler)
    {
        registerByNameHandler(typeid(T), typeid(S), std::move(handler));
    }

protected:
    /// Register the handler function.
    void registerByNameHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo, ParameterByNameHandler&& handler);

    /// Get the handler function.
    ParameterByNameHandler const* getByNameHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo) const;

private:
    /// Registered handler functions identified by the data type and semantic type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<std::pair<const std::type_info&, const std::type_info&>, ParameterByNameHandler>, 256> m_byNameHandlers;
    /// @}

public:
    /// @name Handler functions retrieving parameters by data type, name and index.
    /// @{
    /// Function signature of this handler type.
    using ParameterByNameAndIndexHandler = std::function<bool(dw::core::StringView const&, size_t, void*)>;

    /// Register the handler function for the template type T.
    template <typename T>
    void registerByNameAndIndexHandler(ParameterByNameAndIndexHandler&& handler)
    {
        registerByNameAndIndexHandler<T, T>(std::move(handler));
    }

    /// Register the handler function for the template type T and the semantic template type S.
    template <typename T, typename S>
    void registerByNameAndIndexHandler(ParameterByNameAndIndexHandler&& handler)
    {
        registerByNameAndIndexHandler(typeid(T), typeid(S), std::move(handler));
    }

protected:
    /// Register the handler function.
    void registerByNameAndIndexHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo, ParameterByNameAndIndexHandler&& handler);

    /// Get the handler function.
    ParameterByNameAndIndexHandler const* getByNameAndIndexHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo) const;

private:
    /// Registered handler functions identified by the data type and semantic type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<std::pair<const std::type_info&, const std::type_info&>, ParameterByNameAndIndexHandler>, 256> m_byNameAndIndexHandlers;
    /// @}

public:
    /// @name Handler functions retrieving parameters by data type and index.
    /// @{
    /// Function signature of this handler type.
    using ParameterByIndexHandler = std::function<bool(size_t, void*)>;

    /// Register the handler function for the template type T.
    template <typename T>
    void registerByIndexHandler(ParameterByIndexHandler&& handler)
    {
        registerByIndexHandler<T, T>(std::move(handler));
    }

    /// Register the handler function for the template type T and the semantic template type S.
    template <typename T, typename S>
    void registerByIndexHandler(ParameterByIndexHandler&& handler)
    {
        registerByIndexHandler(typeid(T), typeid(S), std::move(handler));
    }

protected:
    /// Register the handler function.
    void registerByIndexHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo, ParameterByIndexHandler&& handler);

    /// Get the handler function.
    ParameterByIndexHandler const* getByIndexHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo) const;

private:
    /// Registered handler functions identified by the data type and semantic type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<std::pair<const std::type_info&, const std::type_info&>, ParameterByIndexHandler>, 256> m_byIndexHandlers;
    /// @}

public:
    /// @name Handler functions retrieving parameters by data type.
    /// @{
    /// Function signature of this handler type.
    using ParameterByTypeHandler = std::function<bool(void*)>;

    /// Register the handler function for the template type T.
    template <typename T>
    void registerByTypeHandler(ParameterByTypeHandler&& handler)
    {
        registerByTypeHandler<T, T>(std::move(handler));
    }

    /// Register the handler function for the template type T and the semantic template type S.
    template <typename T, typename S>
    void registerByTypeHandler(ParameterByTypeHandler&& handler)
    {
        registerByTypeHandler(typeid(T), typeid(S), std::move(handler));
    }

protected:
    /// Register the handler function.
    void registerByTypeHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo, ParameterByTypeHandler&& handler);

    /// Get the handler function.
    ParameterByTypeHandler const* getByTypeHandler(const std::type_info& dataTypeInfo, const std::type_info& semanticTypeInfo) const;

private:
    /// Registered handler functions identified by the data type and semantic type.
    // Can't use std::type_index since on QNX two type_info/index of the same type
    // might not compare equal to each other / have different hashes
    dw::core::VectorFixed<std::pair<std::pair<const std::type_info&, const std::type_info&>, ParameterByTypeHandler>, 256> m_byTypeHandlers;
    /// @}

protected:
    /**
     * Check if a two pairs of type infos is equal.
     *
     * Due to a limitation/bug on QNX, two type_info for the same type might not compare equal to each other.
     * This function is a WAR and falls back to a string comparison of .name().
     */
    static bool isPairEqual(const std::pair<const std::type_info&, const std::type_info&>& lhs, const std::pair<const std::type_info&, const std::type_info&>& rhs);
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_TYPEBASEDPARAMETERPROVIDER_HPP_
