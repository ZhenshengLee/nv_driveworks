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

#ifndef DW_FRAMEWORK_PARAMETERPROVIDER_HPP_
#define DW_FRAMEWORK_PARAMETERPROVIDER_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>

#include <dwcgf/Exception.hpp>

#include <dw/core/container/StringView.hpp>

#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

namespace dw
{
namespace framework
{

/// The interface to access parameter values identified by name and/or (semantic) type.
class ParameterProvider
{
protected:
    /// Copy constructor.
    ParameterProvider(ParameterProvider const&) = default;
    /// Move constructor.
    ParameterProvider(ParameterProvider&&) = default;
    /// Copy assignment operator.
    ParameterProvider& operator=(ParameterProvider const&) & = default;
    /// Move assignment operator.
    ParameterProvider& operator=(ParameterProvider&&) & = default;

public:
    /// Default constructor.
    ParameterProvider() = default;
    /// Destructor.
    virtual ~ParameterProvider() = default;

    /**
     * Convenience API throwing an exception instead of returning false.
     *
     * @see getOptional(dw::core::StringView const&, T*)
     * @throws Exception if the parameter is not successfully retrieved
     */
    template <typename T>
    void getRequired(dw::core::StringView const& key, T* out) const
    {
        if (!getOptional(key, out))
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "Required parameter not available: ", key);
        }
    }

    /**
     * Convenience API throwing an exception instead of returning false.
     *
     * @see getOptional(dw::core::StringView const&, size_t const, T*)
     * @throws Exception if the parameter is not successfully retrieved
     */
    template <typename T>
    void getRequired(dw::core::StringView const& key, size_t const index, T* out) const
    {
        if (!getOptional(key, index, out))
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "Required parameter not available: ", key);
        }
    }

    /**
     * Convenience API throwing an exception instead of returning a failed status.
     *
     * @see get(dw::core::StringView const&, T*)
     * @throws Exception if the parameter retrieval has a failure
     */
    template <typename T>
    bool getOptional(dw::core::StringView const& key, T* out) const
    {
        try
        {
            return get(key, out);
        }
        catch (Exception& e)
        {
            throw ExceptionWithStatus(e.status(), "Failed to get parameter by name: ", key, " - ", e.message());
        }
    }

    /**
     * Convenience API throwing an exception instead of returning a failed status.
     *
     * @see get(dw::core::StringView const&, size_t const, T*)
     * @throws Exception if the parameter retrieval has a failure
     */
    template <typename T>
    bool getOptional(dw::core::StringView const& key, size_t const index, T* out) const
    {
        try
        {
            return get(key, index, out);
        }
        catch (Exception& e)
        {
            throw ExceptionWithStatus(e.status(), "Failed to get parameter by name and index: ", key, "[", index, "] - ", e.message());
        }
    }

    /**
     * Get a non-array non-enum parameter value.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            !std::is_array<T>::value &&
            !std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        return get(this, key, typeid(T), typeid(T), out);
    }

    /**
     * Get a single non-enum parameter value from an array parameter by index.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, size_t const, T*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            !std::is_array<T>::value &&
            !std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, size_t const index, T* out) const
    {
        return get(this, key, index, typeid(T), typeid(T), out);
    }

    /**
     * Get a vector of non-enum parameter values.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            !std::is_array<T>::value &&
            !std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, std::vector<T>* out) const
    {
        return get(this, key, typeid(std::vector<T>), typeid(std::vector<T>), out);
    }

    /**
     * Get a fixed-size 1 dimension array of non-enum parameter values.
     *
     * @see get(ParameterProvider const* const, std::vector<T>*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            std::is_array<T>::value &&
            std::rank<T>::value == 1 &&
            !std::is_enum<std::remove_extent_t<T>>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        static_assert(std::extent<T>::value > 0, "Array must have size greater zero");
        using TElement = typename std::remove_extent_t<T>;
        std::vector<TElement> value;
        if (!get(key, &value))
        {
            return false;
        }

        constexpr size_t size = sizeof(T) / sizeof(TElement);
        if (value.size() != size)
        {
            throw ExceptionWithStatus(DW_FAILURE, "Array sizes don't match");
        }

        TElement* element = &(*out[0]);
        for (size_t i = 0; i < size; ++i)
        {
            *(element + i) = value[i];
        }
        return true;
    }

    /**
     * Get a fixed-size 1 dimension array of non-enum parameter values of type char8_t.
     *
     * @see get(ParameterProvider const* const, std::vector<T>*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            std::is_array<T>::value &&
            std::rank<T>::value == 1 &&
            std::is_same<T, char8_t>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        static_assert(std::extent<T>::value > 0, "Array must have size greater zero");
        std::string value;
        if (!get(key, &value))
        {
            return false;
        }

        if (value.size() >= std::extent<T, 0>::value)
        {
            throw ExceptionWithStatus(DW_FAILURE, "Array sizes don't match");
        }

        out[0] = '\n';
        strncat(out, value.c_str(), value.size());
        return true;
    }

    /**
     * Get a fixed-size 2 dimension array of non-enum parameter values.
     *
     * @see get(ParameterProvider const* const, std::vector<T>*)
     */
    template <
        typename T,
        typename std::enable_if_t<
            std::is_array<T>::value && std::rank<T>::value == 2 &&
            !std::is_enum<std::remove_all_extents_t<T>>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        static_assert(std::extent<T, 0>::value > 0, "Array must have 1st dimension size greater zero");
        static_assert(std::extent<T, 1>::value > 0, "Array must have 2nd dimension size greater zero");
        using TElement = typename std::remove_all_extents_t<T>;
        std::vector<TElement> value;
        if (!get(key, &value))
        {
            return false;
        }

        constexpr size_t size = sizeof(T) / sizeof(TElement);
        if (value.size() != size)
        {
            throw ExceptionWithStatus(DW_FAILURE, "Array sizes don't match");
        }

        TElement* element = &(out[0][0]);
        for (size_t i = 0; i < size; ++i)
        {
            *(element + i) = value[i];
        }
        return true;
    }

    /**
     * Convenience API throwing an exception instead of returning false.
     *
     * @see getOptional(dw::core::StringView const&, T*)
     * @throws Exception if the parameter is not successfully retrieved
     */
    template <typename S, typename T>
    void getRequired(dw::core::StringView const& key, T* out) const
    {
        if (!getOptional<S, T>(key, out))
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "Required parameter not available: ", key);
        }
    }

    /**
     * Convenience API throwing an exception instead of returning false.
     *
     * @see getOptional(dw::core::StringView const&, size_t const, T*)
     * @throws Exception if the parameter is not successfully retrieved
     */
    template <typename S, typename T>
    void getRequired(dw::core::StringView const& key, size_t const index, T* out) const
    {
        if (!getOptional<S, T>(key, index, out))
        {
            throw ExceptionWithStatus(DW_NOT_AVAILABLE, "Required parameter not available: ", key);
        }
    }

    /**
     * Convenience API throwing an exception instead of returning a failed status.
     *
     * @see get(dw::core::StringView const&, T*)
     * @throws Exception if the parameter retrieval has a failure
     */
    template <typename S, typename T>
    bool getOptional(dw::core::StringView const& key, T* out) const
    {
        try
        {
            return get<S, T>(key, out);
        }
        catch (Exception& e)
        {
            if (key.empty())
            {
                throw ExceptionWithStatus(e.status(), "Failed to get unnamed parameter with mangled semantic type: ", typeid(S).name(), " - ", e.message());
            }
            else
            {
                throw ExceptionWithStatus(e.status(), "Failed to get parameter with semantic by name: ", key, " - ", e.message());
            }
        }
    }

    /**
     * Convenience API throwing an exception instead of returning a failed status.
     *
     * @see get(dw::core::StringView const&, size_t const, T*)
     * @throws Exception if the parameter retrieval has a failure
     */
    template <typename S, typename T>
    bool getOptional(dw::core::StringView const& key, size_t const index, T* out) const
    {
        try
        {
            return get<S, T>(key, index, out);
        }
        catch (Exception& e)
        {
            if (key.empty())
            {
                throw ExceptionWithStatus(e.status(), "Failed to get unnamed parameter with mangled semantic type and index: ", typeid(S).name(), " ", index, " - ", e.message());
            }
            else
            {
                throw ExceptionWithStatus(e.status(), "Failed to get parameter with semantic by name and index: ", key, " ", index, " - ", e.message());
            }
        }
    }

    /**
     * Get a non-enum parameter value with a semantic type.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T,
        std::enable_if_t<!std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        static_assert(!std::is_same<T, dw::core::StringView>::value, "T shouldn't be a dw::core::StringView, use FixedString<N> instead");

        static_assert(!std::is_same<T, std::string>::value, "T shouldn't be a std::string, use FixedString<N> instead");

        // as long as the parameter provider makes sure that the const char* is valid throughout the life time of the parameter struct this is fine
        // atm this is only used by custom parameter providers which provides values from a static singleton
        // static_assert(!std::is_same<T, const char*>::value, "T shouldn't be a C-style string, use FixedString<N> instead");

        return get(this, key, typeid(S), typeid(T), out);
    }

    /**
     * Get a FixedString parameter value with a semantic type.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T, size_t N,
        std::enable_if_t<std::is_same<T, dw::core::FixedString<N>>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, dw::core::FixedString<N>* out) const
    {
        dw::core::StringView str;
        const auto& semanticTypeInfo = std::is_same<S, dw::core::FixedString<N>>::value ? typeid(dw::core::StringView) : typeid(S);
        bool success                 = get(this, key, semanticTypeInfo, typeid(dw::core::StringView), &str);
        if (success)
        {
            if (N <= str.size())
            {
                throw ExceptionWithStatus(DW_BUFFER_FULL, "The FixedString parameter '", key, "' has a maximum capacity of N=", N, " but the value has a length of ", str.size() + 1, "(including trailing \\0)");
            }
            out->copyFrom(str.data(), str.size());
        }
        return success;
    }

    /**
     * Get an enum parameter value with a semantic type.
     *
     * The parameter value retrieved is of type string and is being mapped to the enum type output parameter.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T,
        std::enable_if_t<std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, T* out) const
    {
        // get enum parameter from semantic parameter when key is empty
        if (key.empty())
        {
            return get(this, key, typeid(S), typeid(T), out);
        }

        dw::core::StringView str;
        if (!get(this, key, typeid(dw::core::StringView), typeid(dw::core::StringView), &str))
        {
            return false;
        }
        try
        {
            *out = mapEnumNameToValue<T>(str);
            return true;
        }
        catch (Exception& e)
        {
            throw ExceptionWithStatus(e.status(), "Failed to map enum name '", str, "' for parameter '", key, "' to numeric value: ", e.message());
        }
    }

    /**
     * Get a non-enum parameter value with a semantic type from an array parameter by index.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, size_t const, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T,
        std::enable_if_t<!std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, size_t const index, T* out) const
    {
        return get(this, key, index, typeid(S), typeid(T), out);
    }

    /**
     * Get a FixedString parameter value with a semantic type from an array parameter by index.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, size_t const, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T, size_t N,
        std::enable_if_t<std::is_same<T, dw::core::FixedString<N>>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, size_t const index, dw::core::FixedString<N>* out) const
    {
        dw::core::StringView str;
        const auto& semanticTypeInfo = std::is_same<S, dw::core::FixedString<N>>::value ? typeid(dw::core::StringView) : typeid(S);
        bool success                 = get(this, key, index, semanticTypeInfo, typeid(dw::core::StringView), &str);
        if (success)
        {
            if (N <= str.size())
            {
                throw ExceptionWithStatus(DW_BUFFER_FULL, "The FixedString parameter '", key, "' and index ", index, " has a maximum capacity of N=", N, " but the value has a length of ", str.size() + 1, "(including trailing \\0)");
            }
            out->copyFrom(str.data(), str.size());
        }
        return success;
    }

    /**
     * Get an enum parameter value with a semantic type from an array parameter by index.
     *
     * The parameter value retrieved is of type string and is being mapped to the enum type output parameter.
     *
     * @see get(ParameterProvider const* const, dw::core::StringView const&, size_t const, const std::type_info&, const std::type_info&, T*)
     */
    template <
        typename S, typename T,
        std::enable_if_t<std::is_enum<T>::value>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    bool get(dw::core::StringView const& key, size_t const index, T* out) const
    {
        // get enum parameter from semantic parameter when key is empty
        if (key.empty())
        {
            return get(this, key, index, typeid(S), typeid(T), out);
        }

        dw::core::StringView str;
        if (!get(this, key, index, typeid(dw::core::StringView), typeid(dw::core::StringView), &str))
        {
            return false;
        }
        try
        {
            *out = mapEnumNameToValue<T>(str);
            return true;
        }
        catch (Exception& e)
        {
            throw ExceptionWithStatus(e.status(), "Failed to map enum name '", str, "' for parameter '", key, "' and index ", index, " to numeric value: ", e.message());
        }
    }

    /**
     * Get a parameter.
     *
     * @param[in] parentProvider The parent provider in case the lookup goes through a hierarchy of providers, otherwise nullptr
     * @param[in] key The parameter name, or an empty string if the (semantic) type sufficiently identifies the parameter
     * @param[in] semanticTypeInfo The semantic type info of the parameter, which can be different from the data type info to handle parameters with the same data type differently
     * @param[in] dataTypeInfo The type info of the parameter value
     * @param[out] out The parameter value is returned here if true is returned
     * @return true if the requested parameter value was set in the out parameter,
     *         false if the requested parameter isn't available
     * @throws if the requested parameter failed to be retrieved
     */
    // Overloaded functions are provided for ease of use
    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual bool get(
        ParameterProvider const* const parentProvider,
        dw::core::StringView const& key,
        const std::type_info& semanticTypeInfo,
        const std::type_info& dataTypeInfo,
        void* out) const = 0;

    /**
     * Get a parameter value from an array parameter by index.
     *
     * @param[in] parentProvider The parent provider in case the lookup goes through a hierarchy of providers, otherwise nullptr
     * @param[in] key The parameter name, or an empty string if the (semantic) type sufficiently identifies the parameter
     * @param[in] index The index of the value in the array parameter
     * @param[in] semanticTypeInfo The semantic type info of the parameter, which can be different from the data type info to handle parameters with the same data type differently
     * @param[in] dataTypeInfo The type info of the parameter value
     * @param[out] out The parameter value is returned here if true is returned
     * @return true if the requested parameter value was set in the out parameter,
     *         false if the requested parameter isn't available
     * @throws if the requested parameter failed to be retrieved
     */
    // Overloaded functions are provided for ease of use
    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual bool get(
        ParameterProvider const* const parentProvider,
        dw::core::StringView const& key, size_t const index,
        const std::type_info& semanticTypeInfo,
        const std::type_info& dataTypeInfo,
        void* out) const = 0;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PARAMETERPROVIDER_HPP_
