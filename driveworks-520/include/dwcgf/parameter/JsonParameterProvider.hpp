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

#ifndef DW_FRAMEWORK_JSONPARAMETERPROVIDER_HPP_
#define DW_FRAMEWORK_JSONPARAMETERPROVIDER_HPP_

#include <dwcgf/parameter/TypeBasedParameterProvider.hpp>

#include <modern-json/json.hpp>

namespace dw
{
namespace framework
{

/// A parameter provider which retrieves parameter values from JSON data.
class JsonParameterProvider : public ITypeBasedParameterProviderChild
{
protected:
    /// Copy constructor.
    JsonParameterProvider(JsonParameterProvider const&) = default;
    /// Move constructor.
    JsonParameterProvider(JsonParameterProvider&&) = default;
    /// Copy assignment operator.
    JsonParameterProvider& operator=(JsonParameterProvider const&) & = default;
    /// Move assignment operator.
    JsonParameterProvider& operator=(JsonParameterProvider&&) & = default;

public:
    /// Constructor.
    JsonParameterProvider(nlohmann::json const* const data) noexcept;
    /// Destructor.
    ~JsonParameterProvider() override = default;

    /// Set the JSON data.
    void setJson(nlohmann::json const* const data) noexcept;

    /// Get the JSON data.
    nlohmann::json data() const;

    /// @see dw::framework::ITypeBasedParameterProviderChild::registerAt()
    void registerAt(TypeBasedParameterProvider& provider) const override;

protected:
    /// Get JSON value for the passed key.
    nlohmann::json const* getJson(dw::core::StringView const& key) const;

    /// Handler function to retrieve a bool value.
    bool getBool(dw::core::StringView const& key, bool& out) const;
    /// Handler function to retrieve a int8_t value.
    bool getInt8(dw::core::StringView const& key, int8_t& out) const;
    /// Handler function to retrieve a int16_t value.
    bool getInt16(dw::core::StringView const& key, int16_t& out) const;
    /// Handler function to retrieve a int32_t value.
    bool getInt32(dw::core::StringView const& key, int32_t& out) const;
    /// Handler function to retrieve a int64_t value.
    bool getInt64(dw::core::StringView const& key, int64_t& out) const;
    /// Handler function to retrieve a uint8_t value.
    bool getUint8(dw::core::StringView const& key, uint8_t& out) const;
    /// Handler function to retrieve a uint16_t value.
    bool getUint16(dw::core::StringView const& key, uint16_t& out) const;
    /// Handler function to retrieve a uint32_t value.
    bool getUint32(dw::core::StringView const& key, uint32_t& out) const;
    /// Handler function to retrieve a uint64_t value.
    bool getUint64(dw::core::StringView const& key, uint64_t& out) const;
    /// Handler function to retrieve a float32_t value.
    bool getFloat32(dw::core::StringView const& key, float32_t& out) const;
    /// Handler function to retrieve a float64_t value.
    bool getFloat64(dw::core::StringView const& key, float64_t& out) const;
    /// Handler function to retrieve a dw::core::StringView value.
    bool getStringView(dw::core::StringView const& key, dw::core::StringView& out) const;

    /// Handler function to retrieve a vector of int8_t value.
    bool getVectorInt8(dw::core::StringView const& key, std::vector<int8_t>& out) const;
    /// Handler function to retrieve a vector of int16_t value.
    bool getVectorInt16(dw::core::StringView const& key, std::vector<int16_t>& out) const;
    /// Handler function to retrieve a vector of int32_t value.
    bool getVectorInt32(dw::core::StringView const& key, std::vector<int32_t>& out) const;
    /// Handler function to retrieve a vector of int64_t value.
    bool getVectorInt64(dw::core::StringView const& key, std::vector<int64_t>& out) const;
    /// Handler function to retrieve a vector of uint8_t value.
    bool getVectorUint8(dw::core::StringView const& key, std::vector<uint8_t>& out) const;
    /// Handler function to retrieve a vector of uint16_t value.
    bool getVectorUint16(dw::core::StringView const& key, std::vector<uint16_t>& out) const;
    /// Handler function to retrieve a vector of uint32_t value.
    bool getVectorUint32(dw::core::StringView const& key, std::vector<uint32_t>& out) const;
    /// Handler function to retrieve a vector of uint64_t value.
    bool getVectorUint64(dw::core::StringView const& key, std::vector<uint64_t>& out) const;
    /// Handler function to retrieve a vector of float32_t value.
    bool getVectorFloat32(dw::core::StringView const& key, std::vector<float32_t>& out) const;
    /// Handler function to retrieve a vector of float64_t value.
    bool getVectorFloat64(dw::core::StringView const& key, std::vector<float64_t>& out) const;
    /// Handler function to retrieve a vector of dw::core::StringView value.
    bool getVectorStringView(dw::core::StringView const& key, std::vector<dw::core::StringView>& out) const;

    /// Handler function to retrieve a bool value from an array by index.
    bool getBoolByIndex(dw::core::StringView const& key, size_t const index, bool& out) const;
    /// Handler function to retrieve a int8_t value from an array by index.
    bool getInt8ByIndex(dw::core::StringView const& key, size_t const index, int8_t& out) const;
    /// Handler function to retrieve a int16_t value from an array by index.
    bool getInt16ByIndex(dw::core::StringView const& key, size_t const index, int16_t& out) const;
    /// Handler function to retrieve a int32_t value from an array by index.
    bool getInt32ByIndex(dw::core::StringView const& key, size_t const index, int32_t& out) const;
    /// Handler function to retrieve a int64_t value from an array by index.
    bool getInt64ByIndex(dw::core::StringView const& key, size_t const index, int64_t& out) const;
    /// Handler function to retrieve a uint8_t value from an array by index.
    bool getUint8ByIndex(dw::core::StringView const& key, size_t const index, uint8_t& out) const;
    /// Handler function to retrieve a uint16_t value from an array by index.
    bool getUint16ByIndex(dw::core::StringView const& key, size_t const index, uint16_t& out) const;
    /// Handler function to retrieve a uint32_t value from an array by index.
    bool getUint32ByIndex(dw::core::StringView const& key, size_t const index, uint32_t& out) const;
    /// Handler function to retrieve a uint64_t value from an array by index.
    bool getUint64ByIndex(dw::core::StringView const& key, size_t const index, uint64_t& out) const;
    /// Handler function to retrieve a float32_t value from an array by index.
    bool getFloat32ByIndex(dw::core::StringView const& key, size_t const index, float32_t& out) const;
    /// Handler function to retrieve a float64_t value from an array by index.
    bool getFloat64ByIndex(dw::core::StringView const& key, size_t const index, float64_t& out) const;
    /// Handler function to retrieve a dw::core::StringView value from an array by index.
    bool getStringViewByIndex(dw::core::StringView const& key, size_t const index, dw::core::StringView& out) const;

private:
    /// The JSON data
    nlohmann::json const* m_data;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_JSONPARAMETERPROVIDER_HPP_
