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

#ifndef DW_CORE_CUDACHANNELDESC_HPP_
#define DW_CORE_CUDACHANNELDESC_HPP_

#include <type_traits>
#include <channel_descriptor.h>

#include <dw/core/matrix/BaseMatrix.hpp>

/**
 * @brief cudaChannelFormatDesc cudaCreateChannelDesc<typename T>(void)
 *        A series of helper functions are provided to ease users to create proper
 *        cudaChannelFormatDesc descriptors for common data types including vectors
 *        of primitive data types.
 *        Declaration of helper functions provided are
 *        <tt>template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<typename T>(void)</tt>
 *
 * @note  Data type to create a channel descriptor from. Available types including
 *        <tt>__half</tt>,
 *        <tt>dw::core::Vector<uint8_t, 1>,  dw::core::Vector<uint8_t, 2>, dw::core::Vector<uint8_t, 3></tt>,
 *        <tt>dw::core::Vector<dwFloat16_t, 1>, dw::core::Vector<dwFloat16_t, 2>, dw::core::Vector<dwFloat16_t, 3></tt>,
 *        <tt>dw::core::Vector<float32_t, 1>, dw::core::Vector<float32_t, 2>, dw::core::Vector<float32_t, 3></tt>
 *        These helper functions return a <tt>cudaChannelFormatDesc</tt> instance for given data type @b T.
 */
namespace dw
{
namespace core
{

////////////////////////////////////////////////////////////////////////
/**
 * Helper to determine the channel format kind(in cudaChannelFormatKind type) from a data type.
 */
template <typename T, typename = void>
struct ChannelFormatKind
{
    /// Enum value of enum type cudaChannelFormatKind for input element data type @b T.
    /// This value always equals to @b cudaChannelFormatKindUnsigned.
    static const cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned; // clang-tidy NOLINT
};

/**
 * Helper to determine the channel format kind(in cudaChannelFormatKind type) from float type.
 */
template <typename T>
struct ChannelFormatKind<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{
    /// Enum value of enum type cudaChannelFormatKind value for input float types.
    /// This value always equals to @b cudaChannelFormatKindFloat
    static const cudaChannelFormatKind kind = cudaChannelFormatKindFloat; // clang-tidy NOLINT
};

/**
 * Helper to determine the channel format kind(in cudaChannelFormatKind type) from signed non float type.
 */
template <typename T>
struct ChannelFormatKind<T, typename std::enable_if<!std::is_floating_point<T>::value && std::is_signed<T>::value>::type>
{
    /// Enum value of enum type cudaChannelFormatKind value for input non-floating types.
    /// This value always equals to @b cudaChannelFormatKindSigned
    static const cudaChannelFormatKind kind = cudaChannelFormatKindSigned; // clang-tidy NOLINT
};
}
}

// TODO(dwplc): FP - "cudaCreateChannelDesc is a cuda api in global namespace"
// TODO(dwplc): RFD - Specializations of cudaCreateChannelDesc are widely used in
//                    external/cuda_linux_x86_64_pdk_5_1_15_2/targets/x86_64-linux/include/channel_descriptor.h
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a14_8_2_violation]
/// Create a CUDA channel descriptor from __half(half precision) type
template <>
__inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<__half>()
{
    return cudaCreateChannelDesc(static_cast<int32_t>(8 * sizeof(__half)), 0, 0, 0, cudaChannelFormatKindFloat);
}

// TODO(dwplc): FP - ## is only used once per declaration, and T is template argument which can't be enclosed in parentheses
// coverity[autosar_cpp14_m16_0_6_violation]
#define SPECIALIZE_CHANNEL_VECTOR(T)                                                              \
    template <>                                                                                   \
    __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<dw::core::Vector<T, 1>>(void) \
    {                                                                                             \
        int32_t constexpr e = static_cast<int32_t>(8 * sizeof(T));                                \
        return cudaCreateChannelDesc(e, 0, 0, 0, dw::core::ChannelFormatKind<T>::kind);           \
    }                                                                                             \
    template <>                                                                                   \
    __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<dw::core::Vector<T, 2>>(void) \
    {                                                                                             \
        int32_t constexpr e = static_cast<int32_t>(8 * sizeof(T));                                \
        return cudaCreateChannelDesc(e, e, 0, 0, dw::core::ChannelFormatKind<T>::kind);           \
    }                                                                                             \
    template <>                                                                                   \
    __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<dw::core::Vector<T, 3>>(void) \
    {                                                                                             \
        int32_t constexpr e = static_cast<int32_t>(8 * sizeof(T));                                \
        return cudaCreateChannelDesc(e, e, e, 0, dw::core::ChannelFormatKind<T>::kind);           \
    }                                                                                             \
    template <>                                                                                   \
    __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<dw::core::Vector<T, 4>>(void) \
    {                                                                                             \
        int32_t constexpr e = static_cast<int32_t>(8 * sizeof(T));                                \
        return cudaCreateChannelDesc(e, e, e, e, dw::core::ChannelFormatKind<T>::kind);           \
    }
// TODO(dwplc): FP - "cudaCreateChannelDesc is a cuda api in global namespace"
// TODO(dwplc): RFD - Specializations of cudaCreateChannelDesc are widely used in
//                    external/cuda_linux_x86_64_pdk_5_1_15_2/targets/x86_64-linux/include/channel_descriptor.h
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a14_8_2_violation]
SPECIALIZE_CHANNEL_VECTOR(uint8_t)
// TODO(dwplc): FP - "cudaCreateChannelDesc is a cuda api in global namespace"
// TODO(dwplc): RFD - Specializations of cudaCreateChannelDesc are widely used in
//                    external/cuda_linux_x86_64_pdk_5_1_15_2/targets/x86_64-linux/include/channel_descriptor.h
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a14_8_2_violation]
SPECIALIZE_CHANNEL_VECTOR(dwFloat16_t)
// TODO(dwplc): FP - "cudaCreateChannelDesc is a cuda api in global namespace"
// TODO(dwplc): RFD - Specializations of cudaCreateChannelDesc are widely used in
//                    external/cuda_linux_x86_64_pdk_5_1_15_2/targets/x86_64-linux/include/channel_descriptor.h
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a14_8_2_violation]
SPECIALIZE_CHANNEL_VECTOR(float32_t)

#undef SPECIALIZE_CHANNEL_VECTOR

#endif // DW_CORE_CUDACHANNELDESC_HPP_
