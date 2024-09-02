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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_DEVICESPAN_HPP_
#define DWSHARED_CORE_DEVICESPAN_HPP_

#include <dwshared/dwfoundation/dw/core/container/Span.hpp>
#include "../misc/DevicePtr.hpp"
#include <utility>

namespace dw
{
namespace core
{

/**
 * Wrapper class that holds a span. This is meant as a safety mechanism
 * when we have a span in CPU that points to GPU memory.<br>
 *
 * Example:<br>
 *    // CPU function receives a span of CPU memory, processing happens in CPU
 *  @code{.cpp}
 *    void foo(span<uint8_t,2> data)
 *    {
 *        data(0,0) = data(1,1);
 *    }
 *  @endcode
 *
 *    //Kernel receives a span of GPU memory, processing happens in GPU
 *  @code{.cpp}
 *    __global__ myKernel(span<uint8_t,2> data)
 *    {
 *        data(0,0) = data(1,1);
 *    }
 *  @endcode
 *
 *    // CPU function receives a span of GPU memory, processing happens in GPU
 *  @code{.cpp}
 *    void foo(DeviceSpan<uint8_t,2> data)
 *    {
 *        myKernel<<<1,1>>>(data.get());
 *    }
 *  @endcode
 */
template <typename T, size_t DIMS = 1>
class DeviceSpan
{
public:
    using value_type = T;                                      ///< value type
    using TSpan      = span<T, DIMS>;                          ///< span type
    using SizeArray  = typename TSpan::SizeArray;              ///< Array type for size per dimension
    using TSize      = decltype(std::declval<TSpan>().size()); ///< size type

    /// Marker to represent end of span in a dimension
    static constexpr size_t NPOS = TSpan::NPOS;

    DeviceSpan() = default;

    /// Construction from span
    explicit DeviceSpan(span<T, DIMS> span_)
        : m_span(std::move(span_))
    {
    }

    /// Empty construction
    explicit DeviceSpan(std::nullptr_t)
        : m_span{nullptr}
    {
    }

    /// Conversion to const T
    /// Dummy template so this doesn't get defined for const spans (avoid nvcc warning)
    template <class T_ = T, typename = typename std::enable_if<std::is_same<T_, T>::value>::type>
    // TODO(dwplc): RFD - It is safe and following C++ semantics to implicitly convert from non-const to const span.
    // coverity[autosar_cpp14_a13_5_2_violation]
    operator DeviceSpan<const T_, DIMS>() const // clang-tidy NOLINT
    {
        return DeviceSpan<const T, DIMS>(m_span);
    }

    auto get() const -> span<T, DIMS> { return m_span; } ///< Get span
    void set(span<T, DIMS> s) { m_span = s; }            ///< Set span

    /// Reshapes into a 1D DeviceSpan.
    /// Size can be 0 to have it fill automatically.
    /// Throws if there is padding in the pitch.
    auto reshape(size_t size0) const -> DeviceSpan<T, 1> { return DeviceSpan<T, 1>(m_span.reshape(size0)); }

    /// Reshapes into a 2D DeviceSpan.
    /// One of the sizes can be 0 to have it fill automatically.
    auto reshape(size_t size0, size_t size1) const -> DeviceSpan<T, 2> { return DeviceSpan<T, 2>(m_span.reshape(size0, size1)); }

    /// Helper function in case one needs to pass the span as a simple DevicePtr.
    /// It is always preferred to pass it as a span
    auto data() const -> DevicePtr<T[]> { return MakeDeviceArray(m_span.data()); }

    /// Get size
    auto size() const -> TSize { return m_span.size(); }

    /// Returns ture if DeviceSpan is empty
    bool empty() const { return m_span.empty(); }

    /// Get DeviceSpan that is a subspan defined by offset and size for each dimension.
    auto subspan(SizeArray const& offsetIn = SizeArray{},
                 SizeArray const& sizeIn   = makeFilledArray<TSize, DIMS, NPOS>()) const -> DeviceSpan<T, DIMS>
    {
        return DeviceSpan(m_span.subspan(offsetIn, sizeIn));
    }

    /// Get 1-dimensional DeviceSpan subspan.
    auto subspan(size_t const offsetIn, size_t const sizeIn = NPOS) const -> DeviceSpan<T, DIMS>
    {
        return DeviceSpan(m_span.subspan(offsetIn, sizeIn));
    }

private:
    span<T, DIMS> m_span;
};

/// Helper functions to make a DeviceSpan from a DevicePtr

/// Make 1-dimensional DeviceSpan from DevicePtr (pointer type)
template <typename T>
auto make_span(DevicePtr<T> ptr) -> DeviceSpan<T, 1>
{
    return DeviceSpan<T, 1>(span<T, 1>(ptr.get(), 1));
}

/// Make 1-dimensional DeviceSpan from DevicePtr array (array type)
template <typename T>
auto make_span(DevicePtr<T[]> ptr, size_t s) -> DeviceSpan<T, 1>
{
    return DeviceSpan<T, 1>(span<T, 1>(ptr.get(), s));
}

/// Make n-dimensional DeviceSpan from span
template <typename T, size_t N>
auto makeDeviceSpan(span<T, N> s) -> DeviceSpan<T, N>
{
    return DeviceSpan<T, N>(s);
}

/// Make 1-dimensional DeviceSpan from raw pointer and size
template <typename T>
auto makeDeviceSpan(T* ptr, size_t s) -> DeviceSpan<T, 1>
{
    return makeDeviceSpan(make_span(ptr, s));
}

/// Make n-dimensional DeviceSpan from raw pointer, size per dimension and byte pitch
template <typename T, size_t N>
auto makeDeviceSpan(T* ptr, const Array<size_t, N>& s, size_t pitchBytes) -> DeviceSpan<T, N>
{
    return makeDeviceSpan(make_span(ptr, s, pitchBytes));
}

// Helper functions to copy spans

/// Asynchronous copy from cuda device memory to cuda device memory.
template <typename Td, typename Ts, size_t DIMS>
cudaError_t memcpyAsync(DeviceSpan<Td, DIMS> dst, DeviceSpan<Ts, DIMS> src, cudaStream_t stream = nullptr)
{
    return internal::memcpyAsync(dst.get(), src.get(), cudaMemcpyDeviceToDevice, stream);
}

/// Asynchronous copy from host memory to cuda device memory.
template <typename Td, typename Ts, size_t DIMS>
cudaError_t memcpyAsync(DeviceSpan<Td, DIMS> const dst, span<Ts, DIMS> const src, cudaStream_t const stream = nullptr)
{
    return internal::memcpyAsync(dst.get(), src, cudaMemcpyHostToDevice, stream);
}

/// Asynchronous copy from cuda device memory to host memory.
template <typename Td, typename Ts, size_t DIMS>
cudaError_t memcpyAsync(span<Td, DIMS> const dst, DeviceSpan<Ts, DIMS> const src, cudaStream_t const stream = nullptr)
{
    return internal::memcpyAsync(dst, src.get(), cudaMemcpyDeviceToHost, stream);
}

/////////////////////////////////////////////////////////////////////
// Helper functions to copy DevicePtr
/////////////////////////////////////////////////////////////////////

/// Asynchronous copy from cuda device memory to host memory.
template <typename Td, typename Ts>
cudaError_t memcpyAsync(Td& dst, DevicePtr<Ts> src, cudaStream_t stream = nullptr)
{
    return memcpyAsync(make_span(&dst, 1), make_span(src), stream);
}

/// Asynchronous copy from cuda device memory to cuda device memory.
template <typename Td, typename Ts>
cudaError_t memcpyAsync(DevicePtr<Td> dst, DevicePtr<Ts> src, cudaStream_t stream = nullptr)
{
    return memcpyAsync(make_span(dst), make_span(src), stream);
}

/// Asynchronous copy from host memory to cuda device memory.
template <typename Td, typename Ts>
cudaError_t memcpyAsync(DevicePtr<Ts> dst, Td& src, cudaStream_t stream = nullptr)
{
    return memcpyAsync(make_span(dst), make_span(&src, 1), stream);
}

/// Helper function to zero out a span
template <typename Td, size_t DIMS>
cudaError_t zeroMemAsync(DeviceSpan<Td, DIMS> const dst, cudaStream_t const stream = nullptr)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto s = dst.get();

    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto const s2d = s.reshape(s.size(0), 0);
    return cudaMemset2DAsync(s2d.data(), s2d.pitch_bytes(), 0, s2d.size(0) * sizeof(Td), s2d.size(1), stream);
}

/// Create cudaResourceDesc from 2-dimensional DeviceSpan
template <class T>
cudaResourceDesc createResourceDesc(const DeviceSpan<T, 2>& d_span)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    auto span = d_span.get();

    cudaChannelFormatDesc channelDesc{};
    if (std::is_floating_point<T>::value)
    {
        channelDesc.f = cudaChannelFormatKindFloat;
    }
    else if (std::is_signed<T>::value)
    {
        channelDesc.f = cudaChannelFormatKindSigned;
    }
    else
    {
        channelDesc.f = cudaChannelFormatKindUnsigned;
    }
    channelDesc.x = 8 * sizeof(T);

    // cuda structs use unions, have to disable static checker
    cudaResourceDesc resDesc{};
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = span.data();        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.desc         = channelDesc;        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.pitchInBytes = span.pitch_bytes(); // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.height       = span.size(1);       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.width        = span.size(0);       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    return resDesc;
}

/// DeviceSpan equality comparison operator
template <typename T, size_t N>
bool operator==(const DeviceSpan<T, N>& lhs, const DeviceSpan<T, N>& rhs) noexcept
{
    return lhs.get() == rhs.get();
}

/// DeviceSpan inequality comparison operator
template <typename T, size_t N>
bool operator!=(const DeviceSpan<T, N>& lhs, const DeviceSpan<T, N>& rhs) noexcept
{
    return !operator==(lhs, rhs);
}

} // namespace cuda
} // namespace dw

#endif
