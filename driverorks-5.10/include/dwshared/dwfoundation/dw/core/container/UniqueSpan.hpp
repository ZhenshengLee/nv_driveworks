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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_UNIQUESPAN_HPP_
#define DWSHARED_CORE_UNIQUESPAN_HPP_

#include "Span.hpp"

namespace dw
{

namespace core
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// These classes works like std::unique_ptr but holds memory attributes too for bound-safe access.
// The memory owned by UniqueXXXSpan will be released at object's EOL
template <typename T, size_t DIMS = 1>
class UniqueSpan;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// These enum provides a unified interface to declare/create unique spans in different
/// memory space.
/// Usage:
/// auto uniqueSpan0 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperCPU>::makeUniqueSpan<T>(...);
/// auto uniqueSpan1 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperCUDA>::makeUniqueSpan<T>(...);
/// auto uniqueSpan2 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperPinned>::makeUniqueSpan<T>(...);
/// auto uniqueSpan3 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperMapped>::makeUniqueSpan<T>(...);
enum class UniqueSpanHelperType : uint8_t
{
    CPU = 0, // enum value for UniqueSpan<>
    CUDA,    // enum value for UniqueDeviceSpan<> (provided by dw/cuda/container/UniqueSpan.hpp)
    Pinned,  // enum value for UniquePinnedSpan<> (provided by dw/cuda/container/UniqueSpan.hpp)
    Mapped,  // enum value for UniqueMappedSpan<> (provided by dw/cuda/container/UniqueSpan.hpp)
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// UniqueSpan (host memory)
template <typename T, size_t DIMS_>
class UniqueSpan
{
public:
    /// Type of contained elements
    using ElementType = T;

    /// Number of dimensions for indexed element access
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t DIMS = DIMS_;

    /// Default constructor for empty UniqueSpan
    UniqueSpan()
        : UniqueSpan(nullptr)
    {
    }

    /// Constructor for empty UniqueSpan
    UniqueSpan(std::nullptr_t) // clang-tidy NOLINT(google-explicit-constructor)
        : m_span(nullptr)
    {
    }

    /// Move constructor
    UniqueSpan(UniqueSpan&& other)
        // TODO(dwplc): FP - this is using move semantics
        // coverity[autosar_cpp14_a12_8_4_violation]
        : m_span(std::move(other.m_span))
    {
        static_assert(sizeof(UniqueSpan) == sizeof(span<T, DIMS>), "UniqueSpan shall not add memory overhead");

        // TODO(plc): This line is required for MatrixCovarianceTest.Basic_L0 to pass -> Change the test and remove this line for coverity
        other.m_span = span<T, DIMS>{nullptr};
    }

    /// Move operator
    auto operator=(UniqueSpan&& other) -> UniqueSpan&
    {
        if (m_span.data() != nullptr)
        {
            // TODO(dwplc): FP - delete[] doesn't perform unsigned integer operation
            // coverity[cert_int30_c_violation]
            // coverity[autosar_cpp14_a18_5_2_violation]: FP - Exception applies to user-defined RAII class
            delete[] m_span.data();
        }

        m_span       = std::move(other.m_span);
        other.m_span = span<T, DIMS>{nullptr};
        return *this;
    }

    UniqueSpan(const UniqueSpan&) = delete;
    auto operator=(const UniqueSpan&) -> UniqueSpan& = delete;

    /// Construction with span.
    /// Takes ownership of referenced memory!
    /// Referenced data is assumed to be array allocated
    explicit UniqueSpan(span<T, DIMS> span_)
        : m_span(std::move(span_))
    {
    }

    /// Destructor
    /// Destroys contained objects and releases allocated memory.
    ~UniqueSpan()
    {
        reset();
    }

    /// Destroys contained objects and releases allocated memory.
    void reset()
    {
        if (m_span.data() != nullptr)
        {
            // TODO(dwplc): FP - delete[] doesn't perform unsigned integer operation
            // coverity[cert_int30_c_violation]
            // coverity[autosar_cpp14_a18_5_2_violation]: FP - Exception applies to user-defined RAII class
            delete[] m_span.data();
            m_span = span<T, DIMS>();
        }
    }

    auto get() const -> const span<T, DIMS>& { return m_span; } ///< Get span of elements

    auto operator*() const -> const span<T, DIMS>& { return m_span; }    ///< Get span of elements
    auto operator-> () const -> const span<T, DIMS>* { return &m_span; } ///< Get span of elements

    /// Indexed access for 1-dimensional UniqueSpan
    template <size_t DIMS2 = DIMS> // Make a dummy template so enable_if works
    auto operator[](const typename std::enable_if<DIMS2 == 1, size_t>::type idx) const -> T&
    {
        return m_span[idx];
    }

    /// Indexed access for multi-dimensional UniqueSpan
    /// Returns the span with one dimension less.
    template <size_t DIMS2 = DIMS> // Make a dummy template so enable_if works
    auto operator[](const typename std::enable_if<DIMS2 != 1, size_t>::type idx) const -> span<T, DIMS - 1>
    {
        return m_span[idx];
    }

    /// Releases ownership of stored pointer. Needs to be cleaned up separately!
    auto release() -> span<T, DIMS>
    {
        auto temp = m_span;
        m_span    = span<T, DIMS>{nullptr};
        return temp;
    }

    /// Returns true if UniqueSpan is empty
    bool empty() const
    {
        return m_span.data() == nullptr;
    }

    /// Returns true if UniqueSpan contains data
    explicit operator bool() const
    {
        return !empty();
    }

    /// Fill data with input value
    void fill(T const& value)
    {
        dw::core::fill(m_span, value);
    }

private:
    span<T, DIMS> m_span;
};

template <typename T, size_t DIMS>
bool operator==(const UniqueSpan<T, DIMS>& lhs, const UniqueSpan<T, DIMS>& rhs) noexcept
{
    return lhs.get() == rhs.get();
}

/// create UniqueSpan and allocate host memory
/// (general case of more than 1 dimension)
template <typename T, size_t DIMS, typename std::enable_if<(DIMS > 1), int32_t>::type = 0>
auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> UniqueSpan<T, DIMS>
{
    size_t totalSize = size[0];
    for (size_t i = 1; i < DIMS; i++)
    {
        totalSize = safeMul(totalSize, size[i]).value();
    }

    assertExceptionIf<NumericException>(core::numeric_limits<size_t>::max() / sizeof(T) >= totalSize,
                                        "makeUniqueSpan(): Overflow error");

    return UniqueSpan<T, DIMS>({new T[totalSize], size});
}

/// create UniqueSpan and allocate host memory, uninitialized
/// (special case of one dimension, needed to avoid AUTOSAR complaint about non-reachable code)
template <typename T, size_t DIMS, typename std::enable_if<(DIMS == 1), int32_t>::type = 0>
auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> UniqueSpan<T, DIMS>
{
    // TODO(dwplc): FP - When setting totalSize as const, error is raised:
    // non-constant array new length must be specified without parentheses around the type-id
    // coverity[autosar_cpp14_a7_1_1_violation]
    size_t totalSize = size[0];

    assertExceptionIf<NumericException>(core::numeric_limits<size_t>::max() / sizeof(T) >= totalSize,
                                        "makeUniqueSpan(): Overflow error");

    return UniqueSpan<T, DIMS>({new T[totalSize], size});
}

/// create UniqueSpan, allocate host memory, initialize with provided value
/// (one dimensional)
/// needs a different name from makeUniqueSpan, to distinguish from initialization with multiple sizes
template <typename T, size_t DIMS, typename std::enable_if<(DIMS == 1), int32_t>::type = 0>
auto makeUniqueSpanInitialized(const T& initializationValue, const Array<size_t, DIMS>& size) -> UniqueSpan<T, DIMS>
{
    size_t oneDArraySize           = size[0];
    std::unique_ptr<T[]> arrayUPtr = std::make_unique<T[]>(oneDArraySize);
    for (size_t i = 0; i < oneDArraySize; ++i)
    {
        arrayUPtr[i] = initializationValue;
    }
    return UniqueSpan<T, DIMS>({arrayUPtr.release(), oneDArraySize});
}

/// create UniqueSpan, allocate host memory, copy from provided UniqueSpan
/// (one dimensional)
template <typename T, size_t DIMS, typename std::enable_if<(DIMS == 1), int32_t>::type = 0>
auto makeUniqueSpan(const UniqueSpan<T, DIMS>& other) -> UniqueSpan<T, DIMS>
{
    size_t oneDArraySize           = other.get().size();
    std::unique_ptr<T[]> arrayUPtr = std::make_unique<T[]>(oneDArraySize);
    for (size_t i = 0; i < oneDArraySize; ++i)
    {
        arrayUPtr[i] = other->at(i);
    }
    return UniqueSpan<T, DIMS>({arrayUPtr.release(), oneDArraySize});
}

/// create UniqueSpan and allocate host memory, uninitialized
template <typename T, typename... Sizes>
auto makeUniqueSpan(size_t const size0, Sizes... sizes) -> UniqueSpan<T, 1 + sizeof...(Sizes)>
{
    return makeUniqueSpan<T>(Array<size_t, 1 + sizeof...(Sizes)>{size0, sizes...});
}

/// create UniqueSpan and allocate host memory, initialized
template <typename T, typename... Sizes>
auto makeUniqueSpanInitialized(const T& initializationValue, size_t size0, Sizes... sizes) -> UniqueSpan<T, 1 + sizeof...(Sizes)>
{
    return makeUniqueSpanInitialized<T>(initializationValue, Array<size_t, 1 + sizeof...(Sizes)>{size0, sizes...});
}

struct UniqueSpanHelperHost
{
    template <typename T, size_t DIMS = 1>
    using TSpan = span<T, DIMS>;

    /// create a host span by ptr and size with unified interface
    template <typename T>
    static TSpan<T, 1> makeSpan(T* ptr, size_t s)
    {
        return dw::core::make_span(ptr, s);
    }

    /// create a host span from span with unified interface
    template <typename T, size_t DIMS = 1>
    static TSpan<T, DIMS> makeSpan(span<T, DIMS> s)
    {
        return s;
    }

    /// get raw pointer from a host span with unified interface
    template <typename T>
    static T* getRawPtr(TSpan<T, 1> s)
    {
        return s.data();
    }
};

/// unified interface to declare/create unique spans in different memory space.
template <typename EnumClass, EnumClass EnumVal>
struct TUniqueSpanHelper;

/// unified interface to declare/create unique spans with CPU pageable memory.
template <>
struct TUniqueSpanHelper<UniqueSpanHelperType, UniqueSpanHelperType::CPU> : public UniqueSpanHelperHost
{
    template <typename T, size_t DIMS = 1>
    using TSpan = UniqueSpanHelperHost::TSpan<T, DIMS>;

    template <typename T, size_t DIMS = 1>
    using TUniqueSpan = UniqueSpan<T, DIMS>;

    /// equvalent to core::makeUniqueSpan but using unified interface
    template <typename T, size_t DIMS>
    static auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> TUniqueSpan<T, DIMS>
    {
        return dw::core::makeUniqueSpan<T>(size);
    }

    /// equvalent to core::makeUniqueSpan but using unified interface
    template <typename T, typename... Sizes>
    static auto makeUniqueSpan(size_t const size0, Sizes... sizes) -> TUniqueSpan<T, 1 + sizeof...(Sizes)>
    {
        return dw::core::makeUniqueSpan<T>(size0, sizes...);
    }
};

/// alias to TUniqueSpanHelper for easy usage
/// unified interface to declare/create unique spans in different memory space.
template <UniqueSpanHelperType TYPE>
using UniqueSpanHelper = TUniqueSpanHelper<decltype(TYPE), TYPE>;

/// Helper type UniqueHostSpan for pinned memory
template <typename T, bool USE_PINNED>
struct UniqueHostSpan;

/// Helper type UniqueHostSpan for non-pinned memory
template <typename T>
struct UniqueHostSpan<T, false>
{
    using Type = UniqueSpan<T>; ///< Return type

    /// Allocate buffer with capacity and return UniqueSpan
    static auto allocate(size_t const argCapacity) -> Type
    {
        return makeUniqueSpan<T>(argCapacity);
    }
};

} // namespace core
} // namespace dw

#endif
