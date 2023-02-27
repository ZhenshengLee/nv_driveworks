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

#include <dw/core/container/Span.hpp>
#include <dw/core/container/DeviceSpan.hpp>
#include <dw/cuda/misc/UniqueDevicePtr.hpp>
#include <dw/cuda/misc/Checks.hpp>

namespace dw
{

namespace core
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// These classes works like std::unique_ptr but holds memory attributes too for bound-safe access.
// The memory owned by UniqueXXXSpan will be released at object's EOL
template <typename T, size_t DIMS = 1>
class UniqueSpan;

template <typename T, size_t DIMS = 1>
class UniqueDeviceSpan;

template <typename T, size_t DIMS = 1>
class UniquePinnedSpan;

template <typename T, size_t DIMS = 1>
class UniqueMappedSpan;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// These enum provides a unified interface to declare/create unique spans in different
/// memory space.
/// Usage:
/// auto uniqueSpan0 = UniqueSpanHelper<UniqueSpanType::UniqueUniqueSpanHelperCPU>::makeUniqueSpan<T>(...);
/// auto uniqueSpan1 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperCUDA>::makeUniqueSpan<T>(...);
/// auto uniqueSpan2 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperPinned>::makeUniqueSpan<T>(...);
/// auto uniqueSpan3 = UniqueSpanHelper<UniqueSpanType::UniqueSpanHelperMapped>::makeUniqueSpan<T>(...);
enum class UniqueSpanHelperType : uint8_t
{
    CPU = 0, // enum value for UniqueSpan<>
    CUDA,    // enum value for UniqueDeviceSpan<>
    Pinned,  // enum value for UniquePinnedSpan<>
    Mapped,  // enum value for UniqueMappedSpan<>
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

    assertExceptionIf<NumericException>(util::numeric_limits<size_t>::max() / sizeof(T) >= totalSize,
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

    assertExceptionIf<NumericException>(util::numeric_limits<size_t>::max() / sizeof(T) >= totalSize,
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

/// UniquePinnedSpan (host page-locked memory)
/// pinned memory and mapped memory are both allocated by cudaHostAlloc
/// and freed by cudaFreeHost, so they belong to the same page-locked
/// memory type.
/// So UniquePinnedSpan can also be used as a base class for UniqueMappedSpan,
/// its internal host page-locked memory is the general-purpose UniqueCudaHostPtr
/// instead of UniquePinnedPtr alias. When the class is a pinned span, the internal
/// pointer will be used as a UniquePinnedPtr; when it's the base class of a UniqueMappedSpan,
/// the internal pointer becomes a UniqueMappedPtr.
template <typename T, size_t DIMS_>
class UniquePinnedSpan
{
public:
    /// Type of contained elements
    using ElementType = T;

    /// Number of dimensions for indexed element access
    static constexpr size_t DIMS = DIMS_;

    UniquePinnedSpan()                        = default;                   ///< Default constructor
    UniquePinnedSpan(const UniquePinnedSpan&) = delete;                    ///< No copy constructor
    UniquePinnedSpan(UniquePinnedSpan&&)      = default;                   ///< Move constructor
    auto operator=(const UniquePinnedSpan&) -> UniquePinnedSpan& = delete; ///< No copy operator
    UniquePinnedSpan& operator=(UniquePinnedSpan&&) = default;             ///< Move operator
    ~UniquePinnedSpan()                             = default;             ///< Destructor

    /// Construction from span.
    explicit UniquePinnedSpan(span<T, DIMS> const spanIn)
        : m_ptr(spanIn.data())
        , m_span(spanIn) {}

    auto get() const -> span<T, DIMS> { return m_span; } ///< Get span.

    auto operator*() const -> const span<T, DIMS>& { return m_span; }    ///< Get span
    auto operator-> () const -> const span<T, DIMS>* { return &m_span; } ///< Get span

    /// Indexed access for 1-dimensional UniquePinnedspan
    template <size_t DIMS2 = DIMS> // Make a dummy template so enable_if works
    auto operator[](const typename std::enable_if<DIMS2 == 1, size_t>::type idx) const -> T&
    {
        return m_span[idx];
    }

    /// Indexed access for multi-dimensional UniquePinnedspan
    template <size_t DIMS2 = DIMS> // Make a dummy template so enable_if works
    auto operator[](const typename std::enable_if<DIMS2 != 1, size_t>::type idx) const -> span<T, DIMS - 1>
    {
        return m_span[idx];
    }

    /// Releases ownership of stored pointer and returns
    /// pointer to array.
    /// Does not release memory, needs to be cleaned up separately!
    auto release() -> T*
    {
        m_span = {};
        return m_ptr.release();
    }

    /// Returns true if UniquePinnedSpan contains any data
    explicit operator bool() const // clang-tidy NOLINT
    {
        return m_ptr.operator bool();
    }

private:
    UniqueCudaHostPtr<T[]> m_ptr;
    span<T, DIMS> m_span;
};

/// create UniqueSpan and allocate host memory
template <typename T, size_t DIMS>
auto makeUniquePinnedSpan(const Array<size_t, DIMS>& size) -> UniquePinnedSpan<T, DIMS>
{
    // TODO(dwplc): FP. totalSize is modified within the if statement.
    // coverity[autosar_cpp14_a7_1_1_violation]
    size_t totalSize = size[0];
    if (DIMS > 1)
    {
        for (size_t i = 1; i < DIMS; i++)
        {
            totalSize = safeMul(totalSize, size[i]).value();
        }
    }
    auto ptr = makeUniquePinnedHost<T[]>(totalSize);
    return UniquePinnedSpan<T, DIMS>({ptr.release(), size});
}

/// create UniqueSpan and allocate host memory
template <typename T, typename... Sizes>
auto makeUniquePinnedSpan(size_t const size0, Sizes... sizes) -> UniquePinnedSpan<T, 1 + sizeof...(Sizes)>
{
    return makeUniquePinnedSpan<T>(Array<size_t, 1 + sizeof...(Sizes)>{size0, sizes...});
}

/// UniqueMappedSpan (Multi-Dimensional)
/// Stores an Array in both host pinned memory (inheritence from UniquePinnedSpan)
/// and device memory (DeviceSpan)
template <typename T, size_t DIMS_>
class UniqueMappedSpan : public dw::core::UniquePinnedSpan<T, DIMS_>
{
    using Base = dw::core::UniquePinnedSpan<T, DIMS_>;

public:
    using Base::Base;
    using Base::get;

    UniqueMappedSpan()                        = default;                   ///< Default constructor
    UniqueMappedSpan(const UniqueMappedSpan&) = delete;                    ///< No copy constructor
    UniqueMappedSpan(UniqueMappedSpan&&)      = default;                   ///< Move constructor
    auto operator=(const UniqueMappedSpan&) -> UniqueMappedSpan& = delete; ///< No copy operator
    UniqueMappedSpan& operator=(UniqueMappedSpan&&) = default;             ///< Move operator
    ~UniqueMappedSpan()                             = default;             ///< Destructor

    /// Get DeviceSpan with elements in cuda memory.
    dw::core::DeviceSpan<T, DIMS_> getDeviceSpan() const
    {
        // The API ::cudaHostGetDevicePointer and dw::core::makeDeviceSpan are thread safe
        // and no memory allocation / free
        T* devPtr = nullptr;
#ifndef DW_IS_SAFETY // TODO(dwsafety): DRIV-8190
        // Creates a DeviceSpan that maps the span.
        DW_CHECK_CUDA_ERROR(::cudaHostGetDevicePointer(&devPtr, get().data(), 0));
#endif
        return dw::core::makeDeviceSpan(devPtr, get().size(), get().pitch_bytes());
    }
};

/// UniqueMappedSpan (1-Dimensional)
/// Stores an Array in both host pinned memory (inheritence from UniquePinnedSpan)
/// and device memory (DeviceSpan)
template <typename T>
class UniqueMappedSpan<T, 1> : public dw::core::UniquePinnedSpan<T, 1>
{
    using Base = dw::core::UniquePinnedSpan<T>;

public:
    using Base::Base;
    using Base::get;

    UniqueMappedSpan()                        = default;                   ///< Default constructor
    UniqueMappedSpan(const UniqueMappedSpan&) = delete;                    ///< No copy constructor
    UniqueMappedSpan(UniqueMappedSpan&&)      = default;                   ///< Move constructor
    auto operator=(const UniqueMappedSpan&) -> UniqueMappedSpan& = delete; ///< No copy operator
    UniqueMappedSpan& operator=(UniqueMappedSpan&&) = default;             ///< Move operator
    ~UniqueMappedSpan()                             = default;             ///< Destructor

    /// Get DeviceSpan
    dw::core::DeviceSpan<T> getDeviceSpan() const
    {
        // The API ::cudaHostGetDevicePointer and dw::core::makeDeviceSpan are thread safe
        // and no memory allocation / free
        T* devPtr = nullptr;
#ifndef DW_IS_SAFETY // TODO(dwsafety): DRIV-8190
        // Creates a DeviceSpan that maps the span.
        DW_CHECK_CUDA_ERROR(::cudaHostGetDevicePointer(&devPtr, get().data(), 0));
#endif
        return dw::core::makeDeviceSpan(devPtr, get().size());
    }
};

/// create UniqueSpan and allocate host memory
template <typename T, size_t DIMS>
auto makeUniqueMappedSpan(const Array<size_t, DIMS>& size) -> UniqueMappedSpan<T, DIMS>
{
    // TODO(dwplc): FP. totalSize is modified within the if statement.
    // coverity[autosar_cpp14_a7_1_1_violation]
    size_t totalSize = size[0];
    if (DIMS > 1)
    {
        for (size_t i = 1; i < DIMS; i++)
        {
            totalSize = safeMul(totalSize, size[i]).value();
        }
    }
    auto ptr = makeUniqueMapped<T[]>(totalSize);
    return UniqueMappedSpan<T, DIMS>({ptr.release(), size});
}

template <typename T, typename... Sizes>
auto makeUniqueMappedSpan(size_t const size0, Sizes... sizes) -> UniqueMappedSpan<T, 1 + sizeof...(Sizes)>
{
    return makeUniqueMappedSpan<T>(Array<size_t, 1 + sizeof...(Sizes)>{size0, sizes...});
}

/// UniqueDeviceSpan (GPU device memory)
template <typename T, size_t DIMS_>
class UniqueDeviceSpan
{
public:
    /// Type of contained elements
    using ElementType = T;

    /// Number of dimensions for indexed element access
    static constexpr size_t DIMS = DIMS_;

    UniqueDeviceSpan()
        : UniqueDeviceSpan(span<T, DIMS>{nullptr})
    {
    }

    /// Move constructor
    UniqueDeviceSpan(UniqueDeviceSpan&& other)
        // TODO(dwplc): FP - this is using move semantics
        // coverity[autosar_cpp14_a12_8_4_violation]
        : m_span(std::move(other.m_span))
    {
        static_assert(sizeof(UniqueDeviceSpan) == sizeof(DeviceSpan<T, DIMS>), "UniqueDeviceSpan shall not add memory overhead");
        other.m_span = DeviceSpan<T, DIMS>{nullptr};
    }

    /// Move operator
    auto operator=(UniqueDeviceSpan&& other) -> UniqueDeviceSpan&
    {
        reset();

        m_span       = std::move(other.m_span);
        other.m_span = DeviceSpan<T, DIMS>{nullptr};
        return *this;
    }

    UniqueDeviceSpan(const UniqueDeviceSpan&) = delete;
    auto operator=(const UniqueDeviceSpan&) -> UniqueDeviceSpan& = delete;

    /// Construction with span.
    explicit UniqueDeviceSpan(span<T, DIMS> const spanIn)
        : m_span(spanIn)
    {
    }

    /// Destructor
    /// Frees allocated cuda memory.
    ~UniqueDeviceSpan()
    {
        reset();
    }

    /// Frees allocated cuda memory.
    void reset()
    {
        if (m_span.get().data() != nullptr)
        {
            auto const res = cudaFree(m_span.get().data());
            if (res != cudaSuccess)
            {
                // Note: cannot throw on a deleter, log error instead
                LOGSTREAM_ERROR(nullptr) << "Error freeing cuda pointer" << Logger::State::endl;
            }
        }
    }

    /// Get device span
    auto get() const -> DeviceSpan<T, DIMS>
    {
        return m_span;
    }

    /// Get device span
    auto operator-> () const -> const DeviceSpan<T, DIMS>* { return &m_span; }

    /// Returns true if UniqueDeviceSpan is empty.
    bool empty() const
    {
        return m_span.empty();
    }

    /// Releases ownership of DeviceSpan and
    /// returns it.
    auto release() -> DeviceSpan<T, DIMS>
    {
        auto temp = m_span;
        m_span    = DeviceSpan<T, DIMS>{nullptr};
        return temp;
    }

private:
    DeviceSpan<T, DIMS> m_span;
};

/// create UniqueDeviceSpan and allocate device memory
template <typename T>
auto makeUniqueDeviceSpan(size_t const size0) -> UniqueDeviceSpan<T, 1>
{
    auto ptr = makeUniqueDevice<T[]>(size0);
    return UniqueDeviceSpan<T, 1>(span<T, 1>{ptr.release().get(), size0});
}

namespace detail
{

template <typename T, size_t DIMS, typename std::enable_if<DIMS != 2, bool>::type = true>
auto makeUniqueDeviceSpanCuda2DMalloc(const Array<size_t, DIMS>& size) -> UniqueDeviceSpan<T, DIMS>
{
    size_t totalHeight = size[1];
    for (size_t i = 2; i < size.size(); i++)
    {
        totalHeight = safeMul(totalHeight, size[i]).value();
    }
    auto ptr = makeUniqueDevice<T[]>(size[0], totalHeight);
    return UniqueDeviceSpan<T, DIMS>({ptr.release().get().getPtr(), size, ptr.get().getStrideBytes()});
}

// SFINAE catch needed to satisfy autosar M0-1-9 (unreachable code paths)
template <typename T, size_t DIMS, typename std::enable_if<DIMS == 2, bool>::type = true>
auto makeUniqueDeviceSpanCuda2DMalloc(const Array<size_t, DIMS>& size) -> UniqueDeviceSpan<T, DIMS>
{
    size_t totalHeight = size[1];
    auto ptr           = makeUniqueDevice<T[]>(size[0], totalHeight);
    return UniqueDeviceSpan<T, DIMS>({ptr.release().get().getPtr(), size, ptr.get().getStrideBytes()});
}

} // namespace detail

/// Using cudaMalloc2D forces the pitch to be at least 512-bytes aligned
/// This is necessary when using the GPU texture unit, but might be overkill depending on the access pattern.
template <typename T, bool useCudaMalloc2D = true, size_t DIMS>
auto makeUniqueDeviceSpan(const Array<size_t, DIMS>& size) -> UniqueDeviceSpan<T, DIMS>
{
    if (useCudaMalloc2D)
    {
        return detail::makeUniqueDeviceSpanCuda2DMalloc<T, DIMS>(size);
    }
    else
    {
        size_t totalSize = size[0];
        for (size_t i = 1; i < size.size(); i++)
        {
            totalSize = safeMul(totalSize, size[i]).value();
        }
        auto ptr = makeUniqueDevice<T[]>(totalSize);
        return UniqueDeviceSpan<T, DIMS>({ptr.release().get(), size});
    }
}

template <typename T, bool useCudaMalloc2D = true, typename... Sizes>
auto makeUniqueDeviceSpan(size_t const size0, Sizes... sizes) -> UniqueDeviceSpan<T, 1 + sizeof...(Sizes)>
{
    return makeUniqueDeviceSpan<T, useCudaMalloc2D, 1 + sizeof...(Sizes)>(Array<size_t, 1 + sizeof...(Sizes)>{size0, sizes...});
}

/// Asynchronous memcopy from CPU memory into cuda device memory
template <class T, size_t N>
auto cloneToDeviceAsync(span<T, N> src, cudaStream_t stream = nullptr) -> UniqueDeviceSpan<typename std::remove_const<T>::type, N>
{
    if (src.empty())
    {
        return {};
    }

    using Tout = typename std::remove_const<T>::type;
    auto res   = makeUniqueDeviceSpan<Tout>(src.size());
    memcpyAsync(res.get(), src, stream);
    return res;
}

/// Asynchronous memcopy from cuda device memory into cuda device memory
template <class T, size_t N>
auto cloneToDeviceAsync(DeviceSpan<T, N> src, cudaStream_t stream = nullptr) -> UniqueDeviceSpan<typename std::remove_const<T>::type, N>
{
    if (src.empty())
    {
        return {};
    }

    using Tout = typename std::remove_const<T>::type;
    auto res   = makeUniqueDeviceSpan<Tout>(src.size());
    memcpyAsync(res.get(), src, stream);
    return res;
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

/// unified interface to declare/create unique spans with CUDA memory.
template <>
struct TUniqueSpanHelper<UniqueSpanHelperType, UniqueSpanHelperType::CUDA>
{
    template <typename T, size_t DIMS>
    using TSpan = DeviceSpan<T, DIMS>;

    template <typename T, size_t DIMS = 1>
    using TUniqueSpan = UniqueDeviceSpan<T, DIMS>;

    /// create a DeviceSpan by ptr and size with unified interface
    template <typename T>
    static TSpan<T, 1> makeSpan(T* ptr, size_t s)
    {
        return dw::core::makeDeviceSpan(ptr, s);
    }

    /// create a DeviceSpan from span with unified interface
    template <typename T, size_t DIMS = 1>
    static TSpan<T, DIMS> makeSpan(span<T, DIMS> s)
    {
        return dw::core::makeDeviceSpan(s);
    }

    /// get raw pointer from a DeviceSpan with unified interface
    template <typename T>
    static T* getRawPtr(TSpan<T, 1> s)
    {
        return s.get().data();
    }

    /// equvalent to core::makeUniqueDeviceSpan but using unified interface
    template <typename T>
    static auto makeUniqueSpan(size_t const size0) -> TUniqueSpan<T, 1>
    {
        return dw::core::makeUniqueDeviceSpan<T>(size0);
    }

    /// equvalent to core::makeUniqueDeviceSpan but using unified interface
    template <typename T, bool useCudaMalloc2D = true, size_t DIMS>
    static auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> TUniqueSpan<T, DIMS>
    {
        return dw::core::makeUniqueDeviceSpan<T, useCudaMalloc2D>(size);
    }

    /// equvalent to core::makeUniqueDeviceSpan but using unified interface
    template <typename T, bool useCudaMalloc2D = true, typename... Sizes>
    static auto makeUniqueSpan(size_t const size0, Sizes... sizes) -> TUniqueSpan<T, 1 + sizeof...(Sizes)>
    {
        return dw::core::makeUniqueDeviceSpan<T, useCudaMalloc2D>(size0, sizes...);
    }
};

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

/// unified interface to declare/create unique spans with pinned memory.
template <>
struct TUniqueSpanHelper<UniqueSpanHelperType, UniqueSpanHelperType::Pinned> : public UniqueSpanHelperHost
{
    template <typename T, size_t DIMS = 1>
    using TSpan = UniqueSpanHelperHost::TSpan<T, DIMS>;

    template <typename T, size_t DIMS = 1>
    using TUniqueSpan = UniquePinnedSpan<T, DIMS>;

    /// equvalent to core::makeUniquePinnedSpan but using unified interface
    template <typename T, size_t DIMS>
    static auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> TUniqueSpan<T, DIMS>
    {
        return dw::core::makeUniquePinnedSpan<T>(size);
    }

    /// equvalent to core::makeUniquePinnedSpan but using unified interface
    template <typename T, typename... Sizes>
    static auto makeUniqueSpan(size_t const size0, Sizes... sizes) -> TUniqueSpan<T, 1 + sizeof...(Sizes)>
    {
        return dw::core::makeUniquePinnedSpan<T>(size0, sizes...);
    }
};

/// unified interface to declare/create unique spans with mapped memory.
template <>
struct TUniqueSpanHelper<UniqueSpanHelperType, UniqueSpanHelperType::Mapped> : public UniqueSpanHelperHost
{
    template <typename T, size_t DIMS = 1>
    using TSpan = UniqueSpanHelperHost::TSpan<T, DIMS>;

    template <typename T, size_t DIMS = 1>
    using TUniqueSpan = UniqueMappedSpan<T, DIMS>;

    /// equvalent to core::makeUniqueMappedSpan but using unified interface
    template <typename T, size_t DIMS>
    static auto makeUniqueSpan(const Array<size_t, DIMS>& size) -> TUniqueSpan<T, DIMS>
    {
        return dw::core::makeUniqueMappedSpan<T>(size);
    }

    /// equvalent to core::makeUniqueMappedSpan but using unified interface
    template <typename T, typename... Sizes>
    static auto makeUniqueSpan(size_t const size0, Sizes... sizes) -> TUniqueSpan<T, 1 + sizeof...(Sizes)>
    {
        return dw::core::makeUniqueMappedSpan<T>(size0, sizes...);
    }
};

/// alias to TUniqueSpanHelper for easy usage
/// unified interface to declare/create unique spans in different memory space.
template <UniqueSpanHelperType TYPE>
using UniqueSpanHelper = TUniqueSpanHelper<decltype(TYPE), TYPE>;
} // namespace core

} // namespace dw

#endif
