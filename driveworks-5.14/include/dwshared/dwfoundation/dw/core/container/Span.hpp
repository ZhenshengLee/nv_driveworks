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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_SPAN_HPP_
#define DWSHARED_CORE_SPAN_HPP_

#include "Array.hpp"

#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/ConfigChecks.h>
#include <dwshared/dwfoundation/dw/core/safety/Safety.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

// TODO(dwplc): FP - ## is only used once per declaration, and Type is template argument which can't be enclosed in parentheses
// coverity[autosar_cpp14_m16_0_6_violation]
// coverity[autosar_cpp14_m16_3_1_violation]
#define SPAN_DEFINES(type, suffix)                     \
    using span1##suffix         = span<type, 1>;       \
    using span2##suffix         = span<type, 2>;       \
    using span3##suffix         = span<type, 3>;       \
    using span1##suffix##_const = span<type const, 1>; \
    using span2##suffix##_const = span<type const, 2>; \
    using span3##suffix##_const = span<type const, 3>;

namespace dw
{

namespace core
{

namespace detail_span
{
/// Check that a span with potentially-different T can be compared to.
/// This is used to conditionally-enable things like operator== for different span types (for example:
/// const v.s. non-const element type).
template <typename U, typename V>
struct IsSpanComparable : meta::Conjunction<std::is_same<std::remove_const_t<U>, std::remove_const_t<V>>>
{
};

// -----------------------------------------------------------------------------
CUDA_BOTH_INLINE void expandDimensions(size_t& size0, size_t const totalSize)
{
    if (size0 == 0)
    {
        size0 = totalSize;
    }
    else
    {
        if (size0 != totalSize)
        {
            assertException<InvalidArgumentException>("span: reshape size doesn't match element count");
        }
    }
}

// -----------------------------------------------------------------------------
CUDA_BOTH_INLINE void expandDimensions(size_t& size0, size_t& size1, size_t const totalSize)
{
    if ((size0 == 0) && (size1 == 0))
    {
        if (totalSize == 0)
        {
            return;
        }
        else
        {
            assertException<InvalidArgumentException>("span::reshape: at least one size must be valid");
        }
    }
    else if (size0 == 0)
    {
        // Expand the missing dimension
        size0 = totalSize / size1;
    }
    else
    {
        // Expand the missing dimension
        if (size1 == 0)
        {
            size1 = totalSize / size0;
        }
    }

    // TODO(dwplc): FP - SafetlyResult inherits OptionalBase ctors which have __device__ specifier.
    // coverity[cuda_share_function_indirect]
    if (safeMul(size0, size1).value() != totalSize)
    {
        assertException<InvalidArgumentException>("span::reshape: requested size does not match the size of the original span");
    }
}

// -----------------------------------------------------------------------------
CUDA_BOTH_INLINE void expandDimensions(size_t& size0, size_t& size1, size_t& size2, size_t const totalSize)
{
    uint8_t zeroCount{0};
    if (size0 == 0)
    {
        zeroCount++;
    }
    if (size1 == 0)
    {
        zeroCount++;
    }
    if (size2 == 0)
    {
        zeroCount++;
    }

    if (zeroCount > 1)
    {
        assertException<InvalidArgumentException>("span::reshape: only one size can be zero");
    }
    else
    {
        // Expand the missing dimension
        if (size0 == 0)
        {
            size0 = totalSize / (size1 * size2);
        }
        else if (size1 == 0)
        {
            size1 = totalSize / (size0 * size2);
        }
        else
        {
            if (size2 == 0)
            {
                size2 = totalSize / (size1 * size0);
            }
        }

        // TODO(dwplc): FP - SafetlyResult inherits OptionalBase ctors which have __device__ specifier.
        // coverity[cuda_share_function_indirect]
        if (safeMul(safeMul(size0, size1).value(), size2).value() != totalSize)
        {
            assertException<InvalidArgumentException>("span::reshape: requested size does not match the size of the original span");
        }
    }
}
}

/// span - Holds a pointer+size for a multi-dimensional array
///      - Enforces bounds checks
///
/// A span can be passed between functions and components and ensures that
/// the size of the array is always available and can be checked for safe memory access.
/// All accesses are bounds checked.
///
/// span implies no memory ownership. For ownership see the UniqueSpan<> classes.
///
/// span can be used in cpu or gpu:\<linebreak\>
///     - ok: a span in CPU that points to CPU memory
///     - ok: a span in GPU that points to GPU memory
///     - not ok: a span in CPU that points to GPU memory. See the DeviceSpan class for details
///
/// The DIMS parameter represents the array dimension:\<linebreak\>
///     1 - linear array, all items contiguous in memory
///     2 - 2D array, dim0 has all items contiguous, dim1 has a pitch in bytes
///         to allow for memory alignment
///     3 - 3D array, same as 2D array but with another dimension where all slices are contiguous
template <typename T, size_t DIMS = 1>
class span;

SPAN_DEFINES(uint8_t, ub);
SPAN_DEFINES(uint16_t, us);
SPAN_DEFINES(uint32_t, ui);
SPAN_DEFINES(int8_t, b);
SPAN_DEFINES(int16_t, s);
SPAN_DEFINES(int32_t, i);
SPAN_DEFINES(size_t, sz);
SPAN_DEFINES(float32_t, f);
SPAN_DEFINES(float64_t, d);

} // namespace core
} // namespace dw

#undef SPAN_DEFINES

namespace dw
{

namespace core
{

/// Forward define of DeviceSpan
template <typename T, size_t DIMS>
class DeviceSpan;

namespace traits
{
/// Traits class to determine if a type is a span
template <typename>
struct is_span : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T, size_t DIMS>
struct is_span<span<T, DIMS>> : std::true_type
{
};

template <typename>
struct is_DeviceSpan : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T, size_t DIMS>
struct is_DeviceSpan<DeviceSpan<T, DIMS>> : std::true_type
{
};
}

/// 1D specialization
/// Doesn't use pitch since all items are continuous
template <typename T>
class span<T, 1>
{
public:
    using ElementType = T; ///< Type of elements in span

    // same as STL and other DW containers
    using value_type = T;                   ///< Type of elements in span
    using size_type  = size_t;              ///< Type of size
    using iterator   = T*;                  ///< Type of iterator
    using SizeArray  = Array<size_type, 1>; ///< Type of array with size per dimension

    /// Number of dimsions
    static constexpr size_t DIMS = 1;

    /// Marker to represent end of span in a dimension
    static constexpr size_t NPOS{std::numeric_limits<size_t>::max()};

    // -----------------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    /// Default constructor
    CUDA_BOTH_INLINE constexpr span() // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : span(nullptr)
    {
    }

    /// Empty constructor
    // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init, google-explicit-constructor)
    CUDA_BOTH_INLINE constexpr span(std::nullptr_t)
        : span(static_cast<T*>(nullptr), static_cast<size_t>(0))
    {
    }

    /// Constructor from array pointer and element count
    CUDA_BOTH_INLINE constexpr span(T* const ptr, size_t const count) // clang-tidy NOLINT(google-explicit-constructor)
        : m_ptr(ptr)
        , m_size(count)
    {
    }

    // -----------------------------------------------------------------------------
    /// Constructor from array begin and end pointers
    CUDA_BOTH_INLINE span(T* const beginIn, T const* const endIn)
        : m_ptr(beginIn)
        , m_size(narrow<size_t>(std::abs(endIn - beginIn)))
    {
#if DW_RUNTIME_CHECKS()
        if (beginIn > endIn)
        {
            assertException<InvalidArgumentException>("span: end must be greater than begin");
        }
#endif
    }

    // -----------------------------------------------------------------------------
    /// Constructor from pointer and size per dimension array (Array of size one for 1-dimensional span)
    CUDA_BOTH_INLINE span(T* const ptr, Array<size_t, 1> count) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : span(ptr, count[0])
    {
    }

    // -----------------------------------------------------------------------------
    /// Conversion to const T
    CUDA_BOTH_INLINE auto toConst() const -> span<const T, 1>
    {
        return span<const T, 1>(m_ptr, m_size);
    }

    // -----------------------------------------------------------------------------
    /// Dummy template so this doesn't get defined for const spans (avoid nvcc warning)
    template <class T_ = T, typename = typename std::enable_if<std::is_same<T_, T>::value>::type>
    // TODO(dwplc): RFD - It is safe and following C++ semantics to implicitly convert from non-const to const span.
    // coverity[autosar_cpp14_a13_5_2_violation]
    CUDA_BOTH_INLINE operator span<const T_, 1>() const // clang-tidy NOLINT
    {
        return toConst();
    }

    // -----------------------------------------------------------------------------
    /// Construct span from native array
    template <class U, size_t N>
    CUDA_BOTH_INLINE constexpr span(U (&arr)[N]) // clang-tidy NOLINT(google-explicit-constructor)
        : m_ptr(&arr[0])
        , m_size(N)
    {
    }

    // -----------------------------------------------------------------------------
    /// Conversion from other types that have a data()+size() functions
    /// SFINAE enable only if Cont has a data() member function
    /// Disabled for span types
    template <class Cont,
              class = decltype(std::declval<Cont>().data()),
              class = typename std::enable_if<!traits::is_span<typename std::remove_const<Cont>::type>::value && !traits::is_DeviceSpan<typename std::remove_const<Cont>::type>::value>::type>
    constexpr span(Cont& cont) // clang-tidy NOLINT(google-explicit-constructor)
        : m_ptr(cont.data())
        , m_size(cont.size())
    {
    }

    // -----------------------------------------------------------------------------
    span(const span& other) = default;      ///< Copy constructor
    span(span& other)       = default;      ///< Copy constructor
    span& operator=(const span&) = default; ///< Copy operator
    ~span()                      = default; ///< Destructor

    // -----------------------------------------------------------------------------
    /// Get pointer to data.
    CUDA_BOTH_INLINE auto data() const -> T* { return m_ptr; }

    // -----------------------------------------------------------------------------
    CUDA_BOTH_INLINE auto begin() const -> T* { return m_ptr; }        ///< Get begin iterator
    CUDA_BOTH_INLINE auto end() const -> T* { return m_ptr + m_size; } ///< Get end iterator

    // -----------------------------------------------------------------------------
    CUDA_BOTH_INLINE auto front() const -> T& { return at(0U); } ///< Get first element

    /// Get last element
    CUDA_BOTH_INLINE auto back() const -> T&
    {
        if (size() == 0U)
        {
            assertException<BufferEmptyException>("span::back: access empty span");
        }

        return at(size() - 1U);
    }

    // -----------------------------------------------------------------------------
    /// indexed access
    CUDA_BOTH_INLINE auto at(size_t const x) const -> T&
    {
        if (x >= m_size)
        {
#if !defined(__CUDACC__)
            throwSpanIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("span::at: access out of bounds");
#endif
        }

        // TODO(dwplc): FP - Classes that mimic containers do not violate this rule
        // coverity[autosar_cpp14_a9_3_1_violation]
        return m_ptr[x];
    }

    // -----------------------------------------------------------------------------
    CUDA_BOTH_INLINE auto operator[](size_t const x) const -> T& { return at(x); } ///< Get element at index
    CUDA_BOTH_INLINE auto operator()(size_t const x) const -> T& { return at(x); } ///< Get element at index

    // -----------------------------------------------------------------------------
    /// return size in elements
    constexpr CUDA_BOTH_INLINE size_t size() const noexcept { return m_size; } ///< Get size of span

    // -----------------------------------------------------------------------------
    /// Add a dummy index argument to match the interface of the multidimensional spans
    constexpr CUDA_BOTH_INLINE size_t size(size_t const dim) const noexcept
    {
        if (dim == 0U)
        {
            return m_size;
        }
        else
        {
            return 1U;
        }
    }

    // -----------------------------------------------------------------------------
    /// return size in bytes
    constexpr CUDA_BOTH_INLINE size_t size_bytes() const noexcept
    {
        return safeMul(m_size, sizeof(T)).value_or(core::numeric_limits<size_t>::max());
    }

    // -----------------------------------------------------------------------------
    /// Return true if span is empty.
    constexpr CUDA_BOTH_INLINE bool empty() const noexcept { return m_size == 0U; }

    // -----------------------------------------------------------------------------
    /// a span is implicitly "true" if it's not empty
    explicit constexpr CUDA_BOTH_INLINE operator bool() const noexcept
    {
        return !empty();
    }

    // -----------------------------------------------------------------------------
    /// Equality operator
    template <typename U, typename V>
    CUDA_BOTH_INLINE friend std::enable_if_t<detail_span::IsSpanComparable<U, V>::value, bool>
    operator==(const span<U, 1>& lhs, const span<V, 1>& rhs) noexcept;

    // -----------------------------------------------------------------------------
    /// performs a deep-by-value comparison using T::operator==
    CUDA_BOTH_INLINE bool sameAs(const span& other) const
    {
        if (m_size != other.size())
        {
            return false;
        }

        // lazy check on ptr now size check above passed already
        if (m_ptr == other.data())
        {
            return true;
        }

        for (size_t i{0}; i < m_size; i++)
        {
            if ((*this)[i] != other[i])
            {
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------------
    /// Inequality operator
    template <typename U, typename V>
    CUDA_BOTH_INLINE friend std::enable_if_t<detail_span::IsSpanComparable<U, V>::value, bool>
    operator!=(const span<U, 1>& lhs, const span<V, 1>& rhs) noexcept;

    // -----------------------------------------------------------------------------
    /// Get subspan
    /// @param offset   start of subspan
    /// @param size_    size of subspan
    CUDA_BOTH_INLINE auto subspan(size_t const offset, size_t size_ = NPOS) const -> span
    {
        if (size_ == NPOS)
        {
            size_ = m_size - offset;
        }

        if (safeAdd(offset, size_) > m_size)
        {
#if !defined(__CUDACC__)
            throwSpanIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("span::subspan: access out of bounds");
#endif
        }

        return span(m_ptr + offset, size_);
    }

    // -----------------------------------------------------------------------------
    /// Get subspan
    /// @param offset_   start of subspan (as array with 1 element)
    /// @param size_     size of subspan  (as array with 1 element)
    CUDA_BOTH_INLINE auto subspan(SizeArray const& offset_, SizeArray const& size_ = SizeArray{NPOS}) const -> span
    {
        return subspan(offset_[0], size_[0]);
    }

    // -----------------------------------------------------------------------------
    /// Dereference returns reference to first element
    CUDA_BOTH_INLINE auto operator*() const -> T&
    {
        return (*this)[0];
    }

    // -----------------------------------------------------------------------------
    /// Returns a subspan with offset items
    CUDA_BOTH_INLINE auto operator+(size_t const items) const -> span
    {
        return subspan(items);
    }

    // -----------------------------------------------------------------------------
    /// Updates this span to a subspan with offset items
    CUDA_BOTH_INLINE auto operator+=(size_t const items) -> span&
    {
        *this = (*this) + items;
        return *this;
    }

    // -----------------------------------------------------------------------------
    /// Pre-fix increment
    CUDA_BOTH_INLINE auto operator++() -> span&
    {
        (*this) = (*this) + 1;
        return (*this);
    }

    // -----------------------------------------------------------------------------
    /// Post-fix increment
    CUDA_BOTH_INLINE auto operator++(int32_t) -> span
    {
        span const temp{*this};
        ++(*this);
        return temp;
    }

    // -----------------------------------------------------------------------------
    /// Reshapes the 1D span into a 2D span
    /// One of the sizes can be 0 to force the resulting span to cover the entire memory area according to the other size
    CUDA_BOTH_INLINE auto reshape(size_t const& size0, size_t const& size1) -> span<T, 2>
    {
        return span<T, 2>(*this, size0, size1);
    }

    // -----------------------------------------------------------------------------
    /// Reshapes the 1D span into a 3D span
    /// One of the sizes can be 0 to force the resulting span to cover the entire memory area according to the other size
    CUDA_BOTH_INLINE auto reshape(size_t const& size0, size_t const& size1, size_t const& size2) -> span<T, 3>
    {
        return span<T, 3>(*this, size0, size1, size2);
    }

    // -----------------------------------------------------------------------------
    /// byte type
    using TByte = typename dw::core::copy_const<uint8_t, T>::type;

    /// Convert to a span of bytes
    CUDA_BOTH_INLINE auto toByteSpan() const -> span<TByte>
    {
        return span<TByte>(reinterpret_cast<TByte*>(m_ptr), size_bytes()); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    // -----------------------------------------------------------------------------
    /// Returns a span of requested type
    template <class T2>
    CUDA_BOTH_INLINE auto reinterpretCast() const -> span<T2>
    {
        static_assert(std::is_same<typename std::remove_const<T>::type, uint8_t>::value, "Can only reinterpret from uint8_t spans");
        static_assert(std::is_const<T>::value == std::is_const<T>::value, "Cannot change constness");

        if (!isAligned(m_ptr, std::alignment_of<T2>::value))
        {
            core::assertException<BadAlignmentException>("span: bad alignment when reinterpreting");
        }

        size_t newSize = m_size / sizeof(T2);
        if (newSize * sizeof(T2) != m_size)
        {
            core::assertException<InvalidArgumentException>("span: sizes don't match when reinterpreting");
        }

        return span<T2>(reinterpret_cast<T2*>(m_ptr), newSize); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    // -----------------------------------------------------------------------------
    /// Applies a function to references of each element (similar to std::for_each()).
    /// Returns reference to object to allow "chaining" of operations on the span
    template <class UnaryFunction>
    CUDA_BOTH_INLINE auto for_each(UnaryFunction&& unaryFunction) -> span&
    {
        for (auto& element : *this)
        {
            std::forward<UnaryFunction>(unaryFunction)(element);
        }
        return *this;
    }

    /// Applies a function to references of each element (similar to std::for_each()), but passes the
    /// element's index as first argument.
    /// Returns reference to object to allow "chaining" of operations on the span
    template <class BinaryFunction>
    CUDA_BOTH_INLINE auto for_each_indexed(BinaryFunction&& binaryFunction) -> span&
    {
        auto index = size_t{};
        for (auto& element : *this)
        {
            std::forward<BinaryFunction>(binaryFunction)(index++, element);
        }
        return *this;
    }

private:
    T* m_ptr;      //!< unmanaged pointer copy
    size_t m_size; //!< total size in elements
};

/// Base class for spans of 2D or higher
/// Implements templated access and bounds checking for any dimension
template <typename T, size_t DIMS_>
class BaseSpan
{
public:
    /// Type of elements in span
    using ElementType = T;

    using TByte = typename dw::core::copy_const<std::uint8_t, T>::type; // They byte type to do pointer arithmetic (const or not)

    /// Size per span dimension
    using SizeArray = Array<size_t, DIMS_>;

    /// Number of span dimensions
    static constexpr size_t DIMS{DIMS_};

    /// Marker to represent end of span in a dimension
    static constexpr size_t NPOS{std::numeric_limits<size_t>::max()};

    /// Default constructor
    CUDA_BOTH_INLINE BaseSpan() // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : BaseSpan(nullptr, SizeArray{}, 0U)
    {
    }

    CUDA_BOTH_INLINE BaseSpan(BaseSpan const& other) = default;            ///< Copy constructor
    CUDA_BOTH_INLINE BaseSpan& operator=(const BaseSpan& other) = default; ///< Copy operator
    CUDA_BOTH_INLINE ~BaseSpan()                                = default; ///< Destructor

    // -----------------------------------------------------------------------------
    /// Constructor of multidimensional base class
    CUDA_BOTH_INLINE BaseSpan(T* const ptrIn, SizeArray const& sizeIn, size_t const pitchBytesIn = 0) noexcept
        : m_ptr(ptrIn)
        , m_pitchBytes(0U)
        , m_size(sizeIn)
    {

        bool bOverflow{true};
        if (core::numeric_limits<size_t>::max() / sizeof(T) >= sizeIn[0])
        {
            bOverflow    = false;
            m_pitchBytes = pitchBytesIn >= (sizeIn[0] * sizeof(T)) ? pitchBytesIn : (sizeIn[0] * sizeof(T));

#ifndef __CUDA_ARCH__
            try
#endif
            {
                size_t count{m_pitchBytes / sizeof(T)};
                for (size_t i{1}; i < DIMS; i++)
                {
                    count = safeMul(count, m_size[i]).value();
                }

                if (core::numeric_limits<size_t>::max() / sizeof(T) < count)
                {
                    bOverflow = true;
                }
            }
#ifndef __CUDA_ARCH__
            catch (core::BadSafetyResultAccess&)
            {
                bOverflow = true;
            }
            catch (core::BadOptionalAccess&)
            {
                bOverflow = true;
            }
#endif
        }

        if (bOverflow)
        {
// log error only so we can make ctor as noexcept and defer throwing when accessing later
#ifdef __CUDA_ARCH__
            printf("BaseSpan(): Overflow error");
#else
            static char8_t LOG_TAG[]{"BaseSpan"};
            DW_LOGE << "Overflow error" << Logger::State::endl;
#endif

            m_ptr  = nullptr;
            m_size = SizeArray{};
        }
    }

    // -----------------------------------------------------------------------------
    /// Conversion to const T
    CUDA_BOTH_INLINE auto toConst() const -> span<const T, DIMS>
    {
        return span<const T, DIMS>(m_ptr, m_size, m_pitchBytes);
    }

    // -----------------------------------------------------------------------------
    /// Dummy template so this doesn't get defined for const spans (avoid nvcc warning)
    template <class T_ = T, typename = typename std::enable_if<std::is_same<T_, T>::value>::type>
    // TODO(dwplc): RFD - It is safe and following C++ semantics to implicitly convert from non-const to const span.
    // coverity[autosar_cpp14_a13_5_2_violation]
    CUDA_BOTH_INLINE operator span<const T_, DIMS_>() const // clang-tidy NOLINT
    {
        return toConst();
    }

    // -----------------------------------------------------------------------------
    /// Get pointer to data
    CUDA_BOTH_INLINE auto data() const -> T* { return m_ptr; }

    // -----------------------------------------------------------------------------
    /// Indexed element access for 2-dimensional spans.
    /// Returns a 1-dimensional span.
    template <size_t DIMS2 = DIMS_> // Make a dummy template so enable_if works
    CUDA_BOTH_INLINE auto operator[](typename std::enable_if<DIMS2 == 2, size_t>::type const idx) const -> span<T, 1>
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= m_size[1])
        {
#if !defined(__CUDACC__)
            throwSpanIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("span::operator[]: access out of bounds");
#endif
        }
#endif

        // TODO(dwplc): RFD - needed for support for Span of float32_t
        // coverity[autosar_cpp14_m3_9_3_violation]
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const newPtr = reinterpret_cast<T*>(reinterpret_cast<TByte*>(m_ptr) + m_pitchBytes * idx); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return span<T, 1>(newPtr, m_size[0]);
    }

    // -----------------------------------------------------------------------------
    /// Indexed element access for spans with dimension > 2.
    /// Returns a span with one less dimension.
    template <size_t DIMS2 = DIMS_> // Make a dummy template so enable_if works
    CUDA_BOTH_INLINE auto operator[](typename std::enable_if<DIMS2 != 2, size_t>::type const idx) const -> span<T, DIMS_ - 1>
    {
        return span<T, DIMS_ - 1>(*this, idx);
    }

    // -----------------------------------------------------------------------------
    /// Indexed element access with out of bounds check.
    /// Throws an out of bounds exception if index is out of bounds.
    CUDA_BOTH_INLINE auto at(const Array<size_t, DIMS_>& pos) const -> T&
    {
#if DW_RUNTIME_CHECKS()
        checkAccess(pos, "span::at: access out of bounds");
#endif

        return *pointerToUnchecked(pos);
    }

    // -----------------------------------------------------------------------------
    /// return array with size for each dimension
    constexpr CUDA_BOTH_INLINE auto size() const noexcept -> SizeArray const& { return m_size; }

    /// return size of requestied dimension
    CUDA_BOTH_INLINE size_t size(size_t const i) const { return m_size[i]; }

    /// return total number of elements in span
    CUDA_BOTH_INLINE size_t elementCount() const
    {
        size_t count = 1;
        for (size_t i = 0; i < DIMS_; i++)
        {
            count = safeMul(count, m_size[i]).value();
        }
        return count;
    }

    // -----------------------------------------------------------------------------
    CUDA_BOTH_INLINE size_t width() const { return m_size[0]; }  ///< return size of dimension 0
    CUDA_BOTH_INLINE size_t height() const { return m_size[1]; } ///< return size of dimension 1

    // -----------------------------------------------------------------------------
    /// return size in bytes
    CUDA_BOTH_INLINE size_t size_bytes() const
    {
        size_t res = m_pitchBytes;
        for (size_t i = 1; i < DIMS; i++)
        {
            res = safeMul(res, m_size[i]).value();
        }
        return res;
    }

    // -----------------------------------------------------------------------------
    /// return pitch in bytes
    CUDA_BOTH_INLINE size_t pitch_bytes() const { return m_pitchBytes; }

    // -----------------------------------------------------------------------------
    /// return true if empty
    CUDA_BOTH_INLINE size_t empty() const { return size_bytes() == 0; }

    // -----------------------------------------------------------------------------
    /// a span is implicitly "true" if it's not empty
    CUDA_BOTH_INLINE explicit operator bool() const
    {
        return !empty();
    }

    // -----------------------------------------------------------------------------
    /// Returns true if the pitch includes extra padding besides the size of the elements
    CUDA_BOTH_INLINE bool hasPadding() const
    {
        return m_pitchBytes != safeMul(sizeof(T), m_size[0]).value();
    }

    // -----------------------------------------------------------------------------
    /// performs a deep-by-value comparison using T::operator==
    CUDA_BOTH_INLINE bool sameAs(const span<T, DIMS_>& other) const
    {
        // check size of all DIMs first
        for (size_t i = 0; i < DIMS_; i++)
        {
            if (m_size[i] != other.size(i))
            {
                return false;
            }
        }

        // lazy check on ptr now size check above passed already
        if (m_ptr == other.data())
        {
            return true;
        }

        size_t totalCount = this->elementCount();
        const span<T, 1> flatten      = this->reshape(totalCount);
        const span<T, 1> flattenOther = other.reshape(totalCount);
        return flatten.sameAs(flattenOther);
    }

    // -----------------------------------------------------------------------------
    /// Reshapes into a 1D span
    /// Size can be 0 to have it fill automatically
    /// Throws if there is padding in the pitch
    CUDA_BOTH_INLINE auto reshape(size_t size0) const -> span<T, 1>
    {
        // Check that there is no padding
        if (hasPadding())
        {
            assertException<InvalidArgumentException>("span: cannot reshape to 1D span when there is padding");
        }

        // checks validity of reshape sizes
        detail_span::expandDimensions(size0, elementCount());
        return span<T, 1>(m_ptr, size0);
    }

    // -----------------------------------------------------------------------------
    /// Reshapes into a 2D span
    /// One of the sizes can be 0 to have it fill automatically
    CUDA_BOTH_INLINE auto reshape(size_t size0, size_t size1) const -> span<T, 2>
    {
        return span<T, 2>(*this, size0, size1);
    }

    // -----------------------------------------------------------------------------
    /// Reshapes into a 3D span
    /// One of the sizes can be 0 to have it fill automatically
    CUDA_BOTH_INLINE auto reshape(size_t const& size0, size_t const& size1, size_t const& size2) const -> span<T, 3>
    {
        return span<T, 3>(*this, size0, size1, size2);
    }

    // -----------------------------------------------------------------------------
    /// Get a view of this span of '[offset[n], offset[n] + size[n])' for each dimension.
    ///
    /// @param offset_  Start offset in each dimension.
    /// @param size_in_ The size of each dimension. If any element of this array is @c NPOS, it is interpreted as the
    ///                 end of the span in that dimension. If any size specification would end with a 0-size dimension,
    ///                 the dimension sizes will still be properly set and an empty span will be returned.
    CUDA_BOTH_INLINE
    auto subspan(SizeArray const& offset_,
                 SizeArray const& size_in_ = makeFilledArray<size_t, DIMS_, NPOS>()) const -> span<T, DIMS_>
    {
        return span<T, DIMS_>(*this, offset_, size_in_);
    }

    CUDA_BOTH_INLINE
    size_t getOffset(size_t const idx) const
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= m_size[DIMS_ - 1])
        {
#if !defined(__CUDACC__)
            throwSpanIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("span::operator[]: access out of bounds");
#endif
        }
#endif

        size_t pitchDim{m_pitchBytes};
        for (size_t i{1}; i < DIMS_ - 1; i++)
        {
            pitchDim *= m_size[i];
        }

        return pitchDim * idx;
    }

private:
    FRIEND_TEST(SpanTests, spantest_L0);
    // -----------------------------------------------------------------------------
    /// Check whether index valid for this span.
    /// Throw out of bounds exception if it isn't.
    CUDA_BOTH_INLINE void checkAccess(const Array<size_t, DIMS_>& offset_,
                                      const Array<size_t, DIMS_>& size_,
                                      char8_t const* const outOfBoundsMsg) const
    {
#if DW_RUNTIME_CHECKS()
        for (size_t d{0U}; d < DIMS; ++d)
        {
            if (safeAdd(offset_[d], size_[d]) > m_size[d])
            {
#if !defined(__CUDACC__)
                throwSpanIndexOutOfBounds();
                static_cast<void>(outOfBoundsMsg);
#else
                assertException<OutOfBoundsException>(outOfBoundsMsg);
#endif
            }
        }
#else
        static_cast<void>(offset_);
        static_cast<void>(size_);
        static_cast<void>(outOfBoundsMsg);
#endif
    }

    // -----------------------------------------------------------------------------
    /// Check whether index valid for this span.
    /// Throw out of bounds exception if it isn't.
    CUDA_BOTH_INLINE void checkAccess(const Array<size_t, DIMS_>& offset_,
                                      char8_t const* const outOfBoundsMsg) const
    {
#if DW_RUNTIME_CHECKS()
        for (size_t d{0U}; d < DIMS; ++d)
        {
            if (offset_[d] >= m_size[d])
            {
#if !defined(__CUDACC__)
                throwSpanIndexOutOfBounds();
                static_cast<void>(outOfBoundsMsg);
#else
                assertException<OutOfBoundsException>(outOfBoundsMsg);
#endif
            }
        }
#else
        static_cast<void>(offset_);
        static_cast<void>(outOfBoundsMsg);
#endif
    }
    // -----------------------------------------------------------------------------
    /// Get a pointer to the specified @a offset_. It is a good idea to call @ref checkAccess with the sizes you intend
    /// to use before calling this function.
    CUDA_BOTH_INLINE auto pointerToUnchecked(const Array<size_t, DIMS_>& offset_) const -> T*
    {
        // Do calculations in bytes

        // TODO(dwplc): RFD - needed for support any type where sizeof(T) > 1
        // coverity[autosar_cpp14_m3_9_3_violation]
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto* ptri = reinterpret_cast<TByte*>(m_ptr); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)

        // First dimension has all elements contiguously
        ptri += offset_[0U] * sizeof(T);

        // Second dimension is pitched
        ptri += offset_[1U] * m_pitchBytes;

        // The rest of the dimensions are contiguous
        if (DIMS > 2)
        {
            size_t pitchDimBytes{m_pitchBytes};
            for (size_t d{2U}; d < DIMS; ++d)
            {
                pitchDimBytes = pitchDimBytes * m_size[d - 1];
                ptri += pitchDimBytes * offset_[d];
            }
        }

        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T*>(ptri);
    }

protected:
    FRIEND_TEST(SpanTests, spantest_L0);
    // -----------------------------------------------------------------------------
    /// Construct a subspan by index from a DIM+1 external span
    /// @param other        external span
    /// @param idx          dim index
    CUDA_BOTH_INLINE BaseSpan(BaseSpan<T, DIMS_ + 1> const& other, size_t const& idx) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : BaseSpan()
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto offset = other.getOffset(idx);

        // TODO(dwplc): RFD - needed for support for Span of float32_t
        // coverity[autosar_cpp14_m3_9_3_violation]
        reset(reinterpret_cast<T*>(reinterpret_cast<TByte*>(other.data()) + offset), // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
              other.size().template getFirst<DIMS_>(),
              other.pitch_bytes());
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by '[offset[n], offset[n] + size[n])' for each dimension from external span.
    ///
    /// @param other    external span
    /// @param offset_  Start offset in each dimension.
    /// @param size_in_ The size of each dimension. If any element of this array is @c NPOS, it is interpreted as the
    ///                 end of the span in that dimension. If any size specification would end with a 0-size dimension,
    ///                 the dimension sizes will still be properly set and an empty span will be returned.
    CUDA_BOTH_INLINE // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        BaseSpan(BaseSpan<T, DIMS_> const& other,
                 SizeArray const& offset_, SizeArray const& size_in_)
        : BaseSpan(other)
    {
        SizeArray size_;
        for (size_t i{0}; i < DIMS_; ++i)
        {
            if (size_in_[i] == NPOS)
            {
                size_[i] = size()[i] - offset_[i];
            }
            else
            {
                size_[i] = size_in_[i];
            }
        }

        checkAccess(offset_, size_, "span::subspan: access out of bounds");
        reset(pointerToUnchecked(offset_), size_, pitch_bytes());
    }

    CUDA_BOTH_INLINE
    void reset(T* ptr, SizeArray const& sizeIn, std::size_t const pitchBytesIn)
    {
        m_ptr        = ptr;
        m_size       = sizeIn;
        m_pitchBytes = pitchBytesIn;
    }

private:
    T* m_ptr;            //!< unmanaged pointer copy
    size_t m_pitchBytes; //!< Pitch (in bytes) of the first dimension
    SizeArray m_size;    //!< size in elements
};

/// 2D specialization, adds 2D access methods
template <typename T>
class span<T, 2> : public BaseSpan<T, 2>
{
public:
    /// Type of base class
    using Base = BaseSpan<T, 2>;

    /// Type of size array (size of each dimension)
    using SizeArray = typename Base::SizeArray;

    // Import constructors
    // Note: this should import the 3 constructors below but some compilers (e.g. clang)
    //       don't recognize it.
    // code-comment " using typename Base::BaseSpan;"

    // -----------------------------------------------------------------------------
    // Compiler WAR: constructor is not imported correctly
    /// Default constructor
    span()
        : Base() {}

    span(span const& other) = default;            ///< Copy constructor
    span& operator=(const span& other) = default; ///< Copy operator
    ~span()                            = default; ///< Destructor

    // -----------------------------------------------------------------------------
    /// Empty construction
    explicit span(std::nullptr_t)
        : Base(nullptr, SizeArray{}, 0)
    {
    }

    // -----------------------------------------------------------------------------
    // Compiler WAR: constructor is not imported correctly
    /// Construction with existing array in memory
    /// @param ptrIn          data pointer
    /// @param sizeIn         array with size of each dimension
    /// @param pitchBytesIn   data pitch, that is the distance in bytes of the between
    ///                       elements in dimension 0
    CUDA_BOTH_INLINE span(T* const ptrIn, SizeArray const& sizeIn,
                          std::size_t const pitchBytesIn = 0)
        : Base(ptrIn, sizeIn, pitchBytesIn)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construction with existing 2 dimensional array in memory
    /// @param ptr            data pointer
    /// @param size0          size of dimension 0
    /// @param size1          size of dimension 1
    /// @param pitchBytes     data pitch, that is the distance in bytes of the between
    ///                       elements in dimension 0
    CUDA_BOTH_INLINE span(T* const ptr, std::size_t const size0, std::size_t const size1, std::size_t const pitchBytes = 0)
        : span(ptr, {size0, size1}, pitchBytes)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by index from a DIM+1 external span
    /// @param other          external span
    /// @param idx          dim index
    CUDA_BOTH_INLINE span(core::BaseSpan<T, 3> const& other, std::size_t const& idx)
        : Base(other, idx)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a reshaped 2d span by given 1d span
    /// @param other        external 1D span
    /// @param size0        dim0 size
    /// @param size1        dim1 size
    CUDA_BOTH_INLINE span(core::span<T, 1> const& other,
                          std::size_t size0,
                          std::size_t size1)
        : Base()
    {
        // checks validity of reshape sizes
        detail_span::expandDimensions(size0, size1, other.size());
        this->reset(other.data(), {size0, size1}, safeMul(size0, sizeof(T)).value());
    }

    // -----------------------------------------------------------------------------
    /// Construct a reshaped 2d span by given nD span
    /// @param other        external nD span
    /// @param size0        dim0 size
    /// @param size1        dim1 size
    template <std::size_t N, std::enable_if_t<(N > 1)>* = nullptr>
    CUDA_BOTH_INLINE span(core::BaseSpan<T, N> const& other,
                          std::size_t size0,
                          std::size_t size1)
        : Base()
    {
        // checks validity of reshape sizes
        detail_span::expandDimensions(size0, size1, other.elementCount());

        // Check that there is no padding
        if ((size0 != other.size()[0]) && other.hasPadding())
        {
            core::assertException<core::InvalidArgumentException>("span: cannot reshape to 3D and change size 0 when there is padding");
        }
        this->reset(other.data(), {size0, size1}, other.pitch_bytes());
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by '[offset[n], offset[n] + size[n])' for each dimension from external span.
    ///
    /// @param other    external span
    /// @param offset_  Start offset in each dimension.
    /// @param size_in_ The size of each dimension. If any element of this array is @c NPOS, it is interpreted as the
    ///                 end of the span in that dimension. If any size specification would end with a 0-size dimension,
    ///                 the dimension sizes will still be properly set and an empty span will be returned.
    CUDA_BOTH_INLINE
    span(core::BaseSpan<T, 2> const& other,
         core::Array<std::size_t, 2> const& offset_, core::Array<std::size_t, 2> const& size_in_)
        : Base(other, offset_, size_in_)
    {
    }

    // -----------------------------------------------------------------------------
    using Base::at;

    /// Indexed element access in 2 dimensional span
    CUDA_BOTH_INLINE auto at(std::size_t const x0, std::size_t const x1) const -> T&
    {
        return Base::at({x0, x1});
    }

    // -----------------------------------------------------------------------------
    /// Indexed const element access in 2 dimensional span
    CUDA_BOTH_INLINE auto operator()(std::size_t const x0, std::size_t const x1) const -> T&
    {
        return Base::at({x0, x1});
    }
};

/// 3D specialization, adds 3D access methods
template <typename T>
class span<T, 3> : public BaseSpan<T, 3>
{
public:
    /// Type of base class
    using Base = BaseSpan<T, 3>;

    /// Type of size array (size of each dimension)
    using SizeArray = typename Base::SizeArray;

    // Import constructors
    // Note: this should import the 3 constructors below but some compilers (e.g. clang)
    //       don't recognize it.
    // code-comment " using typename Base::BaseSpan;"

    // -----------------------------------------------------------------------------
    // Compiler WAR: constructor is not imported correctly
    /// Default constructor
    span()
        : Base() {}
    span(span const& other) = default;            ///< Copy constructor
    span& operator=(const span& other) = default; ///< Copy operator
    ~span()                            = default; ///< Destructor

    // -----------------------------------------------------------------------------
    /// Empty construction
    explicit span(std::nullptr_t)
        : Base(nullptr, SizeArray{}, 0)
    {
    }

    // -----------------------------------------------------------------------------
    /// Compiler WAR: constructor is not imported correctly
    /// Construction with existing array in memory
    /// @param ptrIn          data pointer
    /// @param sizeIn         array with size of each dimension
    /// @param pitchBytesIn   data pitch, that is the distance in bytes of the between
    ///                       elements in dimension 0
    CUDA_BOTH_INLINE span(T* const ptrIn, SizeArray const& sizeIn, size_t const pitchBytesIn = 0)
        : Base(ptrIn, sizeIn, pitchBytesIn)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construction with existing 3 dimensional array in memory
    /// @param ptr            data pointer
    /// @param size0          size of dimension 0
    /// @param size1          size of dimension 1
    /// @param size2          size of dimension 2
    /// @param pitchBytes     data pitch, that is the distance in bytes of the between
    ///                       elements in dimension 0
    CUDA_BOTH_INLINE span(T* const ptr, std::size_t const size0, std::size_t const size1, std::size_t const size2, std::size_t const pitchBytes = 0)
        : Base(ptr, {size0, size1, size2}, pitchBytes)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by index from a DIM+1 external span
    /// @param other          external span
    /// @param idx         dim index
    CUDA_BOTH_INLINE span(core::BaseSpan<T, 4> const& other, std::size_t const& idx)
        : Base(other, idx)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a reshaped 3d span by given 1d span
    /// @param other        external 1D span
    /// @param size0        dim0 size
    /// @param size1        dim1 size
    /// @param size2        dim2 size
    CUDA_BOTH_INLINE span(core::span<T, 1> const& other,
                          std::size_t size0,
                          std::size_t size1,
                          std::size_t size2)
        : Base()
    {
        // checks validity of reshape sizes
        detail_span::expandDimensions(size0, size1, size2, other.size());
        this->reset(other.data(), {size0, size1, size2}, safeMul(size0, sizeof(T)).value());
    }

    // -----------------------------------------------------------------------------
    /// Construct a reshaped 3d span by given nD span
    /// @param other        external nD span
    /// @param size0        dim0 size
    /// @param size1        dim1 size
    /// @param size2        dim2 size
    template <std::size_t N, std::enable_if_t<(N > 1)>* = nullptr>
    CUDA_BOTH_INLINE span(core::BaseSpan<T, N> const& other,
                          std::size_t size0,
                          std::size_t size1,
                          std::size_t size2)
        : Base()
    {
        // checks validity of reshape sizes
        detail_span::expandDimensions(size0, size1, size2, other.elementCount());

        // Check that there is no padding
        if ((size0 != other.size()[0]) && other.hasPadding())
        {
            core::assertException<core::InvalidArgumentException>("span: cannot reshape to 3D and change size 0 when there is padding");
        }
        this->reset(other.data(), {size0, size1, size2}, other.pitch_bytes());
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by '[offset[n], offset[n] + size[n])' for each dimension from external span.
    ///
    /// @param other    external span
    /// @param offset_  Start offset in each dimension.
    /// @param size_in_ The size of each dimension. If any element of this array is @c NPOS, it is interpreted as the
    ///                 end of the span in that dimension. If any size specification would end with a 0-size dimension,
    ///                 the dimension sizes will still be properly set and an empty span will be returned.
    CUDA_BOTH_INLINE
    span(core::BaseSpan<T, 3> const& other,
         core::Array<std::size_t, 3> const& offset_, core::Array<std::size_t, 3> const& size_in_)
        : Base(other, offset_, size_in_)
    {
    }

    // -----------------------------------------------------------------------------
    using Base::at;

    /// Indexed element access in 3 dimensional span
    CUDA_BOTH_INLINE auto at(std::size_t const x0, std::size_t const x1, std::size_t const x2) const -> T&
    {
        return Base::at({x0, x1, x2});
    }

    // -----------------------------------------------------------------------------
    /// Indexed element access in 3 dimensional span
    CUDA_BOTH_INLINE auto operator()(std::size_t const x0, std::size_t const x1, std::size_t const x2) const -> T&
    {
        return Base::at({x0, x1, x2});
    }

    // -----------------------------------------------------------------------------
    /// return dimensions
    CUDA_BOTH_INLINE std::size_t depth() const { return Base::size()[2]; }
};

/// N-dimensional specialization
template <typename T, size_t DIMS>
class span : public BaseSpan<T, DIMS>
{
public:
    /// Type of base class
    using Base = dw::core::BaseSpan<T, DIMS>;

    /// Type of size array (size of each dimension)
    using SizeArray = typename Base::SizeArray;
    // Import constructors
    // Note: this should import the 3 constructors below but some compilers (e.g. clang)
    //       don't recognize it.
    // code-comment " using typename Base::BaseSpan;"

    // -----------------------------------------------------------------------------
    // Compiler WAR: constructor is not imported correctly
    /// Default constructor
    span()
        : Base() {}
    span(span const& other) = default;            ///< Copy constructor
    span& operator=(const span& other) = default; ///< Copy operator
    ~span()                            = default; ///< Destructor

    // -----------------------------------------------------------------------------
    /// Empty construction
    explicit span(std::nullptr_t)
        : Base(nullptr, SizeArray{}, 0)
    {
    }

    // -----------------------------------------------------------------------------
    // Compiler WAR: constructor is not imported correctly
    /// Construction with existing array in memory
    /// @param ptrIn          data pointer
    /// @param sizeIn         array with size of each dimension
    /// @param pitchBytesIn   data pitch, that is the distance in bytes of the between
    ///                       elements in dimension 0
    CUDA_BOTH_INLINE span(T* const ptrIn, SizeArray const& sizeIn, size_t const pitchBytesIn = 0)
        : Base(ptrIn, sizeIn, pitchBytesIn)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by index from a DIM+1 external span
    /// @param other          external span
    /// @param idx          dim index
    CUDA_BOTH_INLINE span(core::BaseSpan<T, DIMS + 1> const& other, std::size_t const& idx)
        : Base(other, idx)
    {
    }

    // -----------------------------------------------------------------------------
    /// Construct a subspan by '[offset[n], offset[n] + size[n])' for each dimension from external span.
    ///
    /// @param other    external span
    /// @param offset_  Start offset in each dimension.
    /// @param size_in_ The size of each dimension. If any element of this array is @c NPOS, it is interpreted as the
    ///                 end of the span in that dimension. If any size specification would end with a 0-size dimension,
    ///                 the dimension sizes will still be properly set and an empty span will be returned.
    CUDA_BOTH_INLINE
    span(core::BaseSpan<T, DIMS> const& other,
         typename Base::SizeArray const& offset_, typename Base::SizeArray const& size_in_)
        : Base(other, offset_, size_in_)
    {
    }

    // -----------------------------------------------------------------------------
    /// return dimensions
    CUDA_BOTH_INLINE std::size_t depth() const { return Base::size()[2]; }
};

// -----------------------------------------------------------------------------
/// Make 1-dimensional span from pointer and count
template <typename T>
CUDA_BOTH_INLINE auto make_span(T* const ptr, size_t const s) -> span<T, 1>
{
    return span<T, 1>(ptr, s);
}

// -----------------------------------------------------------------------------
/// Make n-dimensional span from pointer and count per dimension
template <typename T, typename... Sizes>
CUDA_BOTH_INLINE auto make_span(T* const ptr, size_t const size0, Sizes... sizes) -> span<T, 1 + sizeof...(Sizes)>
{
    using span_t = span<T, 1 + sizeof...(Sizes)>;
    return span_t(ptr, typename span_t::SizeArray{size0, sizes...});
}

// -----------------------------------------------------------------------------
/// Make n-dimensional span from pointer, count per dimension, and memory pitch in bytes
template <typename T, size_t N>
CUDA_BOTH_INLINE auto make_span(T* ptr, const Array<size_t, N>& size, size_t pitchBytes) -> span<T, N>
{
    return span<T, N>(ptr, size, pitchBytes);
}

// -----------------------------------------------------------------------------
/// Make 1-dimensional span from pointer and count
/// Overload is to match interface of n-dimensional make_span
template <typename T>
CUDA_BOTH_INLINE auto make_span(T* ptr, const Array<size_t, 1>& size, size_t /* pitchBytes */) -> span<T, 1>
{
    return span<T, 1>(ptr, size);
}

// -----------------------------------------------------------------------------
/// Make 1-dimensional span from C array
template <class T, size_t N>
CUDA_BOTH_INLINE auto make_span(T (&arr)[N]) -> span<T>
{
    return span<T>(arr);
}

// -----------------------------------------------------------------------------
/// Make 2-dimensional span from C array
template <class T, size_t M, size_t N>
CUDA_BOTH_INLINE auto make_span(T (&arr)[M][N]) -> span<T, 2>
{
    return span<T, 2>(&(arr[0][0]), M, N, M * sizeof(T));
}

// -----------------------------------------------------------------------------
/// Conversion from other types that have a data()+size() functions
/// SFINAE enable only if Cont has a data() member function
/// Disabled for span types
template <class Cont,
          class T = typename std::remove_pointer<decltype(std::declval<Cont>().data())>::type,
          class   = typename std::enable_if<!traits::is_span<typename std::remove_const<Cont>::type>::value && !traits::is_DeviceSpan<typename std::remove_const<Cont>::type>::value>::type>
auto make_span(Cont& cont) -> span<T>
{
    return span<T>(cont.data(), cont.size());
}

// -----------------------------------------------------------------------------
/// Make span of one const element
template <typename T>
CUDA_BOTH_INLINE auto make_span_of_one(const T& obj) -> span<const T>
{
    return span<const T>(&obj, 1);
}

// -----------------------------------------------------------------------------
/// Make span of one element
template <typename T>
CUDA_BOTH_INLINE auto make_span_of_one(T& obj) -> span<T>
{
    return span<T>(&obj, 1);
}

// -----------------------------------------------------------------------------
template <typename U, typename V>
CUDA_BOTH_INLINE std::enable_if_t<detail_span::IsSpanComparable<U, V>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) type T can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator==(const span<U, 1>& lhs, const span<V, 1>& rhs) noexcept
{
    return (lhs.m_ptr == rhs.m_ptr) && (lhs.m_size == rhs.m_size);
}

// -----------------------------------------------------------------------------
template <typename U, typename V>
CUDA_BOTH_INLINE std::enable_if_t<detail_span::IsSpanComparable<U, V>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) type T can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator!=(const span<U, 1>& lhs, const span<V, 1>& rhs) noexcept
{
    return !operator==(lhs, rhs);
}

/////////////////////////////////////////////////////////////
/// Memcpy functions
/////////////////////////////////////////////////////////////
namespace internal
{

// -----------------------------------------------------------------------------
/// Internal function.
/// Asynchronous memcpy from 1-dimensional span to 1-dimensional span.
/// Spans can refer to any type of memory (CPU or cuda), but the Memcpy Kind must be set correspondingly.
template <typename Td, typename Ts>
cudaError_t memcpyAsync(span<Td, 1> const dst, span<Ts, 1> const src, cudaMemcpyKind const kind, cudaStream_t const stream)
{
    static_assert(!std::is_const<Td>::value, "Destination span cannot be const");
    static_assert(std::is_same<Td, typename std::remove_const<Ts>::type>::value, "Span types must be the same after removing const");

#if DW_RUNTIME_CHECKS()
    if (dst.size() != src.size())
    {
        throw ExceptionWithStackTrace("memcpyAsync: Spans sizes don't match");
    }
#endif

    return cudaMemcpyAsync(dst.data(), src.data(), dst.size_bytes(), kind, stream);
}

// -----------------------------------------------------------------------------
/// Internal function.
/// Asynchronous memcpy from n-dimensional span to n-dimensional span.
/// Spans can refer to any type of memory (CPU or cuda), but the Memcpy Kind must be set correspondingly.
template <typename Td, typename Ts, size_t DIMS>
cudaError_t memcpyAsync(span<Td, DIMS> const dst, span<Ts, DIMS> const src, cudaMemcpyKind const kind, cudaStream_t const stream)
{
    static_assert(!std::is_const<Td>::value, "Destination span cannot be const");
    static_assert(std::is_same<Td, typename std::remove_const<Ts>::type>::value, "Span types must be the same after removing const");

#if DW_RUNTIME_CHECKS()
    for (size_t i{0}; i < DIMS; i++)
    {
        if (dst.size()[i] != src.size()[i])
        {
            throw ExceptionWithStackTrace("memcpyAsync: Spans sizes don't match");
        }
    }
#endif

    size_t height{dst.size()[1]};
    if (DIMS > 2)
    {
        for (size_t i{2}; i < DIMS; i++)
        {
            height = safeMul(height, dst.size()[i]).value();
        }
    }

    return cudaMemcpy2DAsync(dst.data(), dst.pitch_bytes(), src.data(), src.pitch_bytes(), safeMul(dst.size()[0], sizeof(Td)).value(), height, kind, stream);
}

} // namespace internal

// -----------------------------------------------------------------------------
/// Asynchronous memcpy from n-dimensional span to n-dimensional span, span's are in CPU memory.
template <typename Td, typename Ts, size_t DIMS>
cudaError_t memcpyAsync(span<Td, DIMS> dst, span<Ts, DIMS> src, cudaStream_t stream = nullptr)
{
    return internal::memcpyAsync(dst, src, cudaMemcpyHostToHost, stream);
}

// -----------------------------------------------------------------------------
/// Fill 1-dimensional span with value
template <typename Td>
// TODO(dwplc): FP - permissible to overload the name to add new parameter types
// coverity[autosar_cpp14_m17_0_3_violation]
void fill(dw::core::span<Td> dst, const Td& value)
{
    std::fill(dst.begin(), dst.end(), value);
}

// -----------------------------------------------------------------------------
/// Fill n-dimensional span with value.
template <typename Td, size_t DIMS>
// TODO(dwplc): FP - permissible to overload the name to add new parameter types
// coverity[autosar_cpp14_m17_0_3_violation]
void fill(span<Td, DIMS> dst, const Td& value)
{
    if (dst.pitch_bytes() == safeMul(sizeof(Td), dst.size(0)).value())
    {
        std::fill(dst.data(), dst.data() + dst.elementCount(), value);
    }
    else
    {
        auto dst2d = dst.reshape(dst.size(0), 0);
        for (size_t i = 0; i < dst2d.size(1); ++i)
        {
            auto row = dst2d[i];
            std::fill(row.data(), row.data() + dst2d.size(0), value);
        }
    }
}

} // namespace core
} // namespace dw

// -----------------------------------------------------------------------------
/// memcpy for 1-dimensional spans.
/// Note: define this outside of dw namespace so it overloads the standard memcpy and doesn't replace it
// TODO(dwplc): RFD - memcpy needs to be in the global namespace to overload the standard memcpy
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a7_3_1_violation]
// coverity[autosar_cpp14_m17_0_3_violation] -- FP - permissible to overload the name to add new parameter types
template <typename Td, typename Ts>
void memcpy(dw::core::span<Td> const dst, dw::core::span<Ts> const src)
{
    static_assert(!std::is_const<Td>::value, "Destination span cannot be const");
    static_assert(std::is_same<Td, typename std::remove_const<Ts>::type>::value, "Span types must be the same after removing const");

#if DW_RUNTIME_CHECKS()
    if (dst.size() != src.size())
    {
        throw dw::core::InvalidArgumentException("memcpy: Spans sizes don't match");
    }
#endif

    static_cast<void>(std::copy(src.begin(), src.end(), dst.begin()));
}

// -----------------------------------------------------------------------------
/// memcpy for n-dimensional spans.
/// Note: define this outside of dw namespace so it overloads the standard memcpy and doesn't replace it
// TODO(dwplc): RFD - memcpy needs to be in the global namespace to overload the standard memcpy
// coverity[autosar_cpp14_m7_3_1_violation]
// coverity[autosar_cpp14_a7_3_1_violation]
template <typename Td, typename Ts, size_t DIMS>
void memcpy(dw::core::span<Td, DIMS> dst, dw::core::span<Ts, DIMS> src)
{
    static_assert(!std::is_const<Td>::value, "Destination span cannot be const");
    static_assert(std::is_same<Td, typename std::remove_const<Ts>::type>::value, "Span types must be the same after removing const");

#if DW_RUNTIME_CHECKS()
    if (dst.size() != src.size())
    {
        throw dw::core::InvalidArgumentException("memcpy: Spans sizes don't match");
    }
#endif

    for (size_t i = 0; i < src.size()[DIMS - 1]; ++i)
    {
        memcpy(dst[i], src[i]);
    }
}

#endif
