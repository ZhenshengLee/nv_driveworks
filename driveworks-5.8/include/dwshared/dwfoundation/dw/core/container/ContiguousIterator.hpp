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

#ifndef DW_CORE_CONTAINER_CONTIGUOUSITERATOR_HPP_
#define DW_CORE_CONTAINER_CONTIGUOUSITERATOR_HPP_

#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/language/TypeAliases.hpp>
#include <dw/core/meta/BoolConstant.hpp>
#include <dw/core/meta/Conjunction.hpp>
#include <dw/core/utility/StaticIf.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>

namespace dw
{
namespace core
{
namespace detail
{

[[noreturn]] CUDA_BOTH
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    void
    throwContiguousIteratorOutOfBoundsImpl(std::size_t const elementSize,
                                           char8_t const* const op,
                                           void const* const lowerBound,
                                           void const* const upperBound,
                                           void const* const current);

template <typename TValue>
[[noreturn]] CUDA_BOTH
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    void
    throwContiguousIteratorOutOfBounds(char8_t const* const op,
                                       TValue const* const lowerBound,
                                       TValue const* const upperBound,
                                       TValue const* const current)
{
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    throwContiguousIteratorOutOfBoundsImpl(sizeof(TValue), op, lowerBound, upperBound, current);
}

[[noreturn]] CUDA_BOTH
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    void
    throwContiguousIteratorSubscriptOutOfBoundsImpl(std::size_t const elementSize,
                                                    char8_t const* const op,
                                                    void const* const lowerBound,
                                                    void const* const upperBound,
                                                    void const* const current,
                                                    std::size_t const subscriptOffset);

template <typename TValue>
[[noreturn]] CUDA_BOTH
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    void
    throwContiguousIteratorSubscriptOutOfBounds(char8_t const* op,
                                                TValue const* lowerBound,
                                                TValue const* upperBound,
                                                TValue const* current,
                                                std::size_t subscriptOffset)
{
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    throwContiguousIteratorSubscriptOutOfBoundsImpl(sizeof(TValue), op, lowerBound, upperBound, current, subscriptOffset);
}

/// Check that a @c BasicContiguousIterator with potentially-different @c TValue and @c TContainer can be compared
/// to. This is used to conditionally-enable things like @c operator== for different iterator types (for example:
/// the @c iterator and @c const_iterator for the same container).
template <typename LValue, typename LContainer, typename RValue, typename RContainer>
struct IsContiguousIteratorComparable : meta::Conjunction<std::is_same<std::remove_const_t<LValue>, std::remove_const_t<RValue>>,
                                                          std::is_same<std::remove_const_t<LContainer>, std::remove_const_t<RContainer>>>
{
};

} // namespace dw::core::detail

/// A @c BasicContiguousIterator is an iterator for containers which can be represented with a pointer, with bounds
/// which can be represented with a pair of pointers. The lower and upper bound pointers given at construction time are
/// always checked on dereference operations (@c operator*, @c operator->, and @c operator[]), ensuring the validity of
/// the iterator in the places it matters.
///
/// @note
/// The AUTOSAR/MISRA rule **M5-0-15** states that array indexing is the only form of pointer arithmetic allowed, which
/// this class appears to violate. However, there is an explicit exception to this rule in MISRA C++:2008 which allows
/// implementations of iterators to use pointer math for incrementing and decrementing. This also abides by **M5-0-16**
/// and **M5-0-17** by explicitly enforcing the result of such arithmetic by checking that bounds appear to address the
/// same array. However, since Coverity does not strictly understand this, the usage has been marked as exceptions.
///
/// @tparam TValue The possibly const-qualified value type this iterator should return.
/// @tparam TContainer The possibly const-qualified type this iterator came from. This factors into static comparability
///                    and assignment checking -- an iterator which originates from a @c VectorFixed vs a @c span can
///                    not be compared directly, even if their @c TValue is the same.
template <typename TValue, typename TContainer = void>
class BasicContiguousIterator final
{
    static_assert(!std::is_reference<TValue>::value, "Type TValue must not be a reference");
    static_assert(!std::is_void<TValue>::value, "Type TValue can not be void");
    static_assert(!std::is_reference<TContainer>::value, "Type TContainer must not be a reference");

    // TODO(dwplc): RFD - Being friends with other realizations of this range-checked iterator is more encapsulated and
    //              safer than adding an API for exposing such pointers
    // coverity[autosar_cpp14_a11_3_1_violation]
    // coverity[autosar_cpp14_a7_3_1_violation] -- FP - Not redeclaration or overload function
    template <typename UValue, typename UContainer>
    friend class BasicContiguousIterator;

    /// Check that a @c BasicContiguousIterator with potentially-different @c TValue and @c TContainer can be compared
    /// to. This is used to conditionally-enable things like @c operator== for different iterator types (for example:
    /// the @c iterator and @c const_iterator for the same container).
    template <typename UValue, typename UContainer>
    struct IsComparableIterator : meta::Conjunction<std::is_same<std::remove_const_t<TValue>, std::remove_const_t<UValue>>,
                                                    std::is_same<std::remove_const_t<TContainer>, std::remove_const_t<UContainer>>>
    {
    };

public:
    using value_type        = TValue;                          ///< Type of values
    using reference         = value_type&;                     ///< Type of value references
    using pointer           = value_type*;                     ///< Type of value pointers
    using size_type         = std::size_t;                     ///< Type of size representations
    using difference_type   = std::ptrdiff_t;                  ///< Type of iterator difference
    using iterator_category = std::random_access_iterator_tag; ///< Tag iterator as "random access"

private:
    /// Check that a number is an integer type that is smaller-than @c size_type. This is used in things like
    /// @c operator+ to support traversal with both signed and unsigned integers.
    template <typename TNumeric>
    struct IsRangeNumber : meta::BoolConstant<sizeof(TNumeric) <= sizeof(size_type) && std::is_integral<TNumeric>::value>
    {
    };

public:
    /// Create an instance bound by <tt>[lowerBound, upperBound)</tt> pointing at @a current.
    ///
    /// @param lowerBound The lowest range of the iterator (can be invalid if <tt>lowerBound == upperBound</tt>).
    /// @param upperBound One-past-the-end of the valid range.
    /// @param current The current position of the iterator. This does not have to be within the valid range (when
    ///                implementing @c end, this is @a upperBound).
    CUDA_BOTH_INLINE
    explicit constexpr BasicContiguousIterator(pointer const lowerBound, pointer const upperBound, pointer const current) noexcept
        : m_current(current)
        , m_lowerBound(lowerBound)
        , m_upperBound(upperBound)
    {
    }

    /// Create an unbound instance.
    constexpr BasicContiguousIterator() noexcept
        : BasicContiguousIterator(nullptr, nullptr, nullptr)
    {
    }

    /// Create an instance bound by <tt>[lowerBound, upperBound)</tt> initially pointing at @a lowerBound.
    ///
    /// @param lowerBound The lowest range of the iterator (can be invalid if <tt>lowerBound == upperBound</tt>)
    /// @param upperBound One-past-the-end of the valid range.
    CUDA_BOTH_INLINE
    explicit constexpr BasicContiguousIterator(pointer const lowerBound, pointer const upperBound) noexcept
        : BasicContiguousIterator(lowerBound, upperBound, lowerBound)
    {
    }

    /// Create an instance looking at @a source (which has @a sourceSize) at the given @a index. After creation,
    /// @c *iter is equivalent to @c source[index].
    ///
    /// @param source The source array which this iterator views.
    /// @param sourceSize The size of @a source container.
    /// @param index The index of the container this iterator points to. It is legal for @a index to be out-of-range
    ///              (common when implementing @c end).
    // coverity[autosar_cpp14_a7_1_1_violation]
    CUDA_BOTH_INLINE
    explicit constexpr BasicContiguousIterator(value_type source[], size_type const sourceSize, size_type const index) noexcept
        // TODO(dwplc): RFD - the implementation allow index to be same as array size, to indicate the end
        // coverity[cert_ctr50_cpp_violation]
        : BasicContiguousIterator(source, &source[sourceSize], &source[index])
    {
    }

    /// Create an instance looking at @a source (which has @a sourceSize) at the beginning of it. After creation,
    /// @c *iter is equivalent to @c source[0].
    ///
    /// @param source The source array which this iterator views.
    /// @param sourceSize The size of @a source container.
    // coverity[autosar_cpp14_a7_1_1_violation]
    CUDA_BOTH_INLINE
    explicit constexpr BasicContiguousIterator(value_type source[], size_type const sourceSize) noexcept
        // TODO(dwplc): RFD - the implementation allow index to be same as array size, to indicate the end
        // coverity[cert_ctr50_cpp_violation]
        : BasicContiguousIterator(source, &source[sourceSize], source)
    {
    }

    BasicContiguousIterator(BasicContiguousIterator const&) noexcept = default;
    BasicContiguousIterator(BasicContiguousIterator&&) noexcept      = default;

    // clang-format off
    /// Converting constructor to allow conversion from a non-const iterator type to a const iterator type. Formally,
    /// this constructor is only enabled if:
    ///
    ///  1. @c UValue is different from the @c value_type of this container
    ///  2. A pointer to an array of @c UValue is convertible to an array of @c value_type (this is more restrictive
    ///     than pointer assignment, which would allow conversion to a base class pointer, leading to erroneous behavior
    ///     of pointer math)
    ///  3. Either:
    ///     a. Both the container tags for this instance and @a other are @c void
    ///     b. Neither of the container tags are @c void and a pointer to the container tag for this instance is
    ///        constructible from a pointer to the container tag for @a other
    template <typename UValue,
              typename UContainer,
              typename = std::enable_if_t<  !std::is_same<value_type, UValue>::value
                                         && std::is_convertible<UValue(*)[], value_type(*)[]>::value
                                         && (
                                               (  std::is_void<TContainer>::value
                                               && std::is_void<UContainer>::value
                                               )
                                            ||
                                               (  !std::is_void<TContainer>::value
                                               && !std::is_void<UContainer>::value
                                               && std::is_constructible<TContainer*, UContainer* const&>::value
                                               )
                                            )
                                         >
             >
    // clang-tidy NOLINTNEXTLINE(google-explicit-constructor) -- converting constructor is intentionally implicit
    CUDA_BOTH_INLINE constexpr BasicContiguousIterator(BasicContiguousIterator<UValue, UContainer> const& other) noexcept
        : m_current(other.m_current)
        , m_lowerBound(other.m_lowerBound)
        , m_upperBound(other.m_upperBound)
    {
    }
    // clang-format on

    BasicContiguousIterator& operator=(BasicContiguousIterator const&) noexcept = default;
    BasicContiguousIterator& operator=(BasicContiguousIterator&&) noexcept = default;

    ~BasicContiguousIterator() noexcept = default;

    /// The dereference operator accesses the underlying container by dereferencing the pointer. This can throw if it
    /// exceeds the boundaries of the source.
    CUDA_BOTH_INLINE constexpr auto operator*() const -> reference
    {
        return dereference("operator*");
    }

    /// @see operator*()
    CUDA_BOTH_INLINE constexpr auto operator-> () const -> pointer
    {
        return &dereference("operator->");
    }

    /// Similar to @c operator*(), but adds additional @a offset to the current index. Note that this will also throw
    /// if the base offset is beyond the boundaries of the source even in cases where the offset would be legal, as the
    /// iterator itself is invalid.
    CUDA_BOTH_INLINE constexpr auto operator[](std::size_t const offset) const -> reference
    {
        return subscript("operator[]", offset);
    }

    /// @{
    /// Increment this iterator by 1.
    CUDA_BOTH_INLINE constexpr auto operator++() noexcept -> BasicContiguousIterator&
    {
        // NOTE(dwplc): M5-0-15 makes an exception for this rule for implementing iterators
        // coverity[autosar_cpp14_m5_0_15_violation]
        ++m_current;
        return *this;
    }

    /// Increment this iterator by 1 (postfix).
    CUDA_BOTH_INLINE constexpr auto operator++(int32_t) noexcept -> BasicContiguousIterator
    {
        BasicContiguousIterator const save(*this);
        operator++();
        return save;
    }

    /// Decrement this iterator by 1.
    CUDA_BOTH_INLINE constexpr auto operator--() noexcept -> BasicContiguousIterator&
    {
        // NOTE(dwplc): M5-0-15 makes an exception for this rule for implementing iterators
        // coverity[autosar_cpp14_m5_0_15_violation]
        --m_current;
        return *this;
    }

    /// Decrement this iterator by 1 (postfix).
    CUDA_BOTH_INLINE constexpr auto operator--(int32_t) noexcept -> BasicContiguousIterator
    {
        BasicContiguousIterator const save(*this);
        operator--();
        return save;
    }
    /// @}

    template <typename TNumeric, std::enable_if_t<IsRangeNumber<TNumeric>::value, bool> = true>
    CUDA_BOTH_INLINE constexpr auto operator+=(TNumeric const count) noexcept -> BasicContiguousIterator&
    {
        // NOTE(dwplc): M5-0-15 makes an exception for this rule for implementing iterators
        // coverity[autosar_cpp14_m5_0_15_violation]
        m_current += count;
        return *this;
    }

    template <typename TNumeric, std::enable_if_t<IsRangeNumber<TNumeric>::value, bool> = true>
    CUDA_BOTH_INLINE constexpr auto operator-=(TNumeric const count) noexcept -> BasicContiguousIterator&
    {
        // NOTE(dwplc): M5-0-15 makes an exception for this rule for implementing iterators
        // coverity[autosar_cpp14_m5_0_15_violation]
        m_current -= count;
        return *this;
    }

    template <typename TNumeric, std::enable_if_t<IsRangeNumber<TNumeric>::value, bool> = true>
    CUDA_BOTH_INLINE constexpr auto operator+(TNumeric const count) const noexcept -> BasicContiguousIterator
    {
        BasicContiguousIterator iterator = BasicContiguousIterator(*this);
        iterator += count;
        return iterator;
    }

    template <typename TNumeric, std::enable_if_t<IsRangeNumber<TNumeric>::value, bool> = true>
    CUDA_BOTH_INLINE constexpr auto operator-(TNumeric const count) const noexcept -> BasicContiguousIterator
    {
        BasicContiguousIterator iterator = BasicContiguousIterator(*this);
        iterator -= count;
        return iterator;
    }

    /// Get the difference between this instance and @a other. This overload is only enabled if the type of @a other is
    /// compatible with this type (e.g.: the @c iterator and @c const_iterator for the same container).
    ///
    /// @return The distance between this iterator and @a other. If the instances do not appear to refer to the same
    ///         backing array, the minimum value for @c difference_type is returned.
    template <typename UValue, typename UContainer>
    CUDA_BOTH_INLINE constexpr std::enable_if_t<IsComparableIterator<UValue, UContainer>::value, difference_type>
    operator-(BasicContiguousIterator<UValue, UContainer> const& other) const noexcept
    {
        // Abide by M5-0-17 by checking that we have the same boundaries as the other and return a nonsensically out of
        // bounds number if that is the case.
        if ((m_lowerBound != other.m_lowerBound) || (m_upperBound != other.m_upperBound))
        {
            return std::numeric_limits<difference_type>::min();
        }

        // TODO(dwplc): M5-0-15 makes an exception for this rule for implementing iterator increment operations but does
        // not have an exception for the difference operator. However, there is only one logical way you would implement
        // this operator, so it seems like an oversight on MISRA.
        // coverity[autosar_cpp14_m5_0_15_violation]
        return m_current - other.m_current;
    }

    /// @{
    /// Compare lhs with @a rhs for equality. If lhs and @a rhs do not refer to the same array
    /// boundaries, this will return @c false, even in cases where <tt>&amp;*lhs == &amp;*rhs</tt>.
    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator==(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;

    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator!=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;

    /// Check that lhs refers to a value less than @a rhs. Unlike equality comparisons, ordered comparisons
    /// do not check if iterators appear to refer to the same array.
    /// The reason why operator< does not do bounds checking is because it isn't clear what should happen on unequal boundaries.
    /// Using the bounds will end up with irreflexive operators. We can't throw, since comparison operators must be noexcept.
    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator<(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;

    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator<=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;

    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator>(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;

    template <typename LValue, typename LContainer, typename RValue, typename RContainer>
    CUDA_BOTH_INLINE friend constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
    operator>=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept;
    /// @}

private:
    CUDA_BOTH_INLINE
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    constexpr auto dereference(char8_t const* const op) const -> reference
    {
        if ((m_current < m_lowerBound) || (m_current >= m_upperBound))
        {
            detail::throwContiguousIteratorOutOfBounds(op, m_lowerBound, m_upperBound, m_current);
        }

        return *m_current;
    }

    CUDA_BOTH_INLINE
    // TODO(dwplc): Fix later
    // coverity[autosar_cpp14_a27_0_4_violation] -- will be fixed later after StringView is implemented
    constexpr auto subscript(char8_t const* const op, size_type const offset) const -> reference
    {
        pointer const goal = m_current + offset;

        if (((m_current < m_lowerBound) || (m_current >= m_upperBound)) || ((goal < m_lowerBound) || (goal >= m_upperBound)))
        {
            detail::throwContiguousIteratorSubscriptOutOfBounds(op, m_lowerBound, m_upperBound, m_current, offset);
        }

        return *goal;
    }

private:
    // TODO(dwplc): RFD -- If ContiguousIterator is iterating over character types, Coverity thinks you should use
    //              std::string instead. This rule does not make sense in cases where a pointer to a char type is not
    //              used in a string-like manner.
    // coverity[autosar_cpp14_a27_0_4_violation]
    pointer m_current;
    // coverity[autosar_cpp14_a27_0_4_violation]
    pointer m_lowerBound;
    // coverity[autosar_cpp14_a27_0_4_violation]
    pointer m_upperBound;
};

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator==(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    return ((lhs.m_current == rhs.m_current) && (lhs.m_lowerBound == rhs.m_lowerBound)) && (lhs.m_upperBound == rhs.m_upperBound);
}

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator!=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    return !operator==(lhs, rhs);
}

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator<(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    // TODO(dwplc): RFD - Not clear what should happen on unequal boundaries and can't throw in comparison operators
    // coverity[cert_ctr54_cpp_violation]
    return lhs.m_current < rhs.m_current;
}

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator<=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    return !(rhs < lhs);
}

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator>(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    return rhs < lhs;
}

template <typename LValue, typename LContainer, typename RValue, typename RContainer>
CUDA_BOTH_INLINE constexpr std::enable_if_t<detail::IsContiguousIteratorComparable<LValue, LContainer, RValue, RContainer>::value, bool>
// TODO(dwplc): RFD -- allow potentially-different(const v.s. non-const) @c TValue and @c TContainer can be compared to.
// coverity[autosar_cpp14_a13_5_5_violation]
operator>=(BasicContiguousIterator<LValue, LContainer> const& lhs, BasicContiguousIterator<RValue, RContainer> const& rhs) noexcept
{
    return !(lhs < rhs);
}

/// Advance @a iter by @a count.
template <typename TValue, typename TContainer, typename TNumeric>
CUDA_BOTH_INLINE constexpr auto operator+(TNumeric count, BasicContiguousIterator<TValue, TContainer> const& iter) noexcept -> decltype(std::declval<BasicContiguousIterator<TValue, TContainer> const&>() + std::declval<TNumeric const&>())
{
    return iter + count;
}

extern template class BasicContiguousIterator<std::uint16_t, void>;

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_CONTAINER_CONTIGUOUSITERATOR_HPP_*/
