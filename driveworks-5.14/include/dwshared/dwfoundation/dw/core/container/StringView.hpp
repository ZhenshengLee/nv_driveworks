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

#ifndef DWSHARED_CORE_CONTAINER_STRINGVIEW_HPP_
#define DWSHARED_CORE_CONTAINER_STRINGVIEW_HPP_

#include <dwshared/dwfoundation/dw/core/meta/TypeIdentity.hpp>
#include <dwshared/dwfoundation/dw/core/language/UnformattedOutputStreamTraits.hpp>
#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>
#include <dwshared/dwfoundation/dw/core/language/ToAddress.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <type_traits>

#include "ContiguousIterator.hpp"

namespace dw
{
namespace core
{

// TODO(dwplc): FP -- documentation is on the definition
// coverity[autosar_cpp14_a2_7_3_violation]
template <typename TChar, typename Traits>
class BasicStringView;

/// A string view of 8-bit characters with default traits. This is the most common form of string view.
using StringView = BasicStringView<char8_t, std::char_traits<char8_t>>;

namespace detail
{

// Determines size of string literal at compile time
// Replace w/ std::char_traits<T>::length when available in C++17
template <typename TChar>
constexpr std::size_t constexprSize(TChar const* const s)
{
    size_t count{0U};
    TChar const* str{s};
    constexpr std::size_t ONE_U{1U};
    // TODO(dwplc): RFD -- the size of str is not known so we cannot check that i is within bounds
    // TODO(dwplc): RFD -- not indexing an array, but it is unavoidable
    // coverity[autosar_cpp14_m5_0_15_violation]
    // coverity[cert_ctr50_cpp_violation]
    for (std::size_t i{0U}; str[i] /* !='\0' */; ++i)
    {
        count = count == std::numeric_limits<std::size_t>::max() ? std::numeric_limits<std::size_t>::max() : count + ONE_U;
    }
    return count;
}

/// @throw OutOfBoundsException
[[noreturn]] void throwStringViewIndexOutOfBounds(StringView const& operation, std::size_t const size, std::size_t const offset);

/// @throw OutOfBoundsException
[[noreturn]] void throwStringViewRemoveOutOfBounds(StringView const& operation, std::size_t const size, std::size_t const count);

/// This structure holds the names of operations and other values used by the @c BasicStringView implementation. It
/// exists for compliance with AUTOSAR A5-1-1, which states that only symbolic names are acceptable. You should question
/// if just putting in a @c 0 or @c 1 is really all that bad and if requiring symbolic names in this manner assists in
/// readability in any way. AUTOSAR did not do this, but you, dear reader, still can.
struct BasicStringViewImplValues
{
    /// @c -1
    // TODO(dwplc): FP -- Coverity is complaining about unbraced initialization
    // coverity[autosar_cpp14_a8_5_2_violation]
    template <typename T, std::enable_if_t<std::is_signed<T>::value, T> Value = T{-1}>
    using NegativeOne = std::integral_constant<T, Value>;

    /// @c 0
    // TODO(dwplc): FP -- Coverity is complaining about unbraced initialization
    // coverity[autosar_cpp14_a8_5_2_violation]
    template <typename T, std::enable_if_t<std::is_integral<T>::value, T> Value = T{0}>
    using Zero = std::integral_constant<T, Value>;

    /// @c 1
    // TODO(dwplc): FP -- Coverity is complaining about unbraced initialization
    // coverity[autosar_cpp14_a8_5_2_violation]
    template <typename T, std::enable_if_t<std::is_integral<T>::value, T> Value = T{1}>
    using One = std::integral_constant<T, Value>;

    /// @c "at"
    static StringView const OP_NAME_AT;

    /// @c "operator[]"
    static StringView const OP_NAME_OPERATOR_SUBSCRIPT;

    /// @c "front"
    static StringView const OP_NAME_FRONT;

    /// @c "back"
    static StringView const OP_NAME_BACK;

    /// @c "remove_prefix"
    static StringView const OP_NAME_REMOVE_PREFIX;

    /// @c "remove_suffix"
    static StringView const OP_NAME_REMOVE_SUFFIX;

    /// @c "substr"
    static StringView const OP_NAME_SUBSTR;

    /// @c "copy"
    static StringView const OP_NAME_COPY;

    /// @c "compare"
    static StringView const OP_NAME_COMPARE;
};

} // namespace dw::core::detail

/// An unowned reference to a string.
///
/// @warning
/// String views are not null terminated! If you need a null-terminated string, you must copy it into a buffer and add
/// the @c NUL character at this end. @ref FixedString does this for you on construction, the drawback being that, while
/// a @c BasicStringView instance has an unbound size, a @c FixedString does not.
///
/// There are a few function overloads which are missing. The @c find, @c rfind, @c find_first_of, @c find_last_of,
/// @c find_first_not_of, and @c find_last_not_of functions of @c std::basic_string_view have an overload form which
/// accepts a pointer and a size. However, since all these functions accept a position parameter as a start or end, they
/// take a form like <tt>size_type find(value_type* str, size_type startPosition, size_type strSize) const</tt>. The
/// order of arguments is sufficiently error-prone to warrant their lack of inclusion in this class. This functionality
/// should be replaced with the @c BasicStringView accepting overload.
///
/// @see StringView
template <typename TChar, typename Traits = std::char_traits<TChar>>
class BasicStringView final
{
public:
    /// The character traits of the string.
    using traits_type = Traits;

    /// The value type of a single character.
    // TODO(dwplc): FP -- using things like char8_t here violate this rule here, but not at the use site
    // coverity[autosar_cpp14_a3_9_1_violation]
    using value_type = TChar;

    /// A pointer to a character.
    // TODO(dwplc): RFD -- this is part of std::basic_string_view API, a class that replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    using pointer = value_type*;

    /// A constant pointer to a character.
    // TODO(dwplc): RFD -- this is part of std::basic_string_view API, a class that replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    using const_pointer = value_type const*;

    /// A reference to a character.
    using reference = value_type&;

    /// A constant reference to a character.
    using const_reference = value_type const&;

    /// An iterator which moves forwards.
    using const_iterator = BasicContiguousIterator<value_type const, BasicStringView<TChar, Traits> const>;

    /// An iterator which moves forwards. It is the same as @c const_iterator because @c BasicStringView is immutable.
    using iterator = const_iterator;

    /// An iterator which moves backwards.
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// An iterator which moves backwards.
    using reverse_iterator = const_reverse_iterator;

    /// Describes the @ref size of the view.
    using size_type = std::size_t;

    /// Describes the distance between two characters. This is a signed size type.
    using difference_type = std::ptrdiff_t;

    /// Represents not-a-position in search operations.
    static constexpr size_type npos{std::numeric_limits<size_type>::max()}; // clang-tidy NOLINT(readability-identifier-naming)

private:
    /// The value 0U. This is used as a value frequently enough to avoid AUTOSAR A5-1-1 that it warrants being a member.
    static constexpr size_type zero{0U}; // clang-tidy NOLINT(readability-identifier-naming)

public:
    /// Refer to string @a s with @a count characters.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr BasicStringView(const_pointer const s, size_type const count) noexcept
        : m_data(s)
        , m_size(count)
    {
    }

    /// Create an empty instance.
    // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
    constexpr BasicStringView() noexcept
        : BasicStringView(nullptr, 0U)
    {
    }

    ///< Copy constructor
    constexpr BasicStringView(BasicStringView const&) noexcept = default;
    ///< Move constructor
    constexpr BasicStringView(BasicStringView&&) noexcept = default;

    /// Refer to string @a s with a character count calculated through @c traits_type::length.
    // TODO(dwplc): FP -- std::char_traits<T>::length is not defined as noexcept, so Coverity is wrong on A15-4-4.
    // TODO(dwplc): RFD -- this class replaces C strings, so deviate on A27-0-4
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    // clang-tidy NOLINTNEXTLINE(google-explicit-constructor) -- implicit conversion is intentional
    constexpr BasicStringView(const_pointer const s) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BasicStringView(s, detail::constexprSize(s))
    {
    }

    ///< Copy operator
    constexpr BasicStringView& operator=(BasicStringView const&) noexcept = default;
    ///< Move operator
    constexpr BasicStringView& operator=(BasicStringView&&) noexcept = default;

    ///< Default destructor
    ~BasicStringView() noexcept = default;

    /// Get a pointer to the beginning of this view.
    ///
    /// @warning
    /// This is not equivalent to a @c c_str function -- it is @e not null terminated!
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto data() const noexcept -> const_pointer
    {
        return m_data;
    }

    /// @{
    /// Get the size of this view.
    constexpr auto size() const noexcept -> size_type
    {
        return m_size;
    }

    /// @see size
    constexpr auto length() const noexcept -> size_type
    {
        return size();
    }

    /// The same as @ref size, but gets the size as the signed type. Cases where the unsigned size would overflow the
    /// signed representation are silently converted to 0. If the size is above 2^63, it is likely the instance was
    /// invalid in the first place (or the system has an unfathomably large amount of memory).
    ///
    /// @note
    /// This function is not available on @c std::basic_string_view implementations, but safety requires the use of
    /// signed types frequently enough to warrant exposing this. The name comes from C++20 ranges, which use @c ssize
    /// for the same purpose.
    constexpr auto ssize() const noexcept -> difference_type
    {
        // TODO(dwplc): FP -- this is being checked on the next line
        // coverity[cert_int31_c_violation]
        difference_type const rc{static_cast<difference_type>(m_size)}; // clang-tidy NOLINT(modernize-use-auto)
        if ((rc < detail::BasicStringViewImplValues::Zero<difference_type>::value) || (static_cast<size_type>(rc) != m_size))
        {
            return detail::BasicStringViewImplValues::Zero<difference_type>::value;
        }
        else
        {
            return rc;
        }
    }
    /// @}

    /// Test if this view is empty -- the @ref size is zero.
    constexpr bool empty() const noexcept { return size() == zero; }

    /// Get the maximum number of elements this view can refer to.
    constexpr auto max_size() const noexcept -> size_type
    {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }

    /// @{
    /// Get an iterator pointing to the beginning of this sequence.
    constexpr auto begin() const noexcept -> const_iterator
    {
        // TODO(dwplc): RFD -- Giving these to a ContiguousIterator, which is range-checked
        // coverity[autosar_cpp14_m5_0_15_violation]
        return const_iterator{&m_data[zero], &m_data[size()]};
    }

    /// Get an iterator pointing to the beginning of this sequence.
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    /// Get an iterator pointing to one past the end of this sequence.
    constexpr auto end() const noexcept -> const_iterator
    {
        return begin() + size();
    }

    /// Get an iterator pointing to one past the end of this sequence.
    constexpr auto cend() const noexcept -> const_iterator
    {
        return end();
    }

    /// Get an iterator pointing to the end of this sequence, going backwards.
    constexpr auto rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator{end()};
    }

    /// Get an iterator pointing to the end of this sequence, going backwards.
    constexpr auto crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    /// Get an iterator pointing to one before the beginning of this sequence, going backwards.
    constexpr auto rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    /// Get an iterator pointing to one before the beginning of this sequence, going backwards.
    constexpr auto crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }
    /// @}

    /// Swap this instance with @a other.
    constexpr void swap(BasicStringView& other) noexcept
    {
        using std::swap;

        // TODO(dwplc): FP -- swapping is modification
        // coverity[autosar_cpp14_a8_4_9_violation]
        swap(m_data, other.m_data);
        // TODO(dwplc): FP -- swapping is modification
        // coverity[autosar_cpp14_a8_4_9_violation]
        swap(m_size, other.m_size);
    }

    /// @{
    /// Access the character at the given @a position.
    ///
    /// Throws an @c OutOfBoundsException if @a position is larger than @ref size.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto operator[](size_type const position) const -> const_reference
    {
        return at_impl(detail::BasicStringViewImplValues::OP_NAME_OPERATOR_SUBSCRIPT, position);
    }

    /// Access the character at the given @a position.
    ///
    /// Throws an @c OutOfBoundsException if @a position is larger than @ref size.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto at(size_type const position) const -> const_reference
    {
        return at_impl(detail::BasicStringViewImplValues::OP_NAME_AT, position);
    }
    /// @}

    /// Access the first character.
    ///
    /// Throws an @c OutOfBoundsException if this view is @ref empty.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto front() const -> const_reference
    {
        return at_impl(detail::BasicStringViewImplValues::OP_NAME_FRONT, zero);
    }

    /// Access the last character.
    ///
    /// Throws an @c OutOfBoundsException if this view is @ref empty.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto back() const -> const_reference
    {
        size_type const sz{size()};
        if (sz == zero)
        {
            detail::throwStringViewIndexOutOfBounds(detail::BasicStringViewImplValues::OP_NAME_BACK, zero, zero);
        }

        return at_impl(detail::BasicStringViewImplValues::OP_NAME_BACK, sz - detail::BasicStringViewImplValues::One<size_type>::value);
    }

    /// Remove @a count characters from the beginning of this view.
    ///
    /// Throws an @c OutOfBoundsException if @a count is larger than @ref size.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr void remove_prefix(size_type const count)
    {
        *this = substr_impl(detail::BasicStringViewImplValues::OP_NAME_REMOVE_PREFIX, count, npos);
    }

    /// Remove @a count characters from the end of this view.
    ///
    /// Throws an @c OutOfBoundsException if @a count is larger than @ref size.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr void remove_suffix(size_type const count)
    {
        if (count > size())
        {
            detail::throwStringViewRemoveOutOfBounds(detail::BasicStringViewImplValues::OP_NAME_REMOVE_SUFFIX, size(), count);
        }

        m_size -= count;
    }

    /// Create a sub-view of this instance, starting at @a position with @a count characters.
    ///
    /// Throws an @c OutOfBoundsException if @a position or @a position+count would exceed this instance's @ref size.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto substr(size_type const position = zero, size_type const count = npos) const -> BasicStringView
    {
        return substr_impl(detail::BasicStringViewImplValues::OP_NAME_SUBSTR, position, count);
    }

    /// Copy up to @a count characters to @a dest.
    ///
    /// @param dest The destination buffer to copy characters to. It must have space for at least @a count characters.
    /// @param count The maximum number of characters to copy.
    /// @param position The offset into this view to start copying characters from.
    ///
    /// @return the number of characters copied to @a dest.
    ///
    /// Throws an @c OutOfBoundsException if @a position is greater than @ref size.
    // TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::basic_string_view API
    // TODO(dwplc): FP for A15-4-4 and A27-0-4 -- This function can throw
    // coverity[autosar_cpp14_a8_4_8_violation]
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto copy(value_type* const dest, size_type const count, size_type const position = zero) const -> size_type
    {
        if (position > size())
        {
            detail::throwStringViewIndexOutOfBounds(detail::BasicStringViewImplValues::OP_NAME_COPY, size(), position);
        }

        size_type const rcount{std::min(count, size() - position)};
        static_cast<void>(traits_type::copy(dest, to_address(begin() + position), rcount));
        return rcount;
    }

    /// @{
    /// Compare this string view to @a other lexicographically according to @c traits_type.
    ///
    /// @return @c 0 if this view is equal-to @a other, @c <0 if this view is less-than @a other, or @c >0 if this view
    ///         is greater-than @a other. Note that relying on @c -1 and @c 1 for the @c <0 and @c >0 values is
    ///         incorrect.
    std::int32_t compare(BasicStringView const other) const noexcept
    {
        size_type const compare_size{std::min(size(), other.size())};

        std::int32_t res{};
        try
        {
            res = traits_type::compare(data(), other.data(), compare_size);
        }
        catch (...)
        {
            // std::char_traits implementations will never throw, but pretend it returned less-than for static analysis
            res = detail::BasicStringViewImplValues::NegativeOne<std::int32_t>::value;
        }

        if (res == detail::BasicStringViewImplValues::Zero<std::int32_t>::value)
        {
            if (size() == other.size())
            {
                return detail::BasicStringViewImplValues::Zero<std::int32_t>::value;
            }
            else if (size() < other.size())
            {
                return detail::BasicStringViewImplValues::NegativeOne<std::int32_t>::value;
            }
            else
            {
                return detail::BasicStringViewImplValues::One<std::int32_t>::value;
            }
        }
        else
        {
            return res;
        }
    }

    /// Equivalent to <tt>substr(position, count).compare(other)</tt>.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr std::int32_t compare(size_type const position, size_type const count, BasicStringView const other) const
    {
        return substr_impl("compare", position, count).compare(other);
    }

    /// Equivalent to <tt>substr(position, count).compare(other.substr(other_position, other_count))</tt>.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr std::int32_t compare(size_type const position,
                                   size_type const count,
                                   BasicStringView const other,
                                   size_type const otherPosition,
                                   size_type const otherCount) const
    {
        BasicStringView const selfSub{substr_impl(detail::BasicStringViewImplValues::OP_NAME_COMPARE, position, count)};
        BasicStringView const otherSub{other.substr_impl(detail::BasicStringViewImplValues::OP_NAME_COMPARE, otherPosition, otherCount)};

        return selfSub.compare(otherSub);
    }

    /// Equivalent to <tt>compare(BasicStringView(cStr))</tt>.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr std::int32_t compare(value_type const* const cStr) const
    {
        return compare(BasicStringView(cStr));
    }

    /// Equivalent to <tt>substr(position, count).compare(BasicStringView(cStr))</tt>.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr std::int32_t compare(size_type const position, size_type const count, value_type const* const cStr) const
    {
        return compare(position, count, BasicStringView(cStr));
    }

    /// Equivalent to <tt>substr(position, count).compare(BasicStringView(c_str, strLength))</tt>.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr std::int32_t compare(size_type const position,
                                   size_type const count,
                                   value_type const* const str,
                                   size_type const strLength) const
    {
        return compare(position, count, BasicStringView(str, strLength));
    }
    /// @}

    /// @{
    /// Test if this instance starts with the given @a prefix.
    constexpr bool starts_with(BasicStringView const prefix) const noexcept
    {
        if (size() < prefix.size())
        {
            return false;
        }
        else
        {
            BasicStringView const relevantSelf{substr_impl_noexcept(zero, prefix.size())};
            return relevantSelf.compare(prefix) == detail::BasicStringViewImplValues::Zero<std::int32_t>::value;
        }
    }

    /// Test if this instance starts with the given @a prefix.
    constexpr bool starts_with(value_type const prefix) const noexcept
    {
        size_type const sz{size()};
        if (sz == zero)
        {
            return false;
        }
        else
        {
            // TODO(dwplc): RFD -- this access is checked
            // coverity[autosar_cpp14_m5_0_15_violation]
            return m_data[zero] == prefix;
        }
    }

    /// Test if this instance starts with the given @a prefix.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr bool starts_with(value_type const* const prefix) const
    {
        return starts_with(BasicStringView(prefix));
    }
    /// @}

    /// @{
    /// Test if this instance ends with the given @a suffix.
    constexpr bool ends_with(BasicStringView const suffix) const noexcept
    {
        if (size() < suffix.size())
        {
            return false;
        }
        else
        {
            BasicStringView const relevantSelf{substr_impl_noexcept(size() - suffix.size(), suffix.size())};
            return relevantSelf.compare(suffix) == detail::BasicStringViewImplValues::Zero<std::int32_t>::value;
        }
    }

    /// Test if this instance ends with the given @a suffix.
    constexpr bool ends_with(value_type const suffix) const noexcept
    {
        size_type const sz{size()};
        if (sz == zero)
        {
            return false;
        }
        else
        {
            // TODO(dwplc): RFD -- m_data is a pointer, but we just checked the legality of this operation
            // coverity[autosar_cpp14_m5_0_15_violation]
            return m_data[sz - detail::BasicStringViewImplValues::One<size_type>::value] == suffix;
        }
    }

    /// Test if this instance ends with the given @a suffix.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr bool ends_with(value_type const* const suffix) const
    {
        return ends_with(BasicStringView(suffix));
    }
    /// @}

    /// @{
    /// Find the first occurance of the @a sub character sequence.
    ///
    /// @param sub The character subsequence to search for.
    /// @param startPosition The first position the @a sub character sequence can start at.
    ///
    /// @returns The index of the first character of @a sub in this instance after @a startPosition or @c npos if this
    ///          instance does not contain @a sub after @a startPosition.
    constexpr auto find(BasicStringView const sub, size_type const startPosition = zero) const noexcept -> size_type
    {
        if (startPosition >= size())
        {
            return npos;
        }

        const_iterator const it{std::search(begin() + startPosition, end(), sub.begin(), sub.end())};
        if (it == end())
        {
            return npos;
        }
        else
        {
            difference_type const dist{std::distance(begin(), it)};
            if (dist >= detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                return static_cast<size_type>(dist);
            }
            else
            {
                // unreachable: iterator is always past the beginning
                return npos;
            }
        }
    }

    /// Find the first occurance of the @a ch character.
    ///
    /// @param ch The character to search for.
    /// @param startPosition The first position the @a ch character can be at.
    ///
    /// @returns The index of the first occurance of @a ch in this instance after @a startPosition or @c npos if this
    ///          instance does not contain @a ch after @a startPosition.
    constexpr auto find(value_type const ch, size_type const startPosition = zero) const noexcept -> size_type
    {
        return find(BasicStringView(&ch, detail::BasicStringViewImplValues::One<size_type>::value), startPosition);
    }

    /// Find the first occurance of the @a sub character sequence.
    ///
    /// @param sub The character subsequence to search for.
    /// @param startPosition The first position the @a sub character sequence can start at.
    ///
    /// @returns The index of the first character of @a sub in this instance after @a startPosition or @c npos if this
    ///          instance does not contain @a sub after @a startPosition.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto find(value_type const* const sub, size_type const startPosition = zero) const -> size_type
    {
        return find(BasicStringView(sub), startPosition);
    }
    /// @}

    /// @{
    /// Find the last occurance of the @a sub character sequence.
    ///
    /// @param sub The character subsequence to search for.
    /// @param lastStartPosition The last position the @a sub character sequence can start at.
    ///
    /// @returns The index of the first character of the last occurance of @a sub in this instance or @c npos if this
    ///          instance does not contain @a sub before @a lastStartPosition.
    constexpr auto rfind(BasicStringView const sub, size_type const lastStartPosition = npos) const noexcept -> size_type
    {
        if (sub.size() > size())
        {
            return npos;
        }

        difference_type lastEndPosition{};
        if (lastStartPosition == npos)
        {
            lastEndPosition = ssize();
        }
        else
        {
            lastEndPosition = static_cast<difference_type>(lastStartPosition) + sub.ssize();
            if (lastEndPosition > ssize())
            {
                lastEndPosition = ssize();
            }
        }

        difference_type const reverseStartOffset{ssize() - lastEndPosition};

        const_reverse_iterator const it{std::search(rbegin() + reverseStartOffset, rend(), sub.rbegin(), sub.rend())};
        if (it == rend())
        {
            return npos;
        }
        else
        {
            difference_type rdistance{std::distance(rbegin(), it)};
            if (rdistance < detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                // unreachable: the 'it' we find is always later than 'rbegin()'
                return npos;
            }

            // add word length to get to start
            rdistance += sub.ssize();
            if (rdistance < detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                // realistically unreachable: If a substring match is large enough to overflow, we are out of memory
                return npos;
            }

            difference_type const res{ssize() - rdistance};
            if (res < detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                // unreachable: the word we matched in search is always included in this, but Coverity doesn't know that
                return npos;
            }
            else
            {
                return static_cast<size_type>(res);
            }
        }
    }

    /// Find the last occurance of the @a ch character.
    ///
    /// @param ch The character to search for.
    /// @param lastStartPosition The last position the @a ch character can be.
    ///
    /// @returns The index of the last occurance of @a ch in this instance or @c npos if this instance does not contain
    ///          @a ch before @a lastStartPosition.
    constexpr auto rfind(value_type const ch, size_type const lastStartPosition = npos) const noexcept -> size_type
    {
        return rfind(BasicStringView(&ch, 1U), lastStartPosition);
    }

    /// Find the last occurance of the @a str character sequence.
    ///
    /// @param str The character subsequence to search for.
    /// @param lastStartPosition The last position the @a sub character sequence can start at.
    ///
    /// @returns The index of the first character of the last occurance of @a sub in this instance or @c npos if this
    ///          instance does not contain @a sub before @a lastStartPosition.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto rfind(value_type const* const str, size_type const lastStartPosition = npos) const -> size_type
    {
        return rfind(BasicStringView(str), lastStartPosition);
    }
    /// @}

    /// @{
    /// Find the first character which matches any of the characters in the @a search query, starting at
    /// @a startPosition.
    ///
    /// @param search The character sequence to check for matches.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches in @a search or @c npos if none of the
    ///          characters match.
    constexpr auto find_first_of(BasicStringView const search, size_type const startPosition = zero) const noexcept -> size_type
    {
        return find_first_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search.find(ch) != npos;
            },
            startPosition);
    }

    /// Find the first character which matches the @a search character, starting at @a startPosition.
    ///
    /// @param search The character to search for.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches @a search or @c npos if none of the
    ///          characters match.
    constexpr auto find_first_of(value_type const search, size_type const startPosition = zero) const noexcept -> size_type
    {
        return find(search, startPosition);
    }

    /// Find the first character which matches any of the characters in the @a search query, starting at
    /// @a startPosition.
    ///
    /// @param search The character sequence to check for matches.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches in @a search or @c npos if none of the
    ///          characters match.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto find_first_of(value_type const* const search, size_type const startPosition = zero) const -> size_type
    {
        return find_first_of(BasicStringView(search), startPosition);
    }
    /// @}

    /// @{
    /// Find the last character which matches any of the characters in the @a search query, starting at @a endPosition.
    ///
    /// @param search The character sequence to check for matches.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches in @a search or @c npos if none of the
    ///          characters match.
    constexpr auto find_last_of(BasicStringView const search, size_type const endPosition = npos) const noexcept -> size_type
    {
        return find_last_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search.find(ch) != npos;
            },
            endPosition);
    }

    /// Find the last character which matches the @a search character, starting at @a endPosition.
    ///
    /// @param search The character to search for.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches @a search or @c npos if none of the
    ///          characters match.
    constexpr auto find_last_of(value_type const search, size_type const endPosition = npos) const noexcept -> size_type
    {
        return find_last_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search == ch;
            },
            endPosition);
    }

    /// Find the last character which matches any of the characters in the @a search query, starting at @a endPosition.
    ///
    /// @param search The character sequence to check for matches.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches in @a search or @c npos if none of the
    ///          characters match.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto find_last_of(value_type const* const search, size_type const endPosition) const -> size_type
    {
        return find_last_of(BasicStringView(search), endPosition);
    }
    /// @}

    /// @{
    /// Find the first character which does not match any of the characters in the @a search query, starting at
    /// @a startPosition.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the character not in @a search or @c npos if all characters in the instance match the
    ///          @a search.
    constexpr auto find_first_not_of(BasicStringView const search, size_type const startPosition = zero) const noexcept -> size_type
    {
        return find_first_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search.find(ch) == npos;
            },
            startPosition);
    }

    /// Find the first character which does not match the @a search character, starting at @a startPosition.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the character not in @a search or @c npos if all characters in the instance match the
    ///          @a search.
    constexpr auto find_first_not_of(value_type const search, size_type const startPosition = zero) const noexcept -> size_type
    {
        return find_first_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search != ch;
            },
            startPosition);
    }

    /// Find the first character which does not match any of the characters in the @a search query, starting at
    /// @a startPosition.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param startPosition The first index to search for.
    ///
    /// @returns The index of the character not in @a search or @c npos if all characters in the instance match the
    ///          @a search.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto find_first_not_of(value_type const* const search, size_type const startPosition = zero) const -> size_type
    {
        return find_first_not_of(BasicStringView(search), startPosition);
    }
    /// @}

    /// @{
    /// Find the last character which does not match any of the characters in the @a search query, starting at
    /// @a endPosition.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the character not in @a search or @c npos if all characters in the instance match the
    ///          @a search.
    constexpr auto find_last_not_of(BasicStringView const search, size_type const endPosition = npos) const noexcept -> size_type
    {
        return find_last_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search.find(ch) == npos;
            },
            endPosition);
    }

    /// Find the last character which is not @a search, starting at @a endPosition.
    ///
    /// @param search The character to check for non-matches.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the character which is not @a search or @c npos if all characters in the instance are
    ///          @a search.
    constexpr auto find_last_not_of(value_type const search, size_type const endPosition = npos) const noexcept -> size_type
    {
        return find_last_of_impl(
            [search](value_type const ch) noexcept->bool {
                return search != ch;
            },
            endPosition);
    }

    /// Find the last character which does not match any of the characters in the @a search query, starting at
    /// @a endPosition.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param endPosition The last index to search for.
    ///
    /// @returns The index of the character not in @a search or @c npos if all characters in the instance match the
    ///          @a search.
    // TODO(dwplc): FP -- This function can throw
    // coverity[autosar_cpp14_a15_4_4_violation]
    // coverity[autosar_cpp14_a27_0_4_violation]
    constexpr auto find_last_not_of(value_type const* const search, size_type const endPosition = npos) const -> size_type
    {
        return find_last_not_of(BasicStringView(search), endPosition);
    }
    /// @}

    /// Helper class for hashing strings, using djb2 algorithm
    class Hash
    {
    public:
        /// Hashes the given string
        size_t operator()(const BasicStringView<TChar>& strView) const noexcept
        {
            /// Base of the djb2 hash function
            constexpr size_t HASH_BASE{5381UL};

            return hashImpl(strView, HASH_BASE);
        }

        /// Hashes the given string with other hash as a starting point
        size_t combine(const BasicStringView<TChar>& strView, size_t const hashBase) const noexcept
        {
            return hashImpl(strView, hashBase);
        }

    private:
        /// Implementation of the djb2 hash algorithm
        size_t hashImpl(const BasicStringView<TChar>& strView, size_t const hashBase) const noexcept
        {
            try
            {
                size_t hash{hashBase};
                for (size_t i{0UL}; i < strView.length(); ++i)
                {

                    /// Unsigned type of TChar, used for casting the char type to an unsigned integer
                    // TODO(dwplc): FP -- There is documentation
                    // coverity[autosar_cpp14_a2_7_3_violation]
                    using uType = typename std::make_unsigned<TChar>::type;

                    // TODO(dwplc): RFD -- Checking the value of the character before casting would be a violation
                    //    because of rule M4-5-3, and misinterpreting data is fine here because it's only be added
                    //    to a hash value.
                    // coverity[cert_int31_c_violation]
                    uType const tmpChar{static_cast<uType>(strView[i])};
                    size_t const hashVal{static_cast<size_t>(tmpChar)};

                    /// Shift amount of the djb2 hash function
                    constexpr size_t HASH_SHIFT{5UL};
                    // TODO(dwplc): RFD -- Hash function designed to overflow
                    // coverity[autosar_cpp14_a4_7_1_violation]
                    // coverity[cert_int30_c_violation]
                    hash = ((hash << HASH_SHIFT) + hash) + hashVal;
                }
                return hash;
            }
            catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
            {
                static_cast<void>(e);
            }
            catch (dw::core::OutOfBoundsException const& e)
            {
                static_cast<void>(e);
            }
            return detail::BasicStringViewImplValues::Zero<size_t>::value;
        }
    };

private:
    /// Throws an @c OutOfBoundsException if @a idx is out of range
    // TODO(dwplc): FP -- this function can throw and should not be marked noexcept
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto at_impl(StringView const& operation, size_type const idx) const -> const_reference
    {
        if (idx >= size())
        {
            detail::throwStringViewIndexOutOfBounds(operation, size(), idx);
        }

        // TODO(dwplc): RFD -- not indexing an array, but it is checked and unavoidable
        // coverity[autosar_cpp14_m5_0_15_violation]
        return m_data[idx];
    }

    /// The underlying implementation of all @c substr APIs. The bounds are checked, but out-of-bounds parameters will
    /// result in an empty view.
    constexpr auto substr_impl_noexcept(size_type startOffset, size_type count) const noexcept -> BasicStringView
    {
        if (startOffset > size())
        {
            startOffset = size();
        }

        size_type const bufferSize{size()};
        size_type const endOffset{startOffset + count};
        if ((endOffset < startOffset) || (endOffset >= bufferSize))
        {
            count = bufferSize - startOffset;
        }

        if (count > zero)
        {
            return BasicStringView(to_address(begin() + startOffset), count);
        }
        else
        {
            return BasicStringView();
        }
    }

    /// Throws an @c OutOfBoundsException if @a startOffset is out of range
    // TODO(dwplc): FP -- this function can throw and should not be marked noexcept
    // coverity[autosar_cpp14_a15_4_4_violation]
    constexpr auto substr_impl(StringView const& operation, size_type const startOffset, size_type const count) const -> BasicStringView
    {
        if (startOffset > size())
        {
            detail::throwStringViewIndexOutOfBounds(operation, size(), startOffset);
        }

        return substr_impl_noexcept(startOffset, count);
    }

    /// Find the first character matching the given @a pred query.
    template <typename FPredicate>
    constexpr auto find_first_of_impl(FPredicate const& pred, size_type const startPosition) const noexcept -> size_type
    {
        if (startPosition >= size())
        {
            return npos;
        }

        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const it = std::find_if(begin() + startPosition, end(), pred);
        if (it == end())
        {
            return npos;
        }
        else
        {
            difference_type const rc{std::distance(begin(), it)};
            if (rc < detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                // unreachable: The 'it' is always after 'begin()'
                return npos;
            }
            else
            {
                return static_cast<size_type>(rc);
            }
        }
    }

    /// Finds the last character matching the given @a pred query.
    template <typename FPredicate>
    constexpr auto find_last_of_impl(FPredicate const& pred, size_type const endPosition) const noexcept -> size_type
    {
        size_type reverseStartOffset{};
        if (endPosition >= size())
        {
            reverseStartOffset = zero;
        }
        else
        {
            reverseStartOffset = size() - endPosition - detail::BasicStringViewImplValues::One<size_type>::value;
        }

        if (reverseStartOffset > static_cast<size_type>(PTRDIFF_MAX))
        {
            return npos;
        }

        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const it = std::find_if(rbegin() + static_cast<difference_type>(reverseStartOffset), rend(), pred);
        if (it == rend())
        {
            return npos;
        }
        else
        {
            difference_type const matchDistanceFromEnd{std::distance(rbegin(), it)};
            if (matchDistanceFromEnd < detail::BasicStringViewImplValues::Zero<difference_type>::value)
            {
                return npos;
            }
            else
            {
                // The extra - 1 comes from matching the letter as well
                return size() - static_cast<size_type>(matchDistanceFromEnd) - detail::BasicStringViewImplValues::One<size_type>::value;
            }
        }
    }

private:
    /// @see data
    // TODO(dwplc): RFD -- this replaced std::string
    // coverity[autosar_cpp14_a27_0_4_violation]
    const_pointer m_data;
    /// @see size
    size_type m_size;
};

/// Explicit instantiation of an uncommon template argument to get Coverity checks.
// TODO(dwplc): FP -- it is documented
// coverity[autosar_cpp14_a2_7_3_violation]
extern template class BasicStringView<char16_t, std::char_traits<char16_t>>;

/// Construct a string view directly from a string literal.
///
/// @note
/// Usage of this literal violates AUTOSAR rules A5-1-1 and A5-2-2, as Coverity's frontend interprets the use of a
/// user-defined literal as a C style cast of the @a str parameter and literally writes out the size of the string to
/// pass as the @a size parameter. If you are using it to create a constant, it will also cause violation of A3-3-2 and
/// A7-2-1, as this disables the @c constexpr nature of this function. These are all false positives, but note that
/// adding 4 lines of rule violation might not be worth it.
// TODO(dwplc): RFD -- this is just how user-defined literals work
// coverity[autosar_cpp14_a27_0_4_violation]
inline constexpr StringView operator"" _sv(char8_t const* const str, std::size_t const size) noexcept
{
    return StringView{str, size};
}

////////////////////////////////////////////////////////////////////////////////
// Non-member Comparison Boilerplate                                          //
// Each comparison function has three forms: one which accepts a string view  //
// for both parameters and then one overload for receiving each parameter in  //
// a non-deduced context. This is what allows a StringView to be compared to  //
// a BaseString or a string literal. AUTOSAR objects to this in A13-5-5,      //
// which states that non-member comparison operators should have identical    //
// types. However, std::basic_string_view requires this (as does programmer   //
// expectation) and the parameters are identical in the end, the signatures   //
// just have different deduction rules.                                       //
////////////////////////////////////////////////////////////////////////////////

/// Check if @a lhs is equal to @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator==(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    if (lhs.size() == rhs.size())
    {
        return lhs.compare(rhs) == detail::BasicStringViewImplValues::Zero<std::int32_t>::value;
    }
    else
    {
        return false;
    }
}

/// Check if @a lhs is equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator==(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return lhs == rhs;
}

/// Check if @a lhs is equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator==(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return lhs == rhs;
}

/// Check if @a lhs is not equal to @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator!=(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(lhs == rhs);
}

/// Check if @a lhs is not equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator!=(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return !(lhs == rhs);
}

/// Check if @a lhs is not equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator!=(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(lhs == rhs);
}

/// Check if @a lhs is less than @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator<(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return lhs.compare(rhs) < detail::BasicStringViewImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator<(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return lhs < rhs;
}

/// Check if @a lhs is less than @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator<(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return lhs < rhs;
}

/// Check if @a lhs is less than or equal to @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator<=(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(rhs < lhs);
}

/// Check if @a lhs is less than or equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator<=(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return !(rhs < lhs);
}

/// Check if @a lhs is less than or equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator<=(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(rhs < lhs);
}

/// Check if @a lhs is greater than @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator>(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return rhs < lhs;
}

/// Check if @a lhs is greater than @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator>(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return rhs < lhs;
}

/// Check if @a lhs is greater than @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator>(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return rhs < lhs;
}

/// Check if @a lhs is greater than or equal to @a rhs.
template <typename TChar, typename Traits>
constexpr bool operator>=(BasicStringView<TChar, Traits> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(lhs < rhs);
}

/// Check if @a lhs is greater than or equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator>=(BasicStringView<TChar, Traits> const lhs, meta::TypeIdentityT<BasicStringView<TChar, Traits>> const rhs) noexcept
{
    return !(lhs < rhs);
}

/// Check if @a lhs is greater than or equal to @a rhs.
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename TChar, typename Traits>
constexpr bool operator>=(meta::TypeIdentityT<BasicStringView<TChar, Traits>> const lhs, BasicStringView<TChar, Traits> const rhs) noexcept
{
    return !(lhs < rhs);
}

/// Check if @a lhs is equal to @a rhs.
constexpr bool operator==(StringView const& lhs, StringView const& rhs) noexcept
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (size_t i{0U}; i < lhs.size(); ++i)
    {
        // coverity[autosar_cpp14_m5_0_15_violation]
        if (lhs.data()[i] != rhs.data()[i])
        {
            return false;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Printing                                                                   //
////////////////////////////////////////////////////////////////////////////////

/// Print the @a str into the @a output.
template <typename TOutputStream, typename TChar, typename TCharTraits>
// TODO(dwplc): RFD -- allow references to be returned from << and output is an output parameter when left-hand side is
//              ostream-like
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a13_2_2_violation]
meta::BasicUnformattedOutputStreamFundamentalType<TOutputStream, TChar>& operator<<(TOutputStream& output, BasicStringView<TChar, TCharTraits> const& str)
{
    return output.write(str.data(), str.ssize());
}

/// Place @a v in the @a stream's buffer for logging.
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, StringView const& v);
/// Place @a v in the @a stream's buffer for logging.
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, StringView&& v);

/// Place @a v in the @a stream's buffer for logging.
// TODO(dwplc): RFD -- left hand side is std::ostream-like
// coverity[autosar_cpp14_a13_2_2_violation]
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, StringView const& v)
{
    return stream << v;
}

/// Place @a v in the @a stream's buffer for logging.
// TODO(dwplc): RFD -- left hand side is std::ostream-like
// coverity[autosar_cpp14_a13_2_2_violation]
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, StringView&& v)
{
    return stream << std::move(v);
}

} // namespace dw::core
} // namespace dw

namespace std
{
/// Specialization for std hash
// TODO(dwplc): FP -- std::hash specialization
// coverity[autosar_cpp14_a11_0_2_violation]
template <typename TChar>
struct hash<dw::core::BasicStringView<TChar>>
{
    template <class T>
    size_t operator()(const T& x) const noexcept
    {
        return typename dw::core::BasicStringView<TChar>::Hash()(x);
    }

    template <class T>
    size_t combine(const T& x, size_t const baseHash) const noexcept
    {
        return typename dw::core::BasicStringView<TChar>::Hash().combine(x, baseHash);
    }
};

} // namespace std

#endif /*DWSHARED_CORE_CONTAINER_STRINGVIEW_HPP_*/
