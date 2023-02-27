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

#ifndef DWSHARED_CORE_BASESTRING_HPP_
#define DWSHARED_CORE_BASESTRING_HPP_

#include <cmath>
#include <array>
#include <istream>

#include <dw/core/safety/MathErrors.hpp>
#include <dw/core/language/TypeAliases.hpp>
#include <dw/core/language/cxx17.hpp>

#include "ContiguousIterator.hpp"
#include "StringView.hpp"

namespace dw
{
namespace core
{

/// Forward declaration of Optional to avoid circular dependency
template <class T>
class Optional;

namespace detail
{

/// Wrapper around safePow so that Safety.hpp doesn't need to be included in this file
float64_t safePowImpl(float64_t const base, int32_t const exp);

/// Wrapper around safePow so that Safety.hpp doesn't need to be included in this file
float64_t safePowImpl(float64_t const base, uint32_t const exp);

/// Wrapper around safeIncrement so that Safety.hpp doesn't need to be included in this file
uint64_t safeIncrementImpl(uint64_t var, uint64_t const incr);

/// @throw OutOfBoundsException
[[noreturn]] void throwBaseStringIndexOutOfBounds(StringView const& operation, std::size_t const size, std::size_t const offset);

/// This structure holds the names of operations and other values used by the @c BaseString implementation. It
/// exists for compliance with AUTOSAR A5-1-1, which states that only symbolic names are acceptable.
struct BaseStringImplValues
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

    /// @c 10
    // TODO(dwplc): FP -- Coverity is complaining about unbraced initialization
    // coverity[autosar_cpp14_a8_5_2_violation]
    template <typename T, std::enable_if_t<std::is_integral<T>::value, T> Value = T{10}>
    using Ten = std::integral_constant<T, Value>;

    /// @c \0
    // TODO(dwplc): FP -- Coverity is complaining about unbraced initialization
    // coverity[autosar_cpp14_a8_5_2_violation]
    template <typename T, std::enable_if_t<std::is_integral<T>::value, T> Value = T{'\0'}>
    using NulChar = std::integral_constant<T, Value>;

    /// @c "at"
    static StringView const OP_NAME_AT;

    /// @c "operator[]"
    static StringView const OP_NAME_OPERATOR_SUBSCRIPT;

    /// @c "front"
    static StringView const OP_NAME_FRONT;

    /// @c "back"
    static StringView const OP_NAME_BACK;

    /// @c "substr"
    static StringView const OP_NAME_SUBSTR;

    /// @c "nan"
    static StringView const FLOAT_VAL_NAN;

    /// @c "inf"
    static StringView const FLOAT_VAL_INF;

    /// @c "-inf"
    static StringView const FLOAT_VAL_NEG_INF;
};

// TODO(hlanker): remove this when dw::serialization has been removed.

/// This trait serves as a workaround to distinguish the dw::serialization::AnyArchive
/// from dw::serial archives, without having to add a dependency on dw::serialization here.
/// Moving BaseString::serialize out into a standalone function in dw::serialization and using
/// the std::is_base_of<AnyArchive, T> has caused the problem of wrong functions being called.
/// Therefore adding this solution here.
///
/// @{

/// See above
template <template <class...> class Trait, class AlwaysVoid, class... Args>
struct Detector : std::false_type
{
};

/// See above
template <template <class...> class Trait, class... Args>
struct Detector<Trait, void_t<Trait<Args...>>, Args...> : std::true_type
{
};

/// See above
template <typename T>
using serializeString_method_t = decltype(std::declval<T&>().serializeString(nullptr, 0U, 0U));

/// See above
template <typename T>
using has_serializeString_t = typename Detector<serializeString_method_t, void, T>::type;
/// @}
} // namespace detail

/// String class that stores characters in a std::array. This is meant to be a replacement for std::string that
/// doesn't perform runtime allocations.
///
/// The major difference between this class and std::string is that this class will silently truncate when attempting
/// to assign or append a string that is longer than the @c BufferSize.
///
/// The following funtions from std::string are currently unimplemented:
/// insert, erase, pop_back, starts_with, ends_with, contains, replace, copy, swap
// TODO(dwplc): FP -- There is a copy assignment operator explicitly declared in this class
// coverity[autosar_cpp14_m14_5_3_violation]
template <size_t BufferSize, typename TChar = char8_t>
class BaseString
{
public:
    /// The value type of a single character.
    // TODO(dwplc): FP -- using things like char8_t here violate this rule here, but not at the use site
    // coverity[autosar_cpp14_a3_9_1_violation]
    using value_type = TChar;

    /// A pointer to a character.
    // TODO(dwplc): RFD -- This class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    using pointer = value_type*;

    /// A constant pointer to a character.
    // TODO(dwplc): RFD -- This class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    using const_pointer = value_type const*;

    /// A reference to a character.
    using reference = value_type&;

    /// A constant reference to a character.
    using const_reference = value_type const&;

    /// An iterator which moves forwards.
    using const_iterator = BasicContiguousIterator<value_type const, BaseString<BufferSize, TChar> const>;

    /// An iterator which moves forwards.
    using iterator = BasicContiguousIterator<value_type, BaseString<BufferSize, TChar>>;

    /// An iterator which moves backwards.
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// An iterator which moves backwards.
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Describes the @ref size of the string.
    using size_type = std::size_t;

    /// Represents not-a-position in search operations.
    static constexpr size_type NPOS{std::numeric_limits<size_type>::max()}; // clang-tidy NOLINT(readability-identifier-naming)

    /// The maximum length string that can be stored in this object.
    static size_type const MAX_LENGTH;

    /// The capacity of the underlying buffer of this object.
    static size_type const CAPACITY;

public:
    /// Constructs an empty BaseString
    constexpr BaseString() noexcept // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString(detail::BaseStringImplValues::NulChar<TChar>::value, detail::BaseStringImplValues::Zero<size_type>::value)
    {
    }

    /// Create an instance from the contents of @a str.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a18_1_1_violation]
    template <size_t N>
    inline BaseString(const TChar (&str)[N]) noexcept // clang-tidy NOLINT(google-explicit-constructor, cppcoreguidelines-pro-type-member-init) - we allow implicit converion for strings of known size at compilation time
        : BaseString()
    {
        static_assert(CAPACITY >= N, "Cannot implicitly put a string larger than capacity into FixedString");
        initializeFrom(str, N);
    }

    /// Create an instance from the contents of @a source. If the given @a source has a larger size than this buffer, it
    /// is silently truncated. Note that in cases where @c source.size() is the same as @c CAPACITY, truncation will
    /// still occur, as @c BaseString needs one additional byte for the @c NUL character -- use @c MAX_LENGTH for checks.
    /// Additionally, if @a source contains @c NUL characters, only the content up to the first @c NUL character is
    /// copied -- the remainder of the string is silently truncated.
    explicit BaseString(const BasicStringView<TChar> source) noexcept // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString()
    {
        initializeFrom(source.data(), source.size());
    }

    /// Create an instance from the contents of @a str. If the given @a str has a larger size than this buffer, it
    /// is silently truncated. Note that in cases where the length of @a str is the same as @c CAPACITY, truncation will
    /// still occur, as @c BaseString needs one additional byte for the @c NUL character.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    explicit inline BaseString(const_pointer const str) noexcept // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString()
    {
        try
        {
            initializeFrom(str);
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    /// Create an instance from the contents of @a str, starting at @a start and including up to @a initSize characters.
    /// If @a str is null, @a initSize is zero, or @a start is greater than the length of @a str, the instance will be empty.
    /// if @a initSize is larger than this buffer, it is silently truncated. Note that in cases where @a initSize is the same
    /// as @c CAPACITY, truncation will still occur, as @c BaseString needs one additional byte for the @c NUL character.
    // TODO(dwplc): RFD -- this class replaces C strings A27-0-4
    // coverity[autosar_cpp14_a27_0_4_violation]
    explicit inline BaseString(const_pointer const str, size_type const initSize, size_type const start = detail::BaseStringImplValues::Zero<size_type>::value) noexcept(false) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString()
    {
        // If the provided string is empty, or initSize is zero, initialize as an empty string
        if ((str == nullptr) || (initSize == detail::BaseStringImplValues::Zero<size_type>::value))
        {
            clear();
            return;
        }

        BasicStringView<TChar> const strView{str};

        // If start is greater than the length of the provided string, initialize an empty string
        if (strView.length() <= start)
        {
            clear();
            return;
        }

        // It's now safe to copy the characters from start
        initializeFrom(&strView[start], initSize);
    }

    /// Create an instance from the contents of @a str, starting at @a start and including up to @a initSize characters.
    /// If @a str is empty, @a initSize is zero, or @a start is greater than the length of @a str, the instance will be empty.
    /// if @a initSize is larger than this buffer, it is silently truncated. Note that in cases where @a initSize is the same
    /// as @c CAPACITY, truncation will still occur, as @c BaseString needs one additional byte for the @c NUL character.
    template <std::size_t OtherBufferSize>
    explicit inline BaseString(const BaseString<OtherBufferSize, TChar>& str, size_type initSize, size_type const start = detail::BaseStringImplValues::Zero<size_type>::value) noexcept(false) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString(str.c_str(), initSize, start)
    {
    }

    /// Create an instance from the contents of @a str
    template <std::size_t OtherBufferSize>
    explicit BaseString(const BaseString<OtherBufferSize, TChar>& str) // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init) -- forwarding constructor
        : BaseString(str.c_str())
    {
        static_assert(BufferSize >= OtherBufferSize,
                      "Cannot directly assign larger fixed string to smaller, looses content. "
                      "Use FixedString<>(otherStr.c_str()) to force assignemnt wihout a size check");
    }

    /// Copy constructor
    BaseString(BaseString const&) noexcept = default;

    /// Move constructor
    BaseString(BaseString&&) noexcept = default;

    /// Destructor
    ~BaseString() noexcept = default;

    /// Replaces the contents of the string.
    BaseString& operator=(BaseString const&) noexcept = default;

    /// Replaces the contents of the string.
    BaseString& operator=(BaseString&&) noexcept = default;

    /// Replaces the contents of the string.
    /// If the length of @c other is greater than the size of this string, it will be silenty truncated.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto operator=(const_pointer const other) noexcept -> BaseString&
    {
        initializeFrom(other);
        return *this;
    }

    /// Replaces the contents of the string.
    /// If the length of @c other is greater than the size of this string, it will be silenty truncated.
    template <size_t OtherBufferSize>
    auto operator=(const BaseString<OtherBufferSize, TChar>& other) noexcept -> BaseString&
    {
        initializeFrom(other.data(), other.size());
        return *this;
    }

    /// Copies @c other to this string. The copy will start at the beginning of this string.
    /// Copies at most @c count characters but will stop if '\0' is found.
    /// If count is zero or @c NPOS, it copies until '\0' is found.
    // TODO(dwplc): RFD -- this class replaces C strings A27-0-4
    // coverity[autosar_cpp14_a27_0_4_violation]
    inline void copyFrom(const_pointer const other, size_type const count = NPOS) noexcept
    {
        // For historical reasons, zero would indicate to copy the whole string. Internally this code
        // now uses NPOS to indicate that the entire string should be copied.
        size_type const toCopy{(count == detail::BaseStringImplValues::Zero<size_type>::value) ? NPOS : count};
        initializeFrom(other, toCopy);
    }

    /// Returns a reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than @ref size.
    auto at(size_type const pos) noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_AT, pos);
    }

    /// Returns a const reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than @ref size.
    auto at(size_type const pos) const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_AT, pos);
    }

    /// Returns a reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than @ref size.
    auto operator[](size_type const pos) noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_OPERATOR_SUBSCRIPT, pos);
    }

    /// Returns a reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than @ref size.
    auto operator[](size_type const pos) const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_OPERATOR_SUBSCRIPT, pos);
    }

    /// Returns a reference to the first character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is @ref empty
    auto front() noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_FRONT, detail::BaseStringImplValues::Zero<size_type>::value);
    }

    /// Returns a reference to the first character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is @ref empty
    auto front() const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_FRONT, detail::BaseStringImplValues::Zero<size_type>::value);
    }

    /// Returns reference to the last character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is @ref empty
    auto back() noexcept(false) -> reference
    {
        if (empty())
        {
            detail::throwBaseStringIndexOutOfBounds(detail::BaseStringImplValues::OP_NAME_BACK,
                                                    detail::BaseStringImplValues::Zero<size_type>::value,
                                                    detail::BaseStringImplValues::Zero<size_type>::value);
        }

        return at_impl(detail::BaseStringImplValues::OP_NAME_BACK, size() - detail::BaseStringImplValues::One<size_type>::value);
    }

    /// Returns reference to the last character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is @ref empty
    auto back() const noexcept(false) -> const_reference
    {
        if (empty())
        {
            detail::throwBaseStringIndexOutOfBounds(detail::BaseStringImplValues::OP_NAME_BACK,
                                                    detail::BaseStringImplValues::Zero<size_type>::value,
                                                    detail::BaseStringImplValues::Zero<size_type>::value);
        }

        return at_impl(detail::BaseStringImplValues::OP_NAME_BACK, size() - detail::BaseStringImplValues::One<size_type>::value);
    }

    /// Returns a pointer to the first character in the string.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto data() noexcept -> pointer
    {
        return m_data.data();
    }

    /// Returns a pointer to the first character in the string.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto data() const noexcept -> const_pointer
    {
        return m_data.data();
    }

    /// Returns a pointer to the first character in the string.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto c_str() noexcept -> pointer
    {
        return m_data.data();
    }

    /// Returns a pointer to the first character in the string.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto c_str() const noexcept -> const_pointer
    {
        return m_data.data();
    }

    /// Returns a BasicStringView of the underlying string.
    // TODO(dwplc): RFD -- This operator is intentionally implicit to match the C++ standard (https://eel.is/c++draft/basic.string#string.accessors-7)
    // coverity[autosar_cpp14_a13_5_2_violation]
    operator BasicStringView<TChar>() const noexcept // clang-tidy NOLINTNEXTLINE(google-explicit-constructor) -- implicit conversion is intentional
    {
        return BasicStringView<TChar>(data(), size());
    }

    /// Get an iterator to the first character in the string.
    auto begin() noexcept -> iterator
    {
        return iterator{data(), &m_data[size()]};
    }

    /// Get a const iterator to the first character in the string.
    auto begin() const noexcept -> const_iterator
    {
        return const_iterator{data(), &m_data[size()]};
    }

    /// Get a const iterator to the first character in the string.
    auto cbegin() const noexcept -> const_iterator
    {
        return const_iterator{data(), &m_data[size()]};
    }

    /// Get an iterator to one past the last character in the string.
    auto end() noexcept -> iterator
    {
        return begin() + size();
    }

    /// Get an iterator to one past the last character in the string.
    auto end() const noexcept -> const_iterator
    {
        return begin() + size();
    }

    /// Get an iterator to one past the last character in the string.
    auto cend() const noexcept -> const_iterator
    {
        return begin() + size();
    }

    /// Get an iterator to the last character of the string, going backwards.
    auto rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator{end()};
    }

    /// Get an iterator to the last character of the string, going backwards.
    auto rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator{end()};
    }

    /// Get an iterator to the last character of the string, going backwards.
    auto crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator{end()};
    }

    /// Get an iterator to one character before the first character in the string, going backwards.
    auto rend() noexcept -> reverse_iterator
    {
        return reverse_iterator{begin()};
    }

    /// Get an iterator to one character before the first character in the string, going backwards.
    auto rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator{begin()};
    }

    /// Get an iterator to one character before the first character in the string, going backwards.
    auto crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator{begin()};
    }

    /// Checks if the string has no characters.
    constexpr bool empty() const noexcept
    {
        return size() == detail::BaseStringImplValues::Zero<size_type>::value;
    }

    /// Checks if there is no more free space available in the string
    constexpr bool full() const noexcept
    {
        return m_length >= MAX_LENGTH;
    }

    /// Returns the number of elements in the string.
    constexpr size_type size() const noexcept
    {
        return m_length;
    }

    /// Returns the number of elements in the string.
    constexpr size_type length() const noexcept
    {
        return size();
    }

    /// Returns the maximum number of elements that the string can hold.
    auto max_size() const noexcept -> size_type
    {
        return MAX_LENGTH;
    }

    /// Returns the amount of space allocated for this string
    constexpr size_type capacity() const noexcept
    {
        return CAPACITY;
    }

    /// Removes all characters from the string
    constexpr void clear()
    {
        m_length                                                        = detail::BaseStringImplValues::Zero<size_type>::value;
        m_data.at(detail::BaseStringImplValues::Zero<size_type>::value) = detail::BaseStringImplValues::NulChar<TChar>::value;
    }

    /// @{
    /// Compare this string view to @a other lexicographically
    ///
    /// @return @c 0 if this view is equal-to @a other, @c <0 if this view is less-than @a other, or @c >0 if this view
    ///         is greater-than @a other. Note that relying on @c -1 and @c 1 for the @c <0 and @c >0 values is
    ///         incorrect.
    template <size_t OtherBufferSize>
    std::int32_t compare(const BaseString<OtherBufferSize, TChar>& other) const noexcept
    {
        return toStringView(*this).compare(toStringView(other));
    }

    /// Equivalent to <tt>substr(position, count).compare(other)</tt>.
    template <size_t OtherBufferSize>
    std::int32_t compare(size_type const position, size_type const count, const BaseString<OtherBufferSize, TChar>& other) const noexcept(false)
    {
        return toStringView(*this).compare(position, count, toStringView(other));
    }

    /// Equivalent to <tt>substr(position, count).compare(other.substr(otherPosition, otherCount))</tt>.
    template <size_t OtherBufferSize>
    std::int32_t compare(size_type const position,
                         size_type const count,
                         const BaseString<OtherBufferSize, TChar>& other,
                         size_type const otherPosition,
                         size_type const otherCount) const noexcept(false)
    {
        return toStringView(*this).compare(position, count, toStringView(other), otherPosition, otherCount);
    }

    /// Equivalent to <tt>compare(BaseString(other))</tt>.
    // coverity[autosar_cpp14_a27_0_4_violation]
    std::int32_t compare(const_pointer const other) const noexcept(false)
    {
        return toStringView(*this).compare(other);
    }

    /// Equivalent to <tt>substr(position, count).compare(BaseString(other))</tt>.
    // coverity[autosar_cpp14_a27_0_4_violation]
    std::int32_t compare(size_type const position, size_type const count, const_pointer const other) const noexcept(false)
    {
        return toStringView(*this).compare(position, count, other);
    }

    /// Equivalent to <tt>substr(position, count).compare(BaseString(other, otherLength))</tt>.
    // coverity[autosar_cpp14_a27_0_4_violation]
    std::int32_t compare(size_type const position,
                         size_type const count,
                         const_pointer const other,
                         size_type const otherLength) const noexcept(false)
    {
        return toStringView(*this).compare(position, count, other, otherLength);
    }

    /// @}

    /// Appends the given character @a ch to the string.
    ///
    /// @note
    /// Unlike std::basic_string::push_back, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently do nothing. This is to preserve the
    /// legacy behavior of this class. In the future it may be desirable to throw @c BufferFullException in this case.
    void push_back(value_type const ch) noexcept
    {
        try
        {

            if (m_length < MAX_LENGTH)
            {
                m_data.at(m_length) = ch;
                m_length++;
                m_data.at(m_length) = detail::BaseStringImplValues::NulChar<TChar>::value;
            }
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    /// Append a string to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    template <size_t OtherBufferSize>
    auto operator+=(const BaseString<OtherBufferSize, TChar>& str) noexcept -> BaseString<BufferSize, TChar>&
    {
        return this->operator+=(str.data());
    }

    /// Append a string view to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(BasicStringView<TChar> const& src) noexcept -> BaseString<BufferSize, TChar>&
    {
        return append(src.data(), src.size());
    }

    /// Append a string to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto operator+=(const_pointer const str) noexcept -> BaseString<BufferSize, TChar>&
    {

        if (str == nullptr)
        {
            return *this;
        }

        appendFrom(str, NPOS);

        return *this;
    }

    /// Appends the given character @a ch to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently do nothing. This is to preserve the
    /// legacy behavior of this class. In the future it may be desirable to throw @c BufferFullException in this case.
    auto operator+=(value_type const ch) noexcept -> BaseString<BufferSize, TChar>&
    {
        try
        {
            if (full())
            {
                return *this;
            }

            push_back(ch);

            return *this;
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
            return *this;
        }
    }

    /// Appends the given integer @c inumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(int64_t inumber) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        try
        {

            uint64_t unumber{0UL};
            if (inumber < detail::BaseStringImplValues::Zero<int64_t>::value)
            {
                constexpr TChar DASH{'-'};
                this->operator+=(DASH);
                if (inumber == std::numeric_limits<int64_t>::min())
                {
                    inumber += detail::BaseStringImplValues::One<int64_t>::value;
                    inumber = -inumber;
                    unumber = static_cast<uint64_t>(inumber);
                    unumber += detail::BaseStringImplValues::One<uint64_t>::value;
                }
                else
                {
                    inumber = -inumber;
                    unumber = static_cast<uint64_t>(inumber);
                }
            }
            else
            {
                unumber = static_cast<uint64_t>(inumber);
            }

            this->operator+=(unumber);

            return *this;
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
            return *this;
        }
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint64_t unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        try
        {

            if (full())
            {
                return *this;
            }

            // output number to string (this will output in reverse order)
            size_type startIdx{m_length};
            constexpr uint8_t ZERO_CHAR{static_cast<uint8_t>('0')};
            constexpr uint64_t BASE{10ULL};
            do
            {
                uint8_t const digit{static_cast<uint8_t>(unumber % BASE)};
                unumber /= BASE;

                push_back(static_cast<TChar>(digit + ZERO_CHAR));
            } while ((unumber > detail::BaseStringImplValues::Zero<uint64_t>::value) && (m_length < MAX_LENGTH));

            size_t lastIdx{m_length - detail::BaseStringImplValues::One<size_type>::value};

            // reverse number to show in correct order
            while ((startIdx < lastIdx))
            {
                std::swap(m_data.at(startIdx), m_data.at(lastIdx));
                startIdx++;
                lastIdx--;
            }

            return *this;
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
            return *this;
        }
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(float64_t const fnumber) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        constexpr uint32_t DECIMAL_PLACES{4UL};
        return this->append(fnumber, DECIMAL_PLACES);
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(float32_t const fnumber) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        this->operator+=(static_cast<float64_t>(fnumber));
        return *this;
    }

    /// Appends the given integer @c inumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(int32_t const inumber) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        this->operator+=(static_cast<int64_t>(inumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint16_t const unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        this->operator+=(static_cast<uint64_t>(unumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint32_t const unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        this->operator+=(static_cast<uint64_t>(unumber));
        return *this;
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @param fnumber The number to append
    /// @param decimalPlaces  The number of decimal places to append
    /// @param useScientific Whether or not to append in scientific notation
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto append(float64_t fnumber, uint32_t const decimalPlaces, bool useScientific = false) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        // handle nan, +/-inf
        if (appendHandleSpecialCases(fnumber))
        {
            return *this;
        }

        constexpr float64_t ZERO_F{0.0};
        constexpr float64_t TEN_F{10.0};

        if (fnumber < ZERO_F)
        {
            constexpr TChar DASH_CHAR{'-'};
            constexpr float64_t NEGATIVE_ONE_F{-1.0};
            this->operator+=(DASH_CHAR);
            fnumber *= NEGATIVE_ONE_F;
        }

        // Use a copy of fnumber for calculation so that we can keep the original fnumber for use later.
        // This is done because the 'exponent' value is needed in both blocks when useScientific is true.
        // It would be nice to simply declare the exponent variable here so it can be used in both blocks,
        // but doing so is an Autosar violation because the variable is not used when useScientific is false.
        float64_t floatVal{fnumber};

        // Do not use scientific notation for 0 (-inf exponent)
        // Do always use scientific notation for huge numbers
        useScientific = useScientific && (floatVal > ZERO_F);
        useScientific = useScientific || (std::fabs(floatVal) > static_cast<float64_t>(std::numeric_limits<uint32_t>::max()));

        if (useScientific)
        {
            float64_t const flog10{std::log10(floatVal)};
            dw::core::resetMathErrors();
            // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
            // coverity[cert_flp34_c_violation]
            int32_t const exponent{static_cast<int32_t>(std::floor(flog10))};
            float64_t const fpow10{detail::safePowImpl(TEN_F, exponent)};
            dw::core::resetMathErrors();
            floatVal /= fpow10; // floatVal is now in format a.xxxxxx where 1 <= a < 10
        }

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        uint64_t unumber{static_cast<uint64_t>(std::floor(floatVal))};
        float64_t const decimal10{detail::safePowImpl(TEN_F, decimalPlaces)};
        dw::core::resetMathErrors();

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        uint64_t floatingPart{static_cast<uint64_t>(std::round((floatVal - std::floor(floatVal)) * decimal10))};

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        if (floatingPart == static_cast<uint64_t>(decimal10))
        {
            // The decimals rounded up to the next unit
            unumber += detail::BaseStringImplValues::One<uint64_t>::value;
            floatingPart -= static_cast<uint64_t>(decimal10);
        }

        this->operator+=(unumber);

        constexpr TChar DOT_CHAR{'.'};
        this->operator+=(DOT_CHAR);

        // Count how many zeroes should be added in the beginning. This handles cases like 0.0039.
        uint64_t digitCount{0UL};
        uint64_t tmpFloatingPart{floatingPart};
        while (tmpFloatingPart > detail::BaseStringImplValues::Zero<uint64_t>::value)
        {
            digitCount = detail::safeIncrementImpl(digitCount, detail::BaseStringImplValues::One<uint64_t>::value);
            constexpr uint64_t TEN_ULL{10ULL};
            tmpFloatingPart /= TEN_ULL;
        }

        // Add zeroes
        while (digitCount < decimalPlaces)
        {
            constexpr TChar ZERO_CHAR{'0'};
            digitCount++;
            this->operator+=(ZERO_CHAR);
        }

        // Add floating part if non-zero
        if (floatingPart > 0U)
        {
            this->operator+=(floatingPart);
        }

        if (useScientific)
        {
            constexpr TChar E_CHAR{'e'};
            this->operator+=(E_CHAR);

            // Recalculate exponent
            float64_t const flog10{std::log10(fnumber)};
            dw::core::resetMathErrors();
            // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
            // coverity[cert_flp34_c_violation]
            int32_t const exponent{static_cast<int32_t>(std::floor(flog10))};

            if (exponent > detail::BaseStringImplValues::Zero<int32_t>::value)
            {
                constexpr TChar PLUS_CHAR{'+'};
                this->operator+=(PLUS_CHAR);
            }
            this->operator+=(exponent);
        }

        return *this;
    }

    // -----------------------------------------------------------------------------
    /// Append @a src to this string. If the source is too long to fit into this string, it is silently truncated.
    ///
    /// @param src The source string to append -- it does not need to be null terminated.
    /// @param srcSize The size of @a src.
    /// @note
    /// Unlike std::basic_string::append, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto append(const_pointer const src, size_type const srcSize) noexcept -> BaseString<BufferSize, TChar>&
    {
        if (src == nullptr)
        {
            return *this;
        }

        appendFrom(src, srcSize);

        return *this;
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @param fnumber The number to append
    /// @param decimalPlaces  The number of decimal places to append
    /// @param useScientific Whether or not to append in scientific notation
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto append(float32_t const fnumber, uint32_t const decimalPlaces, const bool useScientific = false) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        return this->append(static_cast<float64_t>(fnumber), decimalPlaces, useScientific);
    }

    /// Returns a substring [pos, pos+count). If the requested substring extends past the end of the string,
    /// or if count == NPOS, the returned substring is [pos, size())
    /// Template parameter @c SubStringSize will be incremented by one to account for the null character
    /// Throws an @c OutOfBoundsException if @a pos is larger than @ref size.
    template <size_t SubStringSize = BufferSize - 1>
    auto substr(size_type const pos = 0, size_type const count = NPOS) const -> BaseString<SubStringSize + 1, TChar>
    {
        if (pos > size())
        {
            detail::throwBaseStringIndexOutOfBounds(detail::BaseStringImplValues::OP_NAME_SUBSTR,
                                                    pos,
                                                    size());
        }

        return BaseString<SubStringSize + 1, TChar>(c_str(), count, pos);
    }

    /// Truncates the string to the new size.
    /// If @a count is greater than @ref length, this function does nothing.
    void truncate(size_type const count) noexcept
    {
        if (length() <= count)
        {
            return;
        }

        m_length         = count;
        m_data.at(count) = detail::BaseStringImplValues::NulChar<TChar>::value;
    }

    /// Resizes the string to contain @a count characters.
    /// If @a count is greater than @a MAX_LENGTH, it will be set to @a MAX_LENGTH.
    ///
    /// @note
    /// Unlike std::string, if @a count is greater than @ref length, the characters between @ref length
    /// and @a count will not be set to null characters. They will be left unchanged to preserve the legacy behavior
    /// of this class. Only the existing null terminator will be changed to a space character.
    void resize(size_type const count) noexcept
    {
        size_type const newLength{std::min(count, MAX_LENGTH)};

        constexpr TChar SPACE{' '};
        m_data.at(m_length) = SPACE;

        m_length             = newLength;
        m_data.at(newLength) = detail::BaseStringImplValues::NulChar<TChar>::value;
    }

    /// Find the first occurance of the @a ch character.
    ///
    /// @param ch The character to search for.
    /// @param pos The first position the @a ch character can be at.
    ///
    /// @returns The index of the first occurance of @a ch in this instance after @a pos or @c NPOS if this
    ///          instance does not contain @a ch after @a pos.
    auto find(value_type const ch, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find(ch, pos);
    }

    /// Find the first occurance of the @a str character sequence.
    ///
    /// @param str The character sequence to search for.
    /// @param pos The first position the @a str character sequence can start at.
    ///
    /// @returns The index of the first character of @a str in this instance after @a pos or @c NPOS if this
    ///          instance does not contain @a str after @a pos.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto find(const_pointer const str, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find(str, pos);
    }

    /// Find the first occurance of the @a str character sequence.
    ///
    /// @param str The character sequence to search for.
    /// @param pos The first position the @a str character sequence can start at.
    ///
    /// @returns The index of the first character of @a str in this instance after @a pos or @c NPOS if this
    ///          instance does not contain @a str after @a pos.
    template <size_t OtherBufferSize>
    auto find(const BaseString<OtherBufferSize, TChar>& str, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        return find(str.c_str(), pos);
    }

    /// Find the last occurance of the @a ch character.
    ///
    /// @param ch The character to search for.
    /// @param pos The last position the @a ch character can be at.
    ///
    /// @returns The index of the lost occurance of @a ch in this instance or @c NPOS if this
    ///          instance does not contain @a ch before @a pos.
    auto rfind(value_type const ch, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).rfind(ch, pos);
    }

    /// Find the last occurance of the @a str character sequence.
    ///
    /// @param str The character sequence to search for.
    /// @param pos The last position the @a str character sequence can be.
    ///
    /// @returns The index of the first character of the last occurance @a str in this instance or @c NPOS if this
    ///          instance does not contain @a str before @a pos.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto rfind(const_pointer const str, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).rfind(str, pos);
    }

    /// Find the last occurance of the @a str character sequence.
    ///
    /// @param str The character sequence to search for.
    /// @param pos The last position the @a str character sequence can be.
    ///
    /// @returns The index of the first character of the last occurance @a str in this instance or @c NPOS if this
    ///          instance does not contain @a str before @a pos.
    template <size_t OtherBufferSize>
    auto rfind(const BaseString<OtherBufferSize, TChar>& str, size_type const pos = NPOS) const noexcept -> size_type
    {
        return rfind(str.c_str(), pos);
    }

    /// Find the first character which matches the @a search character, starting at @a pos.
    ///
    /// @param search The character to search for.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches @a search or @c NPOS if none of the
    ///          characters match.
    auto find_first_of(value_type const search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_first_of(search, pos);
    }

    /// Find the first character which matches any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for matches.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches in @a search or @c NPOS if none of the
    ///          characters match.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto find_first_of(const_pointer const search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_first_of(search, pos);
    }

    /// Find the first character which matches any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for matches.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which matches in @a search or @c NPOS if none of the
    ///          characters match.
    template <size_t OtherBufferSize>
    auto find_first_of(const BaseString<OtherBufferSize, TChar>& search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        return find_first_of(search.c_str(), pos);
    }

    /// Find the first character which does not match the @a search character, starting at @a pos.
    ///
    /// @param search The character to search for.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which does not match @a search or @c NPOS all of the
    ///          characters match.
    auto find_first_not_of(value_type const search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_first_not_of(search, pos);
    }

    /// Find the first character which does not match any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which does not match a character in @a search
    ///          or @c NPOS if all of the characters match.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto find_first_not_of(const_pointer const search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_first_not_of(search, pos);
    }

    /// Find the first character which does not match any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param pos The first index to search for.
    ///
    /// @returns The index of the first character in this instance which does not match a character in @a search
    ///          or @c NPOS if all of the characters match.
    template <size_t OtherBufferSize>
    auto find_first_not_of(const BaseString<OtherBufferSize, TChar>& search, size_type const pos = detail::BaseStringImplValues::Zero<size_type>::value) const noexcept -> size_type
    {
        return find_first_not_of(search.c_str(), pos);
    }

    /// Find the last character which matches the @a search character, starting at @a pos.
    ///
    /// @param search The character to search for.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches @a search or @c NPOS if none of the
    ///          characters match.
    auto find_last_of(value_type const search, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_last_of(search, pos);
    }

    /// Find the last character which matches any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for matches.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches in @a search or @c NPOS if none of the
    ///          characters match.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto find_last_of(const_pointer const search, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_last_of(search, pos);
    }

    /// Find the last character which matches any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for matches.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which matches in @a search or @c NPOS if none of the
    ///          characters match.
    template <size_t OtherBufferSize>
    auto find_last_of(const BaseString<OtherBufferSize, TChar>& search, size_type const pos = NPOS) const noexcept -> size_type
    {
        return find_last_of(search.c_str(), pos);
    }

    /// Find the last character which does not match the @a search character, starting at @a pos.
    ///
    /// @param search The character to search for.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which does not match @a search or @c NPOS all of the
    ///          characters match.
    auto find_last_not_of(value_type const search, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_last_not_of(search, pos);
    }

    /// Find the last character which does not match any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which does not match a character in @a search
    ///          or @c NPOS if all of the characters match.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto find_last_not_of(const_pointer const search, size_type const pos = NPOS) const noexcept -> size_type
    {
        // To directly return BasicStringView::npos from this function, it must be equal to BaseString::NPOS
        static_assert(NPOS == BasicStringView<TChar>::npos, "BaseString::NPOS and BasicStringView::npos must be equal");
        return toStringView(*this).find_last_not_of(search, pos);
    }

    /// Find the last character which does not match any of the characters in the @a search query, starting at
    /// @a pos.
    ///
    /// @param search The character sequence to check for non-matches.
    /// @param pos The last index to search for.
    ///
    /// @returns The index of the last character in this instance which does not match a character in @a search
    ///          or @c NPOS if all of the characters match.
    template <size_t OtherBufferSize>
    auto find_last_not_of(const BaseString<OtherBufferSize, TChar>& search, size_type const pos = NPOS) const noexcept -> size_type
    {
        return find_last_not_of(search.c_str(), pos);
    }

    /// Serialize this string to the provided archive
    template <class TArchive,
              typename U = typename std::enable_if<detail::has_serializeString_t<TArchive>::value, TArchive>::type> // disable this for archives from dw::serial
    void
    serialize(TArchive& archive)
    {
        size_type size_{};
        if (archive.isOutput())
        {
            size_ = m_length;
            archive.prepareForString(size_);
            archive.serializeString(c_str(), capacity(), size_);
        }
        else
        {
            archive.prepareForString(size_);
            if (!archive.isValidating())
            {
                m_length = size_;
            }
            archive.serializeString(c_str(), capacity(), size_);
        }
    }

    /// Helper class for hashing strings, using djb2 algorithm
    class Hash
    {
    public:
        /// Hashes the given string
        size_t operator()(const BaseString<BufferSize, TChar>& str) const noexcept
        {
            return hashImpl(toStringView(str));
        }

        /// Hashes the given string
        // TODO(dwplc): RFD -- this class replaces C strings
        // coverity[autosar_cpp14_a27_0_4_violation]
        size_t operator()(const_pointer const str) const noexcept
        {
            return hashImpl(BasicStringView<TChar>(str));
        }

    private:
        /// Implementation of the djb2 hash algorithm
        size_t hashImpl(const BasicStringView<TChar>& strView) const noexcept
        {
            try
            {
                /// Base of the djb2 hash function
                constexpr size_t HASH_BASE{5381UL};

                size_t hash{HASH_BASE};
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
                return detail::BaseStringImplValues::Zero<size_t>::value;
            }
        }
    };

private:
    /// Delegating constructor to be used by the public constructors
    explicit constexpr BaseString(TChar const c, size_type const length) noexcept
        : m_data{c}
        , m_length(length)
    {
    }

    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    inline void appendFrom(const_pointer const other, size_type const count) noexcept
    {
        // The usage of operator[] in this function could throw, but never would because all access
        // is within bounds. Add a try/catch anyway so that this function can be marked noexcept.
        try
        {
            if (full())
            {
                return;
            }

            size_type const remainingSpace{(MAX_LENGTH - m_length)};
            size_type const copyCount{std::min(remainingSpace, count)};
            const BasicStringView<TChar> view{other, copyCount};

            size_type i{0UL};
            for (; i < copyCount; ++i)
            {
                if (view[i] == detail::BaseStringImplValues::NulChar<TChar>::value)
                {
                    break;
                }
                m_data.at(m_length + i) = view[i];
            }

            m_length += i;
            m_data.at(m_length) = detail::BaseStringImplValues::NulChar<TChar>::value;
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    inline void initializeFrom(const_pointer const other, size_type const count) noexcept
    {
        try
        {

            if ((other == nullptr) || (count == detail::BaseStringImplValues::Zero<size_type>::value))
            {
                clear();
                return;
            }

            m_length = detail::BaseStringImplValues::Zero<size_type>::value;
            appendFrom(other, count);
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    /// Copies the other string into this
    // TODO(dwplc): RFD -- this class replaces C strings A27-0-4
    // coverity[autosar_cpp14_a27_0_4_violation]
    inline void initializeFrom(const_pointer const other) noexcept
    {
        initializeFrom(other, NPOS);
    }

    /// Throws an @c OutOfBoundsException if @a idx is out of range
    auto at_impl(StringView const& operation, size_type const idx) noexcept(false) -> reference
    {
        if (idx > size())
        {
            detail::throwBaseStringIndexOutOfBounds(operation, size(), idx);
        }

        return m_data.at(idx);
    }

    /// Throws an @c OutOfBoundsException if @a idx is out of range
    auto at_impl(StringView const& operation, size_type const idx) const noexcept(false) -> const_reference
    {
        if (idx > size())
        {
            detail::throwBaseStringIndexOutOfBounds(operation, size(), idx);
        }

        return m_data.at(idx);
    }

    /// Checks for special cases (nan, inf) of the provided float and appends the appropriate value
    bool appendHandleSpecialCases(float64_t const fnumber) noexcept
    {
        if (std::isnan(std::abs(fnumber)))
        {
            this->operator+=(detail::BaseStringImplValues::FLOAT_VAL_NAN);
            return true;
        }

        if (std::isinf(fnumber))
        {
            constexpr float64_t ZERO_F{0.0};
            if (fnumber < ZERO_F)
            {
                this->operator+=(detail::BaseStringImplValues::FLOAT_VAL_NEG_INF);
            }
            else
            {
                this->operator+=(detail::BaseStringImplValues::FLOAT_VAL_INF);
            }

            return true;
        }
        return false;
    }

private:
    /// Array that contains the data for the string
    /// Ideally this would be a dw::core::Array, but that currently creates circular dependencies
    /// via ExceptionWithStackTrace and other safety code.
    std::array<TChar, BufferSize> m_data;

    /// The length of the string.
    size_type m_length;
};

/// This variable could be defined inside the class, but doing so causes compilation errors, so move it out here.
// TODO(dwplc): FP -- Coverity thinks that 'BufferSize' is not a symbolic name
// coverity[autosar_cpp14_a5_1_1_violation]
template <size_t BufferSize, typename TChar>
typename BaseString<BufferSize, TChar>::size_type const BaseString<BufferSize, TChar>::MAX_LENGTH{BufferSize - detail::BaseStringImplValues::One<size_type>::value};

/// This variable could be defined inside the class, but doing so causes compilation errors, so move it out here.
template <size_t BufferSize, typename TChar>
typename BaseString<BufferSize, TChar>::size_type const BaseString<BufferSize, TChar>::CAPACITY{BufferSize};

/// Convert a BaseString to a BasicStringView
template <size_t BufferSize, typename TChar>
auto toStringView(const BaseString<BufferSize, TChar>& base) noexcept -> BasicStringView<TChar>
{
    return base;
}

/// Write @a value to the output stream @a os.
template <typename TOutputStream, size_t BufferSize, typename TChar>
meta::BasicUnformattedOutputStreamFundamentalType<TOutputStream, TChar>&
// TODO(dwplc): RFD -- allow references to be returned from << when left-hand side is ostream-like
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a13_2_2_violation]
operator<<(TOutputStream& os, BaseString<BufferSize, TChar> const& value)
{
    return os.write(value.c_str(), static_cast<typename TOutputStream::int_type>(value.size()));
}

/// Stream based string concatenation. @c value taken by value for trivial types
template <size_t BufferSize, typename TChar, typename ValueT, typename std::enable_if_t<std::is_trivial<ValueT>::value, bool> = true>
// TODO(dwplc): RFD -- allow references to be returned from << when left-hand side is ostream-like
// TODO(dwplc): FP -- this function can throw and should not be marked noexcept A15-4-4
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a13_2_2_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
auto operator<<(BaseString<BufferSize, TChar>& lhs, ValueT const value) noexcept(false) -> BaseString<BufferSize, TChar>&
{
    lhs += value;
    return lhs;
}

/// Stream based string concatenation. @c value taken by reference for non-trivial types
template <size_t BufferSize, typename TChar, typename ValueT, typename std::enable_if_t<!std::is_trivial<ValueT>::value, bool> = true>
// TODO(dwplc): RFD -- allow references to be returned from << when left-hand side is ostream-like
// TODO(dwplc): FP -- this function can throw and should not be marked noexcept A15-4-4
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a13_2_2_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
auto operator<<(BaseString<BufferSize, TChar>& lhs, const ValueT& value) noexcept(false) -> BaseString<BufferSize, TChar>&
{
    lhs += value;
    return lhs;
}

/// Define dw::string<N> as a string holding char8_t as a basic type
template <std::size_t N>
using FixedString = BaseString<N, char8_t>;

/// Check if @a lhs is equal to @a rhs.
template <size_t BufferSize, typename TChar>
bool operator==(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return lhs.compare(rhs) == detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): FP -- this function can throw and should not be marked noexcept A15-4-4
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator==(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs) noexcept(false)
{
    return lhs.compare(rhs) == detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): FP -- this function can throw and should not be marked noexcept A15-4-4
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator==(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs) noexcept(false)
{
    return rhs == lhs;
}

/// Check if @a lhs is not equal to @a rhs.
template <size_t BufferSize, typename TChar>
bool operator!=(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return !(lhs == rhs);
}

/// Check if @a lhs is not equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator!=(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs)
{
    return !(lhs == rhs);
}

/// Check if @a lhs is not equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator!=(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs)
{
    return !(lhs == rhs);
}

/// Check if @a lhs is less than @a rhs.
template <size_t BufferSize, typename TChar>
bool operator<(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return lhs.compare(rhs) < detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator<(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs)
{
    return lhs.compare(rhs) < detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator<(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs)
{
    return rhs.compare(lhs) >= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
bool operator<=(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return lhs.compare(rhs) <= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator<=(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs)
{
    return lhs.compare(rhs) <= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is less than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator<=(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs)
{
    return rhs.compare(lhs) > detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than @a rhs.
template <size_t BufferSize, typename TChar>
bool operator>(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return lhs.compare(rhs) > detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator>(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs)
{
    return lhs.compare(rhs) > detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator>(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs)
{
    return rhs.compare(lhs) <= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
bool operator>=(const BaseString<BufferSize, TChar>& lhs, const BaseString<BufferSize, TChar>& rhs) noexcept
{
    return lhs.compare(rhs) >= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator>=(const BaseString<BufferSize, TChar>& lhs, const TChar* rhs)
{
    return lhs.compare(rhs) >= detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Check if @a lhs is greater than or equal to @a rhs.
template <size_t BufferSize, typename TChar>
// TODO(dwplc): RFD -- use of non-deduced context in overload meets expected user behavior of conversion operators. A13-5-5
// TODO(dwplc): RFD -- this class replaces C strings A27-0-4
// coverity[autosar_cpp14_a13_5_5_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
bool operator>=(const TChar* lhs, const BaseString<BufferSize, TChar>& rhs)
{
    return rhs.compare(lhs) < detail::BaseStringImplValues::Zero<std::int32_t>::value;
}

/// Defined in the dw::core namespace because gcc compiler doesn't have it.
/// Makes sure there are no buffer overflows and that the destination string is zero terminated.
/// Is inline so it doesn't conflict with Windows' implementation.
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the strcpy_s API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the strcpy_s API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
inline void strcpy_s(char8_t* const dest, size_t destsz, char8_t const* const src)
{
    if (destsz == detail::BaseStringImplValues::Zero<size_t>::value)
    {
        return;
    }

    const BasicStringView<char8_t> srcView{src};

    // Nothing to copy if the src string is empty
    if (srcView.size() == detail::BaseStringImplValues::Zero<size_t>::value)
    {
        return;
    }

    destsz--; //Leave space for terminator
    BasicStringView<char8_t>::size_type const copied{srcView.copy(dest, destsz)};

    // TODO(dwplc): RFD for M5-0-15 -- Pointer arithmetic has to happen here
    // TODO(dwplc): RFD for CERT CTR50-CPP - copied will be less than destsz, and it's up to callers of this function
    //                to ensure there is enough room in the destination buffer
    // coverity[autosar_cpp14_m5_0_15_violation]
    // coverity[cert_ctr50_cpp_violation]
    dest[copied] = detail::BaseStringImplValues::NulChar<char8_t>::value;
}

/// A preferred alternative to std::stoi because it doesn't raise exceptions
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::stol API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the std::stol API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
dw::core::Optional<int64_t> stol(char8_t const* const s, int32_t const base = detail::BaseStringImplValues::Ten<int32_t>::value);

/// A preferred alternative to std::stoi because it doesn't raise exceptions
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::stoul API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the std::stoul API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
dw::core::Optional<uint64_t> stoul(char8_t const* const s, int32_t const base = detail::BaseStringImplValues::Ten<int32_t>::value);

/// A preferred alternative to std::stod because it doesn't raise exceptions
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::stod API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the std::stod API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
dw::core::Optional<float64_t> stod(char8_t const* const s);

/// Reads from istream until the newline character is found or FixedString is full
/// Endline character not part of returned line string
/// Alternative to std::getline using FixedString instead
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::getline API
// coverity[autosar_cpp14_a8_4_8_violation]
template <std::size_t N>
std::istream& getline(std::istream& stream, FixedString<N>& line)
{
    line.clear();
    char8_t c = '\n';
    while (stream.get(c) && c != '\n')
    {
        if (!line.full())
        {
            line += c;
        }
        else
        {
            break;
        }
    }

    // handle windows CRLF
    if (!line.empty() && (line[line.size() - detail::BaseStringImplValues::One<size_t>::value] == '\r'))
    {
        line.resize(line.size() - detail::BaseStringImplValues::One<size_t>::value);
    }

    return stream;
};

// -----------------------------------------------------------------------------
/// Reads from istream until the delimiter is found or FixedString is full
/// Delimiter character not part of returned line string
/// Alternative to std::getline using FixedString instead
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::getline API
// coverity[autosar_cpp14_a8_4_8_violation]
template <std::size_t N>
std::istream& getline(std::istream& stream, FixedString<N>& line, char8_t const delim)
{
    line.clear();
    char8_t c = delim;
    while (stream.get(c) && c != delim)
    {
        if (!line.full())
        {
            line += c;
        }
        else
        {
            break;
        }
    }

    // if we did read at least one character, we need to unset the failbit
    // set by stream.get(c). Leave other flags intact.
    if ((line.length() > detail::BaseStringImplValues::Zero<size_t>::value) && stream.fail())
    {
        stream.clear(stream.rdstate() & ~std::ios_base::failbit);
    }

    return stream;
}

} // namespace core
} // namespace dw

namespace std
{
/// Specialization for std hash
// TODO(dwplc): FP -- std::hash specialization
// coverity[autosar_cpp14_a11_0_2_violation]
template <size_t BufferSize, typename TChar>
struct hash<dw::core::BaseString<BufferSize, TChar>>
{
    /// Hashes a string using BaseString's Hash operator
    template <class T>
    size_t operator()(const T& x) const noexcept
    {
        return typename dw::core::BaseString<BufferSize, TChar>::Hash()(x);
    }
};

/// Specialization for std equal
/// so that String can be compared to char8_t *
/// TODO: remove this and replace with std::equal_to<> when C++14 is enabled in CUDA
// TODO(dwplc): FP -- std::equal_to specialization
// coverity[autosar_cpp14_a11_0_2_violation]
// coverity[autosar_cpp14_m17_0_2_violation]
template <size_t BufferSize, typename CharacterT>
struct equal_to<dw::core::BaseString<BufferSize, CharacterT>>
{
    /// Checks if two strings are equal
    template <class T1, class T2>
    constexpr bool operator()(const T1& lhs, const T2& rhs) const
    {
        return lhs == rhs;
    }
};
} // namespace std

#endif // DW_CORE_BASESTRING_HPP_
