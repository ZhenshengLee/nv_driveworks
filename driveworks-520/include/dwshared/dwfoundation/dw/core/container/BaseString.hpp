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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dwshared/dwfoundation/dw/core/base/TypeAliases.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx17.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <dwshared/dwfoundation/dw/core/base/StringBuffer.hpp>

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
class BaseString : public StringBuffer<BufferSize, TChar>
{
public:
    using Base = StringBuffer<BufferSize, TChar>;

    using Base::NPOS;
    using Base::clear;
    using Base::empty;
    using Base::data;
    using Base::length;
    using Base::c_str;
    using Base::capacity;
    using Base::size;

    using value_type      = typename Base::value_type;
    using pointer         = typename Base::pointer;
    using const_pointer   = typename Base::const_pointer;
    using reference       = typename Base::reference;
    using const_reference = typename Base::const_reference;

    /// An iterator which moves forwards.
    using const_iterator = BasicContiguousIterator<value_type const, BaseString<BufferSize, TChar> const>;

    /// An iterator which moves forwards.
    using iterator = BasicContiguousIterator<value_type, BaseString<BufferSize, TChar>>;

    /// An iterator which moves backwards.
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// An iterator which moves backwards.
    using reverse_iterator = std::reverse_iterator<iterator>;

    /// Describes the `size()` of the string.
    using size_type = std::size_t;

public:
    /// Constructs an empty buffer
    constexpr BaseString() noexcept
        : Base()
    {
    }

    /// Create an instance from the contents of @a str.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a18_1_1_violation]
    template <size_t N>
    inline BaseString(const TChar (&str)[N]) noexcept // clang-tidy NOLINT(google-explicit-constructor, cppcoreguidelines-pro-type-member-init) - we allow implicit converion for strings of known size at compilation time
        : BaseString()
    {
        static_assert(Base::CAPACITY >= N, "Cannot implicitly put a string larger than capacity into FixedString");
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
    /// Throws an @c OutOfBoundsException if @a pos is larger than `size()`.
    auto at(size_type const pos) noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_AT, pos);
    }

    /// Returns a const reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than `size()`.
    auto at(size_type const pos) const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_AT, pos);
    }

    /// Returns a reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than `size()`.
    auto operator[](size_type const pos) noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_OPERATOR_SUBSCRIPT, pos);
    }

    /// Returns a reference to the character at @c pos.
    ///
    /// Throws an @c OutOfBoundsException if @a pos is larger than `size()`.
    auto operator[](size_type const pos) const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_OPERATOR_SUBSCRIPT, pos);
    }

    /// Returns a reference to the first character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is `empty()`
    auto front() noexcept(false) -> reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_FRONT, detail::BaseStringImplValues::Zero<size_type>::value);
    }

    /// Returns a reference to the first character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is `empty()`
    auto front() const noexcept(false) -> const_reference
    {
        return at_impl(detail::BaseStringImplValues::OP_NAME_FRONT, detail::BaseStringImplValues::Zero<size_type>::value);
    }

    /// Returns reference to the last character in the string.
    ///
    /// Throws an @c OutOfBoundsException if the string is `empty()`
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
    /// Throws an @c OutOfBoundsException if the string is `empty()`
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

    /// Append a string to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
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
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(BasicStringView<TChar> const& src) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::appendStr(src.data(), src.size());
        return *this;
    }

    /// Append a string to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    // TODO(dwplc): RFD -- this class replaces C strings
    // coverity[autosar_cpp14_a27_0_4_violation]
    auto operator+=(const_pointer const str) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(str);
        return *this;
    }

    /// Appends the given character @a ch to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently do nothing. This is to preserve the
    /// legacy behavior of this class. In the future it may be desirable to throw @c BufferFullException in this case.
    auto operator+=(value_type const ch) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(ch);
        return *this;
    }

    /// Appends the given integer @c inumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(int64_t inumber) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(inumber);
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint64_t unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(unumber);
        return *this;
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(float64_t const fnumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        constexpr uint32_t DECIMAL_PLACES{4UL};
        Base::appendNum(fnumber, DECIMAL_PLACES);
        return *this;
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(float32_t const fnumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(static_cast<float64_t>(fnumber));
        return *this;
    }

    /// Appends the given integer @c inumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(int32_t const inumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(static_cast<int64_t>(inumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint16_t const unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(static_cast<uint64_t>(unumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// `size()` > `max_size()`. Instead, the operation will silently truncate the string to fit this buffer.
    /// This is to preserve the legacy behavior of this class. In the future it may be desirable to throw
    ///  @c BufferFullException in this case.
    auto operator+=(uint32_t const unumber) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::operator+=(static_cast<uint64_t>(unumber));
        return *this;
    }

    /// @see StringBuffer::append()
    auto append(const_pointer const src, size_type const srcSize) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::appendStr(src, srcSize);
        return *this;
    }

    /// @see StringBuffer::append()
    auto append(float32_t const fnumber, uint32_t const decimalPlaces, const bool useScientific = false) noexcept(false) -> BaseString<BufferSize, TChar>&
    {
        Base::appendNum(fnumber, decimalPlaces, useScientific);
        return *this;
    }

    /// @see StringBuffer::append()
    auto append(float64_t fnumber, uint32_t const decimalPlaces, const bool useScientific = false) noexcept -> BaseString<BufferSize, TChar>&
    {
        Base::appendNum(fnumber, decimalPlaces, useScientific);
        return *this;
    }

    /// Returns a substring [pos, pos+count). If the requested substring extends past the end of the string,
    /// or if count == NPOS, the returned substring is [pos, size())
    /// Template parameter @c SubStringSize will be incremented by one to account for the null character
    /// Throws an @c OutOfBoundsException if @a pos is larger than `size()`.
    template <size_t SubStringSize = BufferSize - 1>
    auto substr(size_type const pos = 0, size_type const count = NPOS) const -> BaseString<SubStringSize + 1, TChar>
    {
        if (pos > size())
        {
            detail::throwBaseStringIndexOutOfBounds(detail::BaseStringImplValues::OP_NAME_SUBSTR,
                                                    size(),
                                                    pos);
        }

        return BaseString<SubStringSize + 1, TChar>(c_str(), count, pos);
    }

    /// Truncates the string to the new size.
    /// If @a count is greater than `length()`, this function does nothing.
    void truncate(size_type const count) noexcept(false)
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
    /// Unlike std::string, if @a count is greater than `length()`, the characters between `length()`
    /// and @a count will not be set to null characters. They will be left unchanged to preserve the legacy behavior
    /// of this class. Only the existing null terminator will be changed to a space character.
    void resize(size_type const count) noexcept(false)
    {
        size_type const newLength{std::min(count, Base::MAX_LENGTH)};

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
            return typename BasicStringView<TChar>::Hash()(toStringView(str));
        }

        /// Hashes the given string
        // TODO(dwplc): RFD -- this class replaces C strings
        // coverity[autosar_cpp14_a27_0_4_violation]
        size_t operator()(const_pointer const str) const noexcept
        {
            return typename BasicStringView<TChar>::Hash()(BasicStringView<TChar>(str));
        }
    };

private:
    using Base::m_data;
    using Base::m_length;

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
            this->appendFrom(other, count);
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
};

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

// extend LoggerStream to support FixedString
template <size_t BufferSize>
inline Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, FixedString<BufferSize> const& v)
{
    return stream << v.c_str();
}

// extend LoggerStream to support FixedString
template <size_t BufferSize>
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, FixedString<BufferSize> const& v)
{
    return stream << v.c_str();
}

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

/// Defined in the core namespace because gcc compiler doesn't have it.
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
core::Optional<int64_t> stol(char8_t const* const s, int32_t const base = detail::BaseStringImplValues::Ten<int32_t>::value);

/// A preferred alternative to std::stoi because it doesn't raise exceptions
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::stoul API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the std::stoul API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
core::Optional<uint64_t> stoul(char8_t const* const s, int32_t const base = detail::BaseStringImplValues::Ten<int32_t>::value);

/// A preferred alternative to std::stod because it doesn't raise exceptions
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::stod API
// TODO(dwplc): RFD for A27-0-4 -- C-style strings part of the std::stod API
// coverity[autosar_cpp14_a8_4_8_violation]
// coverity[autosar_cpp14_a27_0_4_violation]
core::Optional<float64_t> stod(char8_t const* const s);

/// Reads from istream until the newline character is found or FixedString is full
/// Endline character not part of returned line string
/// Alternative to std::getline using FixedString instead
// TODO(dwplc): RFD for A8-4-8 -- output parameter is part of the std::getline API
// coverity[autosar_cpp14_a8_4_8_violation]
template <std::size_t N>
std::istream& getline(std::istream& stream, FixedString<N>& line)
{
    line.clear();
    char8_t c{'\n'};
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
    char8_t c{delim};
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
template <size_t BufferSize, typename CharacterT>
// coverity[autosar_cpp14_m17_0_2_violation]
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
