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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_BASE_STRINGBUFFER_HPP_
#define DWSHARED_CORE_BASE_STRINGBUFFER_HPP_

#include <limits>
#include <cmath>
#include <type_traits>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <stdexcept>

#include "TypeAliases.hpp"

#define ASSERT_EXCEPTION_IF(exceptionType, condition, msg) dw::core::assertExceptionIf<exceptionType>(condition, msg " ((" #condition ")==false)");
#define ASSERT_ARGUMENT(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidArgumentException, condition, msg)
#define ASSERT_STATE(condition, msg) ASSERT_EXCEPTION_IF(dw::core::InvalidStateException, condition, msg)

namespace dw
{
namespace core
{
namespace detail
{
/// This structure holds the names of operations and other values used by the @c StringBuffer implementation. It
/// exists for compliance with AUTOSAR A5-1-1, which states that only symbolic names are acceptable.
struct StringBufferImplValues
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
};

template <typename T, typename = T>
struct has_c_str : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T>
struct has_c_str<T,
                 std::enable_if_t<
                     std::is_member_function_pointer<decltype(&T::c_str)>::value, T>> : std::true_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T, typename = T>
struct has_data_and_size : std::false_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

template <typename T>
struct has_data_and_size<T,
                         std::enable_if_t<
                             std::is_member_function_pointer<decltype(&T::data)>::value && std::is_member_function_pointer<decltype(&T::size)>::value, T>> : std::true_type // clang-tidy NOLINT(readability-identifier-naming)
{
};

} // end namespace detail

/// StringBuffer class that for low level stream like operations to serialize values into a given buffer as a string
/// StringBuffer doesn't own data, it requires the provided data container to support Tchar& at(size_t index); function.
/// This implementation is meant to be used in Exception as well as in BaseString class.
///
/// The buffer will silently truncate when attempting to assign or append a string that is longer than the @c BufferSize.
/// None of the provided functions throw, unless used StringContainer class throws.
template <size_t BufferSize, typename TChar = char8_t, class StringContainer = std::array<TChar, BufferSize>>
class StringBuffer
{
public:
    /// The value type of a single character.
    using value_type = TChar;

    /// A pointer to a character.
    using pointer = value_type*;

    /// A constant pointer to a character.
    using const_pointer = value_type const*;

    /// A reference to a character.
    using reference = value_type&;

    /// A constant reference to a character.
    using const_reference = value_type const&;

    /// Describes the @ref size of the string.
    using size_type = std::size_t;

    /// Represents not-a-position in search operations.
    static constexpr size_type NPOS{std::numeric_limits<size_type>::max()}; // clang-tidy NOLINT(readability-identifier-naming)

    /// The maximum length string that can be stored in this object.
    static size_type const MAX_LENGTH;

    /// The capacity of the underlying buffer of this object.
    static size_type const CAPACITY;

public:
    /// Constructs an empty buffer
    constexpr StringBuffer() noexcept // NOLINT(cppcoreguidelines-pro-type-member-init) - FP member vars are initialized in the delegated constructor
        : StringBuffer(detail::StringBufferImplValues::NulChar<TChar>::value, detail::StringBufferImplValues::Zero<size_type>::value)
    {
    }

    /// Constructs an empty buffer
    explicit StringBuffer(const_pointer const str) noexcept
        : m_data()
        , m_length(0)
    {
        (*this) += str;
    }

    /// Copy constructor
    StringBuffer(StringBuffer const&) noexcept = default;

    /// Move constructor
    StringBuffer(StringBuffer&&) noexcept = default;

    /// Replaces the contents of the buffer.
    StringBuffer& operator=(StringBuffer<BufferSize, TChar, StringContainer> const&) noexcept = default;

    /// Replaces the contents of the buffer.
    StringBuffer& operator=(StringBuffer<BufferSize, TChar, StringContainer>&&) noexcept = default;

    /// Destructor
    ~StringBuffer() noexcept = default;

    /// Returns a pointer to the first character in the string.
    auto data() noexcept -> pointer
    {
        return m_data.data();
    }

    /// Returns a pointer to the first character in the string.
    auto data() const noexcept -> const_pointer
    {
        return m_data.data();
    }

    auto get() const noexcept -> StringContainer const&
    {
        return m_data;
    }

    /// Returns a pointer to the first character in the string.
    auto c_str() noexcept -> pointer
    {
        return m_data.data();
    }

    /// Returns a pointer to the first character in the string.
    auto c_str() const noexcept -> const_pointer
    {
        return m_data.data();
    }

    /// Checks if the string has no characters.
    constexpr bool empty() const noexcept
    {
        return size() == detail::StringBufferImplValues::Zero<size_type>::value;
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
    constexpr void clear() noexcept
    {
        m_length                                                       = detail::StringBufferImplValues::Zero<size_type>::value;
        m_data[detail::StringBufferImplValues::Zero<size_type>::value] = detail::StringBufferImplValues::NulChar<TChar>::value;
    }

    /// Appends the given character @a ch to the string.
    ///
    /// @note
    /// Unlike std::basic_string::push_back, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently do nothing.
    void push_back(value_type const ch) noexcept
    {
        try
        {
            if (m_length < MAX_LENGTH)
            {
                m_data.at(m_length) = ch;
                m_length++;
                m_data.at(m_length) = detail::StringBufferImplValues::NulChar<TChar>::value;
            }
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    template <size_t OtherBufferSize>
    auto operator+=(StringBuffer<OtherBufferSize, TChar, std::array<TChar, OtherBufferSize>> const& src) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        return operator+=(src.c_str());
    }

    /// Append a container supporting c_str() function to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    template <class ContainerT, std::enable_if_t<detail::has_c_str<ContainerT>::value, bool> = true>
    auto operator+=(ContainerT const& src) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        return operator+=(src.c_str());
    }

    /// Append a container supporting c_str() function to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    template <class ContainerT, std::enable_if_t<detail::has_data_and_size<ContainerT>::value && !detail::has_c_str<ContainerT>::value, bool> = true>
    auto operator+=(ContainerT const& src) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        return appendStr(src.data(), src.size());
    }

    /// Append a string to this one.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    auto operator+=(const_pointer const str) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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
    /// @ref size() > @ref max_size(). Instead, the operation will silently do nothing.
    auto operator+=(value_type const ch) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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
    auto operator+=(int64_t inumber) noexcept(false) -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        try
        {
            uint64_t unumber{0UL};
            if (inumber < detail::StringBufferImplValues::Zero<int64_t>::value)
            {
                constexpr TChar DASH{'-'};
                this->operator+=(DASH);
                if (inumber == std::numeric_limits<int64_t>::min())
                {
                    inumber += detail::StringBufferImplValues::One<int64_t>::value;
                    inumber = -inumber;
                    unumber = static_cast<uint64_t>(inumber);
                    unumber += detail::StringBufferImplValues::One<uint64_t>::value;
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
    auto operator+=(uint64_t unumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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
            } while ((unumber > detail::StringBufferImplValues::Zero<uint64_t>::value) && (m_length < MAX_LENGTH));

            size_t lastIdx{m_length - detail::StringBufferImplValues::One<size_type>::value};

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
    auto operator+=(float64_t const fnumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        constexpr uint32_t DECIMAL_PLACES{4UL};
        return this->appendNum(fnumber, DECIMAL_PLACES);
    }

    /// Appends the given float @c fnumber to the string, to 4 decimal places.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    auto operator+=(float32_t const fnumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        this->operator+=(static_cast<float64_t>(fnumber));
        return *this;
    }

    /// Appends the given integer @c inumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    auto operator+=(int32_t const inumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        this->operator+=(static_cast<int64_t>(inumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    auto operator+=(uint16_t const unumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        this->operator+=(static_cast<uint64_t>(unumber));
        return *this;
    }

    /// Appends the given unsigned integer @c unumber to the string.
    /// @note
    /// Unlike std::basic_string::operator+=, this function will not throw if the operation would result in
    /// @ref size() > @ref max_size(). Instead, the operation will silently truncate the string to fit this buffer.
    auto operator+=(uint32_t const unumber) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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
    auto appendNum(float64_t fnumber, uint32_t const decimalPlaces, bool useScientific = false) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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

            // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
            // coverity[cert_flp34_c_violation]
            int64_t const exponent{static_cast<int64_t>(std::floor(flog10))};
            float64_t const fpow10{this->safePow(TEN_F, exponent)};

            floatVal /= fpow10; // floatVal is now in format a.xxxxxx where 1 <= a < 10
        }

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        uint64_t unumber{static_cast<uint64_t>(std::floor(floatVal))};
        float64_t const decimal10{this->safePow(TEN_F, decimalPlaces)};

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        uint64_t floatingPart{static_cast<uint64_t>(std::round((floatVal - std::floor(floatVal)) * decimal10))};

        // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
        // coverity[cert_flp34_c_violation]
        if (floatingPart == static_cast<uint64_t>(decimal10))
        {
            // The decimals rounded up to the next unit
            unumber += detail::StringBufferImplValues::One<uint64_t>::value;
            floatingPart -= static_cast<uint64_t>(decimal10);
        }

        this->operator+=(unumber);

        constexpr TChar DOT_CHAR{'.'};
        this->operator+=(DOT_CHAR);

        // Count how many zeroes should be added in the beginning. This handles cases like 0.0039.
        appendLeadingFloatingZeros(floatingPart, decimalPlaces);

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

            // TODO(dwplc): Safe Solution Needed: Need safe wrapper for converting floats to ints
            // coverity[cert_flp34_c_violation]
            int32_t const exponent{static_cast<int32_t>(std::floor(flog10))};

            if (exponent > detail::StringBufferImplValues::Zero<int32_t>::value)
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
    auto appendStr(const_pointer const src, size_type const srcSize) noexcept -> StringBuffer<BufferSize, TChar, StringContainer>&
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
    auto appendNum(float32_t const fnumber, uint32_t const decimalPlaces, const bool useScientific = false) noexcept(false) -> StringBuffer<BufferSize, TChar, StringContainer>&
    {
        return this->appendNum(static_cast<float64_t>(fnumber), decimalPlaces, useScientific);
    }

protected:
    /// Delegating constructor to be used by the public constructors
    explicit constexpr StringBuffer(TChar const c, size_type const length) noexcept
        : m_data{c}
        , m_length(length)
    {
    }

    /// Helper that appends a given number of elements from provided source
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

            size_type i{0UL};
            for (; i < copyCount; ++i)
            {
                if (other[i] == detail::StringBufferImplValues::NulChar<TChar>::value)
                {
                    break;
                }
                m_data.at(m_length + i) = other[i];
            }

            m_length += i;
            m_data.at(m_length) = detail::StringBufferImplValues::NulChar<TChar>::value;
        }
        catch (std::out_of_range const& e) // Logging is not possible as Logger depends on BaseString.
        {
            static_cast<void>(e);
        }
    }

    /// Checks for special cases (nan, inf) of the provided float and appends the appropriate value
    bool appendHandleSpecialCases(float64_t const fnumber) noexcept
    {
        if (std::isnan(std::abs(fnumber)))
        {
            this->operator+=("nan");
            return true;
        }

        if (std::isinf(fnumber))
        {
            constexpr float64_t ZERO_F{0.0};
            if (fnumber < ZERO_F)
            {
                this->operator+=("-inf");
            }
            else
            {
                this->operator+=("inf");
            }

            return true;
        }
        return false;
    }

    /// Count how many zeroes should be added in the beginning. This handles cases like 0.0039.
    void appendLeadingFloatingZeros(uint64_t floatingPart, uint32_t decimalPlaces)
    {
        uint64_t digitCount{0UL};
        while (floatingPart > detail::StringBufferImplValues::Zero<uint64_t>::value)
        {
            if (digitCount < std::numeric_limits<decltype(digitCount)>::max() - 1)
            {
                digitCount++;
            }

            constexpr uint64_t TEN_ULL{10ULL};
            floatingPart /= TEN_ULL;
        }

        // Add zeroes
        while (digitCount < decimalPlaces)
        {
            constexpr TChar ZERO_CHAR{'0'};
            digitCount++;
            this->operator+=(ZERO_CHAR);
        }
    }

    /// return x^y with some simplified assumptions about the exponent and the base (limited domain errors)
    constexpr float64_t safePow(float64_t base, int64_t exponent)
    {
        float64_t result{1.0};
        uint64_t e{static_cast<uint64_t>(std::fabs(exponent))};
        for (uint64_t i{0}; i < e; i++)
        {
            if (exponent >= 0)
            {
                result *= base;
            }
            else
            {
                result /= base;
            }
        }

        // Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
        // coverity[autosar_cpp14_m5_0_21_violation]
        static_cast<void>(std::feclearexcept(FE_ALL_EXCEPT));

        return result;
    }

protected:
    /// Array that contains the data for the string
    StringContainer m_data;

    /// The length of the string.
    size_type m_length;
};

/// This variable could be defined inside the class, but doing so causes compilation errors, so move it out here.
// TODO(dwplc): FP -- Coverity thinks that 'BufferSize' is not a symbolic name
// coverity[autosar_cpp14_a5_1_1_violation]
template <size_t BufferSize, typename TChar, class StringContainer>
typename StringBuffer<BufferSize, TChar, StringContainer>::size_type const StringBuffer<BufferSize, TChar, StringContainer>::MAX_LENGTH{BufferSize - detail::StringBufferImplValues::One<size_type>::value};

/// This variable could be defined inside the class, but doing so causes compilation errors, so move it out here.
template <size_t BufferSize, typename TChar, class StringContainer>
typename StringBuffer<BufferSize, TChar, StringContainer>::size_type const StringBuffer<BufferSize, TChar, StringContainer>::CAPACITY{BufferSize};

/// Stream based string concatenation. @c value taken by value for trivial types
template <size_t BufferSize, typename TChar, class StringContainer, typename ValueT, typename std::enable_if_t<std::is_trivial<ValueT>::value, bool> = true>
auto operator<<(StringBuffer<BufferSize, TChar, StringContainer>& lhs, ValueT const value) noexcept(false) -> StringBuffer<BufferSize, TChar, StringContainer>&
{
    lhs += value;
    return lhs;
}

/// Stream based string concatenation. @c value taken by reference for non-trivial types
template <size_t BufferSize, typename TChar, class StringContainer, typename ValueT, typename std::enable_if_t<!std::is_trivial<ValueT>::value, bool> = true>
auto operator<<(StringBuffer<BufferSize, TChar, StringContainer>& lhs, const ValueT& value) noexcept(false) -> StringBuffer<BufferSize, TChar, StringContainer>&
{
    lhs += value;
    return lhs;
}

} // namespace core
} // namespace dw

#endif // DWSHARED_CORE_BASE_STRINGBUFFER_HPP_
