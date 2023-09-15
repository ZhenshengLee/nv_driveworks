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

#ifndef DWSHARED_CORE_STATIC_SPAN_HPP_
#define DWSHARED_CORE_STATIC_SPAN_HPP_

#include "Array.hpp"

#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

namespace dw
{

namespace core
{

/**
 *  \class StaticSpan
 *
 *  \brief Generates static sized views on static containers
 *
 *  \tparam T underlying data type of the underlying container element
 *  \tparam NUMBER_OF_ELEMENTS is the number of elements the StaticSpan view is handling
 *
 *  \note NUMBER_OF_ELEMENTS is required to be > 0
 *
 */

template <class T, size_t NUMBER_OF_ELEMENTS>
class StaticSpan
{
public:
    static_assert(NUMBER_OF_ELEMENTS > 0, "Static size Span is not allowed to have NUMBER_OF_ELEMENTS == 0");

    /**
     * \brief C-tor to generate a static view over data with first element at ptr
     *
     * \param[in] ptr is a T* pointer to the First element that shall be handled by the StaticSpan
     *
     * \note ptr is required to be != nullptr. Otherwise it will throw
     *
     */
    CUDA_BOTH_INLINE
    explicit StaticSpan(T* const ptr) noexcept(false)
        : m_ptr(ptr)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-1733
        assertExceptionIf<InvalidArgumentException>(ptr != nullptr, "StaticSpan C-tor pointer is null");
    };

    // explicitly delete
    StaticSpan(std::nullptr_t ptr) = delete;

    // explicitly delete
    StaticSpan() = delete;

    /**
     * \brief operator[] for element wise access to the StaticSpan underlying data
     *
     * \param[in] index index of the element that should be accessed. The first elment in the span has index = 0. The last element has index = NUMBER_OF_ELEMENTS - 1
     *
     * \return reference to the element at position index
     *
     * \note method will throw in case of out of bound access, i.e. index >= NUMBER_OF_ELEMENTS
     */
    CUDA_BOTH_INLINE
    auto operator[](size_t const index) const noexcept(false) -> T&
    {
        return at(index);
    }

    /**
     * \brief get<NUM_ELEMENT>() for element wise access to the StaticSpan underlying data, with static boundary check
     *
     * \tparam NUM_ELEMENT index of the element that should be accessed. The first elment in the span has NUM_ELEMENT = 0. The last element has NUM_ELEMENT =  NUMBER_OF_ELEMENTS - 1
     *
     * \return reference to the element at position NUM_ELEMENT
     *
     * \note method has a compile time check if the requested element is part of the StaticSpan
     */
    template <size_t NUM_ELEMENT>
    CUDA_BOTH_INLINE auto get() const noexcept -> T&
    {
        static_assert(isContained(NUM_ELEMENT), "get<NUM_ELEMENT>() called with NUM_ELEMENT out of bounds");
        // coverity[autosar_cpp14_a9_3_1_violation] RFD Accepted: TID-0830
        return m_ptr[NUM_ELEMENT];
    }

    /**
     *  \brief size() static function returning NUMBER_OF_ELEMENTS
     *
     *  \return NUMBER_OF_ELEMENTS
     */
    CUDA_BOTH_INLINE
    static constexpr size_t size() noexcept
    {
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        return NUMBER_OF_ELEMENTS;
    }

    /**
     * \brief makeStaticSubSpan<SUBSPAN_SIZE, START_IDX> generates a new StaticSpan on a sub series of the managed objects from instance from this class
     *
     * \tparam SUBSPAN_SIZE the size of the new subspan
     * \tparam START_IDX the index of the first element (in current StaticSpan) Which will serve as the first element in the new generated SubSpan
     *
     * \return StaticSpan span un the sub set series of elements
     *
     * \note this method has a static check whether the requested "sub" StaticSpan is part of the original StaticSpan
     */
    template <size_t SUBSPAN_SIZE, size_t START_IDX>
    CUDA_BOTH_INLINE auto
    makeStaticSubSpan() const noexcept(false) -> StaticSpan<T, SUBSPAN_SIZE>
    {
        static_assert(START_IDX + SUBSPAN_SIZE <= NUMBER_OF_ELEMENTS, "makeStaticSubSpan sizes do not match to parent StaticSpan");
        static_assert(SUBSPAN_SIZE > 0, "makeStaticSubSpan SUBSPAN_SIZE should be > 0");
        static_assert(isContained(START_IDX), "makeStaticSubSpan START_IDX should be part of the parent StaticSpan");
        return StaticSpan<T, SUBSPAN_SIZE>(&m_ptr[START_IDX]);
    }

    /**
     * \brief toConst() generates a new StaticSpan view over const elements
     *
     * \return StaticSpan<T const, NUMBER_OF_ELEMENTS>
     */
    CUDA_BOTH_INLINE
    auto toConst() const noexcept(false) -> StaticSpan<T const, NUMBER_OF_ELEMENTS>
    {
        return StaticSpan<T const, NUMBER_OF_ELEMENTS>(m_ptr);
    }

    /**
     * \brief begin() Get begin iterator
     *
     * \return iterator to first element
     */
    CUDA_BOTH_INLINE auto begin() const noexcept -> T* { return m_ptr; }

    /**
     * \brief end() Get end iterator
     *
     * \return iterator to end()
     */
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_a9_3_1_violation] RFD Accepted: TID-0830
    // coverity[autosar_cpp14_m9_3_1_violation] RFD Accepted: TID-0830
    // coverity[autosar_cpp14_m5_0_15_violation] RFD Accepted: TID-1702
    CUDA_BOTH_INLINE auto end() const noexcept -> T* { return m_ptr + NUMBER_OF_ELEMENTS; } ///< Get end iterator

private:
    /**
     * \brief at() private helper method to have autosar compliant acces to underlying data
     *
     * \param[in] index index of the element that should be accessed. The first elment in the span has index = 0. The last element has index = NUMBER_OF_ELEMENTS - 1
     *
     * \return reference to the element at position index
     *
     * \note method will throw in case of out of bound access, i.e. index >= NUMBER_OF_ELEMENTS, if DW_RUNTIME_CHECKS enabled
     */
    CUDA_BOTH_INLINE
    auto at(size_t const index) const noexcept(false) -> T&
    {
        if (!isContained(index))
        {
#if DW_RUNTIME_CHECKS()
#if !defined(__CUDACC__)
            throwSpanIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>(
                "StaticSpan::at(): Tried to get index outside of StaticSpan");
#endif
#endif
        }
        // coverity[autosar_cpp14_a9_3_1_violation] RFD Accepted: TID-0830
        // coverity[autosar_cpp14_m9_3_1_violation] RFD Accepted: TID-0830
        // coverity[autosar_cpp14_m5_0_15_violation] RFD Pending: TID-1957
        return m_ptr[index];
    }

    /**
     * \brief isContained() static helper method to check whether an element with certain index is part of StaticSpan or not
     *
     * \param[in] index index of the element that should be checked.
     *
     * \return boolean indicating if index is part of span(true) or not(false)
     */
    CUDA_BOTH_INLINE
    static constexpr bool isContained(size_t const index) noexcept
    {
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        return index < NUMBER_OF_ELEMENTS;
    }

private:
    // Pointer to first managed element
    T* m_ptr;
};

/**
 * \brief makeStaticSpan(ContainerType&) helper function that generates a Static Span over static sized containers
 *
 * \tparam ContainerType underlying type of the static sized container, e.g. dw::core::Array or dw::core::Vector
 *
 * \param[in] container actual instance of the container where a StaticSpan view should be generated
 *
 * \return StaticSpan over container
 *
 * \note currently dw::core::Vector and all static sized continuos containers are supported that have static constexpr method static_size() and data() method.
 */
template <typename ContainerType,
          class T     = typename std::remove_pointer<decltype(std::declval<ContainerType>().data())>::type,
          size_t SIZE = ContainerType::Base::ELEMENT_COUNT,
          typename std::enable_if<ContainerType::Base::IS_VECTOR, bool>::type = true>
CUDA_BOTH_INLINE auto
    // coverity[autosar_cpp14_a8_4_8_violation] RFD Accepted: TID-1464
    makeStaticSpan(ContainerType& container) -> StaticSpan<T, SIZE>
{
    return StaticSpan<T, SIZE>(container.data());
}

/**
 * SFINAE for StaticSpan from generic container
 */
template <typename ContainerType,
          class T     = typename std::remove_pointer<decltype(std::declval<ContainerType>().data())>::type,
          size_t SIZE = ContainerType::size()>
CUDA_BOTH_INLINE auto
    // coverity[autosar_cpp14_a8_4_8_violation] RFD Accepted: TID-1464
    makeStaticSpan(ContainerType& container) -> StaticSpan<T, SIZE>
{
    return StaticSpan<T, SIZE>(container.data());
}

/**
 * makeStaticSpan from c-array
 */
template <class T, size_t SIZE>
CUDA_BOTH_INLINE auto makeStaticSpan(T (&arr)[SIZE]) -> StaticSpan<T, SIZE>
{
    static_assert(SIZE > 0, "array must be larger than 0");
    return StaticSpan<T, SIZE>(&arr[0]);
}
}
}
#endif
