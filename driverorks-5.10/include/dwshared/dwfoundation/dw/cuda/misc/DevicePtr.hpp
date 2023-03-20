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

#ifndef DW_CORE_DEVICE_PTR_HPP_
#define DW_CORE_DEVICE_PTR_HPP_

#include <dw/core/language/BasicTypes.hpp>
#include <dw/core/container/Array.hpp>

#include "CudaChannelDesc.hpp"

#include <cassert>
#include <memory>
#include <cstddef>
#include <type_traits>

namespace dw
{

namespace core
{
///////////////////////////////////////////////////
// Classes
///////////////////////////////////////////////////

/**
 * @brief Base class for DevicePtr. A wrapper class for any pointers used on device only.
 *
 * @note Users are encouraged to use DevicePtr class rather than using this base class.
 *
 * @tparam T Data type. This could be array or normal types.
 */
template <typename T>
class DevicePtrBase
{
public:
    /**
    * The type of the elements. For normal types sames as T.
    * For array types, the type of an individual element.
    */
    using Ti = typename std::remove_all_extents<T>::type;

    /// Create a DevicePtrBase instance with its internal pointer to be empty
    DevicePtrBase()
        : m_ptr(nullptr)
    {
    }

    /**
     * @brief Creates a DevicePtrBase instance with given pointer
     * @param value Internal pointer's value
     */
    explicit DevicePtrBase(Ti* const value)
        : m_ptr(value)
    {
    }

    /**
     * @brief Checks whether the internal pointer is NULL
     * @return true if internal pointer is not null, false otherwise
     */
    CUDA_BOTH_INLINE explicit operator bool() const // clang-tidy NOLINT
    {
        return (m_ptr != nullptr);
    }

    /**
    * @brief  Checks whether 2 DevicePtrBase instances are equal in pointer value level
    *
	* @tparam C Pointer type of 2 instances
    *
	* @param lhs Pointer to compare
    * @param rhs Pointer to compare
    * @return true if @p lhs equals to @p rhs in pointer values. false otherwise
    *
	* @note std::nullptr_t is accepted for both @p lhs and @p rhs
    */
    template <typename C>
    CUDA_BOTH_INLINE friend bool operator==(DevicePtrBase<C> const& lhs, DevicePtrBase<C> const& rhs) noexcept;

    /**
     * @brief  Checks whether 2 DevicePtrBase instances are not equal in pointer value level
     *
     * @tparam C Pointer type of 2 instances
     *
     * @param lhs Pointer to compare
     * @param rhs Pointer to compare
     * @return true if @p lhs does not equal to @p rhs in pointer values. false otherwise
     *
     * @note std::nullptr_t is accepted for both @p lhs and @p rhs
     */
    template <typename C>
    CUDA_BOTH_INLINE friend bool operator!=(DevicePtrBase<C> const& lhs, DevicePtrBase<C> const& rhs) noexcept;

    /**
    * @brief Set internal pointer's value
    * @param value Pointer value to set
    */
    CUDA_BOTH_INLINE void set(Ti* value)
    {
        m_ptr = value;
    }

    /**
     * @brief Retrieve the reference to internal wrapped pointer
     * @return A reference to internal pointer
     */
    Ti** getRef()
    {
        return &m_ptr;
    }

    /**
     * @brief Access internal pointer
     * @return Internal pointer value(a pointer type of Ti \*)
     */
    CUDA_BOTH_INLINE auto get() const -> Ti*
    {
        return m_ptr;
    }

#ifdef __CUDACC__
    /**
     * @brief  De-reference the internal pointer.
     * @return De-referenced internal pointer i.e. type of Ti&
     *
     * @note Normally GPU kernels receive direct pointers obtained through DevicePtr::get()
     *     but sometimes it is easier to have a struct on CPU with DevicePtrs and simply pass the entire struct.
     *     This method allows CUDA to treat the DevicePtr as a normal pointer, but it is device-only so memory cannot
     *     be dereferenced in CPU. For example <tt>DevicePtr<uint32_t> count</tt>, This DevicePtr instance
     *     can be passed directly to the kernel and the elements accessed as if they were naked pointers.
     */
    CUDA_INLINE auto operator*() const -> Ti&
    {
        return *m_ptr;
    }

    /**
     * @brief Access array member via index in case internal pointer
     *        is an array pointer.
     * @param idx Array index
     * @return Reference to array member with subscript equal to @p idx.
     */
    CUDA_INLINE auto operator[](size_t idx) const -> Ti&
    {
        return m_ptr[idx];
    }
#endif

private:
    Ti* m_ptr;
};

template <typename C>
CUDA_BOTH_INLINE bool operator==(DevicePtrBase<C> const& lhs, DevicePtrBase<C> const& rhs) noexcept
{
    return (lhs.m_ptr == rhs.m_ptr);
}

template <typename C>
CUDA_BOTH_INLINE bool operator!=(DevicePtrBase<C> const& lhs, DevicePtrBase<C> const& rhs) noexcept
{
    return !(lhs == rhs);
}

// TODO(dwplc): RFD -- comparison with null meets expected user behavior of comparison operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename C>
CUDA_BOTH_INLINE bool operator==(DevicePtrBase<C> const& lhs, std::nullptr_t) noexcept
{
    return (lhs == DevicePtrBase<C>(nullptr));
}

// TODO(dwplc): RFD -- comparison with null meets expected user behavior of comparison operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename C>
CUDA_BOTH_INLINE bool operator!=(DevicePtrBase<C> const& lhs, std::nullptr_t) noexcept
{
    return !(lhs == DevicePtrBase<C>(nullptr));
}

// TODO(dwplc): RFD -- comparison with null meets expected user behavior of comparison operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename C>
CUDA_BOTH_INLINE bool operator==(std::nullptr_t, DevicePtrBase<C> const& rhs) noexcept
{
    return (DevicePtrBase<C>(nullptr) == rhs);
}

// TODO(dwplc): RFD -- comparison with null meets expected user behavior of comparison operators.
// coverity[autosar_cpp14_a13_5_5_violation]
template <typename C>
CUDA_BOTH_INLINE bool operator!=(std::nullptr_t, DevicePtrBase<C> const& rhs) noexcept
{
    return !(DevicePtrBase<C>(nullptr) == rhs);
}

/**
 * @brief DevicePtr - non-const version of device memory pointer class.
 *        Represents a pointer in device (gpu) memory space.
 *        This should be the type of all variables in cpu space that point to gpu space.
 *
 * @tparam T Data type. Could be array or normal types.
 */
template <typename T, typename = void>
class DevicePtr : public DevicePtrBase<T>
{
public:
    using Base = DevicePtrBase<T>;
    using Ti   = typename Base::Ti;
    using Base::Base;

    /// Default constructor. Creates a DevicePtr instance with empty internal pointer
    DevicePtr()
        : DevicePtrBase<T>()
    {
        //Guarantee that size is the same as a pointer so we can reinterpret
        static_assert(sizeof(*this) == sizeof(T*),
                      "Size of DevicePtr must be the same size as a normal pointer");
    }

    /**
     * @brief Return a 'const' version DevicePtr instance: its internal pointer is a
     *        const pointer type.
     *
     * @return A new DevicePtr<const T> instance with internal pointer in const pointer
     *         type pointing to internal pointer of calling instance.
     */
    auto toConst() const -> DevicePtr<const T>
    {
        return DevicePtr<const T>(Base::get());
    }
};

/**
 * DevicePtr - const version of device memory pointer class.
 * Represents a pointer in device (gpu) memory space.
 * This should be the type of all variables in cpu space that point
 * to gpu space.
 *
 * @tparam T Data type. Could be array or normal types.
 *
 * @note This class is only valid if data type @b T is const type
 */
template <typename T>
class DevicePtr<T, typename std::enable_if<std::is_const<T>::value>::type> : public DevicePtrBase<T>
{
public:
    using Base = DevicePtrBase<T>;
    using Ti   = typename Base::Ti;
    using Base::Base;

    /// Default constructor. Creates a DevicePtr instance with empty internal pointer
    DevicePtr()
        : DevicePtrBase<T>()
    {
        //Guarantee that size is the same as a pointer so we can reinterpret
        static_assert(sizeof(*this) == sizeof(T*),
                      "Size of DevicePtr must be the same size as a normal pointer");
    }

    /**
     * @brief Construct from a non-const version DevicePtr instance
     * @param other A DevicePtr instance whose internal pointer is not a const pointer
     */
    DevicePtr(const DevicePtr<typename std::remove_const<T>::type>& other) // clang-tidy NOLINT
        : DevicePtrBase<T>(other.get())
    {
    }
};

/**
 * @brief Helper function to create a DevicePtr from a pointer.
 *        Shortens <tt>DevicePtr<T> a = DevicePtr<T>(b);</tt>
 *        To <tt>auto a = MakeDevicePtr(b);</tt> So that T is automatically inferred.
 *
 * @tparam T data type to wrap with DevicePtr<T>
 *
 * @param p pointer of type @p T
 * @return A DevicePtr<T> instance with internal pointer value equals to @p p
 */
template <typename T>
auto MakeDevicePtr(T* p) -> DevicePtr<T>
{
    return DevicePtr<T>(p);
}

/**
 * @brief Helper function to create a DevicePtr from an array.
 *        Shortens <tt>DevicePtr<T[]> a = DevicePtr<T[]>(b);</tt>
 *        To <tt>auto a = MakeDeviceArray(b);</tt> So that T is automatically inferred.
 *
 * @tparam T data type to wrap with DevicePtr<T>
 *
 * @param p array of type @p T
 * @return A DevicePtr<T> instance with internal pointer value equals to array @p p
 *
 */
template <typename T>
auto MakeDeviceArray(T* p) -> DevicePtr<T[]>
{
    return DevicePtr<T[]>(p);
}

/**
 * Helper function to create a DevicePtr from a pointer.
 * Shortens <tt>DevicePtr<T> a(reinterpret_cast<T*>(b));</tt>
 * To <tt>auto a = ReinterpretDevicePtr<T>(b);</tt>
 * So that @p T needs to be specified only once.
 *
 * @note This function only works for non array pointers. Otherwise
 *       there will be compilation errors
 *
 * @tparam Tin Input data type of input pointer @p p
 *
 * @tparam Tout Output data type
 * @param p Input pointer in type @p Tin
 *
 * @return DevicePtr<Tout> instance whose internal pointer equals to @p p
 *
 */
template <typename Tout, typename Tin>
auto ReinterpretDevicePtr(Tin* p) ->
    typename std::enable_if<
        !std::is_array<Tout>::value,
        DevicePtr<Tout>>::type
{
    return DevicePtr<Tout>(safeReinterpretCast<Tout*>(p));
}

/**
 * Helper function to create a DevicePtr from a pointer.
 * Shortens <tt>DevicePtr<T> a(reinterpret_cast<T*>(b));</tt>
 * To <tt>auto a = ReinterpretDevicePtr<T>(b);</tt>
 * So that T needs to be specified only once.
 *
 * @note This function only works for arrays. Otherwise
 *       there will be compilation errors
 *
 * @tparam Tin Input data type of input array @p p
 * @tparam Tout Output data type
 *
 * @return DevicePtr<Tout> instance whose internal pointer equals to @p p
 *
 */
// TODO(pshu): Cannot add doxygen directive @param because doxygen cannot
//       distinguish 2 overloaded ReinterpretDevicePtr functions though
//       both functions differ from return types(indicated by ->)
template <typename Tout, typename Tin>
auto ReinterpretDevicePtr(Tin* p) ->
    typename std::enable_if<
        std::is_array<Tout>::value,
        DevicePtr<Tout>>::type
{
    using Titem = typename std::remove_extent<Tout>::type;
    return DevicePtr<Tout>(safeReinterpretCast<Titem*>(p));
}

///////////////////////
/// SizeVoid
/// A replacement to sizeof(T) that returns 1 for void
/// Used by PitchPtr to support void and still calculate pitch sizes
template <class T>
struct SizeVoid
{
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t value = sizeof(T); // clang-tidy NOLINT
};
/// Specialized template class SizeVoid with T=void
template <>
struct SizeVoid<void>
{
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t value = 1; // clang-tidy NOLINT
};
/// Specialized template class SizeVoid with T=const void
template <>
struct SizeVoid<const void>
{
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t value = 1; // clang-tidy NOLINT
};

///////////////////////
// PitchPtr
// Forward declarations

// Non-const
template <typename T, typename = void>
struct PitchPtr;

// Const
template <typename T>
struct PitchPtr<T, typename std::enable_if<std::is_const<typename std::remove_all_extents<T>::type>::value>::type>;

/**
 * Base class of BasePitchPtr. It wraps a Pitch pointer and its stride size.
 *
 * @note Please use PitchPtr or DevicePitchPtr. This is a base class.
 *
 * @tparam T Pitch data type. This should be of array(2D) types.
 */
template <typename T>
struct BaseBasePitchPtr
{
    /// Data type of elements in Pitch.
    using Ti = typename std::remove_all_extents<T>::type;

    /// The size of an element. Use this instead of sizeof(Ti) because Ti can be void
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t SIZE_TI = SizeVoid<Ti>::value;

    /// Default constructor
    CUDA_BOTH_INLINE
    BaseBasePitchPtr()
        : m_ptr(nullptr)
        , m_strideItems(0)
    {
    }

    /**
     * Constructor with data pointer and stride
     *
     * @param ptrIn Data pointer of type @p T. This should be an Pitch allocated from CUDA.
     * @param strideItemsIn Stride(actual row width) size of Pitch.
     *
     * @note @b Stride here has different meaning than the term 'Stride' in CUDA which is (blockDim * gridDim).
     */
    CUDA_BOTH_INLINE
    BaseBasePitchPtr(Ti* const ptrIn, size_t const strideItemsIn)
        : m_ptr(ptrIn)
        , m_strideItems(strideItemsIn)
    {
    }

    /**
     * Space occupied by a single row in Pitch
     * @return Bytes occupied by a single row in Pitch
     */
    CUDA_BOTH_INLINE
    size_t getStrideBytes() const
    {
        return m_strideItems * SIZE_TI;
    }

    /**
     * Set stride size for Pitch
     * @note Total size of Pitch won't change
     * @param other New stride size
     */
    CUDA_BOTH_INLINE
    void setStrideItems(size_t other)
    {
        m_strideItems = other;
    }

    /// Return stride size of Pitch
    CUDA_BOTH_INLINE
    size_t getStrideItems() const
    {
        return m_strideItems;
    }

    /**
     * Return the address of @p row -th Row in Pitch
     * @param row Nth row
     * @return Address of first element in Nth row in Pitch
     */
    CUDA_BOTH_INLINE
    auto getRow(uint32_t row) const -> Ti* { return m_ptr + row * m_strideItems; }

    /**
     * Replace internal Pitch data with given one @p other
     * @param other New Pitch data
     */
    CUDA_BOTH_INLINE
    void setPtr(Ti* other)
    {
        m_ptr = other;
    }

    /**
     * Access internal pitch
     * @return Pitch data pointer in type ::dw::core::BaseBasePitchPtr::Ti
     */
    CUDA_BOTH_INLINE
    Ti* getPtr() const
    {
        return m_ptr;
    }

    /**
     * Return the address of @p row -th Row in Pitch.
     *
     * @note This operator could be called upon the result
     *       returned by itself - thus it enables the syntax: ptr[y][x] = value;
     *
     * @param row Nth row
     * @return Address of first element in Nth row in Pitch
     */
    CUDA_BOTH_INLINE
    auto operator[](uint32_t row) const -> Ti* { return getRow(row); }

    /**
     * @brief Fast reinterpret cast to a ::PitchPtr<T> with same stride. New Pitch has
     *        the same element @b size with current pitch elements. For example,
     *        <tt>PitchPtr\<int32_t\> p; PitchPtr\<uint32_t\> p2 = p.reinterpret();</tt>
     *
     * @note This method is used when caller doesn't change the element data size of internal Pitch.
     *       As a result, stride size won't change in converted Pitch too.
     *
     * @tparam TNew New data type to convert Pitch data into
     *
     * @return A ::PitchPtr<T> instance containing original data in internal Pitch,
     *         with the stride size unchanged. However data in returned PitchPtr instance are
     *         interpreted as @p TNew type.
     */
    template <typename TNew>
    CUDA_BOTH_INLINE auto reinterpret() const -> typename std::enable_if<sizeof(TNew) == SIZE_TI, PitchPtr<TNew[]>>::type
    {
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return PitchPtr<TNew[]>(reinterpret_cast<TNew*>(m_ptr), m_strideItems);
    }

    /**
     * @brief Full reinterpret cast to a ::PitchPtr<T> with different stride sizes. For example
     *        <tt>PitchPtr\<int32_t\> p; PitchPtr\<char\> p2 = p.reinterpret();</tt>
     *
     * @note This method is called when caller wants to change the element data size of internal Pitch.
     *
     * @tparam TNew New data type to convert Pitch data into
     *
     * @return A ::PitchPtr<T> instance with original Pitch data but with stride
     *         size automatically adjusted. Moreover, data in returned PitchPtr instance are
     *         interpreted as @p TNew type.
     */
    template <typename TNew>
    CUDA_BOTH_INLINE auto reinterpret() const -> typename std::enable_if<sizeof(TNew) != SIZE_TI, PitchPtr<TNew[]>>::type
    {
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return PitchPtr<TNew[]>(reinterpret_cast<TNew*>(m_ptr), getStrideBytes() / sizeof(TNew));
    }

    /**
     * @brief Special reinterpret to void case. Returned Pitch has void \* type elements(rather than void type
     *        elements - that's not allowed)
     *
     * @note This method is used when caller want a Pitch with elements in void type.
     *
     * @tparam TNew New data type to convert Pitch data into. It must be @p void. In return,
     *              returned Pitch has void \* type elements.
     *
     * @return A ::PitchPtr<T> instance containing original data in internal Pitch,
     *         with the stride size unchanged. Pitch elements are converted to void \* types
     *         in returned PitchPtr instance.
     */
    template <typename TNew>
    CUDA_BOTH_INLINE auto reinterpret() const -> typename std::enable_if<std::is_same<typename std::remove_const<TNew>::type, void>::value, PitchPtr<TNew>>::type
    {
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return PitchPtr<TNew>(reinterpret_cast<TNew*>(m_ptr), getStrideBytes());
    }

    /**
    *  Allocate a cudaResourceDesc instance for given array size.
    *  @param size Array size(with 2 dimensions)
    *  @return A cudaResourceDesc descriptor instance for cudaResourceTypePitch2D
    *          type and with internal Pitch data pointer exposed.
    */
    inline cudaResourceDesc getCudaResourceDesc(Array2ui size) const
    {
        // cuda structs contain unions, have to disable static checker
        cudaResourceDesc resDesc{};
        resDesc.resType                  = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.desc         = cudaCreateChannelDesc<typename std::remove_const<Ti>::type>(); // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.devPtr       = m_ptr;                                                         // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.pitchInBytes = m_strideItems * SIZE_TI;                                       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.width        = size[0];                                                       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.height       = size[1];                                                       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        return resDesc;
    }

private:
    Ti* m_ptr;
    size_t m_strideItems;
};

/**
 * Extends ::dw::core::BaseBasePitchPtr. This class is usually used as the base of ::PitchPtr<T>
 *
 * @tparam T Pitch data type. This should be of array(2D) types.
 */
template <typename T>
struct BasePitchPtr : public BaseBasePitchPtr<T>
{
    using Base = BaseBasePitchPtr<T>;

    /// Data type of elements in Pitch. If @p T is an array then it's the type
    /// of array elements. Otherwise it's the same as @p T
    using Ti = typename Base::Ti;

    /// Same constructor as BaseBasePitchPtr()
    CUDA_BOTH_INLINE
    BasePitchPtr()
        : Base()
    {
    }

    /**
     * Constructor with data pointer and stride
     *
     * @param ptrIn Data pointer of type @p T. This should be an Pitch allocated from CUDA.
     * @param strideItemsIn Stride(actual row width) size of Pitch.
     *
     * @note @b Stride here has different meaning than the term 'Stride' in CUDA which is (blockDim * gridDim).
     */
    CUDA_BOTH_INLINE
    BasePitchPtr(Ti* const ptrIn, size_t const strideItemsIn)
        : Base(ptrIn, strideItemsIn)
    {
    }

    /**
     * Access Pitch element Pitch[row][col].
     *
     * @note This enables the syntax: ptr.at(x,y) = value;
     *
     * @param row Row number
     * @param col Column number
     *
     * @return Reference to Pitch element in @p row -th Row and @p col -th column.
     */
    CUDA_BOTH_INLINE
    auto at(uint32_t col, uint32_t row) const -> Ti& { return Base::getRow(row)[col]; }
};

/// void specialization to avoid sizeof() and reference issues. Not used for common cases.
/// @note ::dw::core::BaseBasePitchPtr::SIZE_TI which is the Pitch element size for this class is 1.
template <>
struct BasePitchPtr<void> : public BaseBasePitchPtr<void>
{
    using Base = BaseBasePitchPtr<void>;
    using Ti   = typename Base::Ti;

    // Same constructors
    using BaseBasePitchPtr<void>::BaseBasePitchPtr;
};

/// const void specialization to avoid sizeof() and reference issues. Not used for common cases.
/// @note ::dw::core::BaseBasePitchPtr::SIZE_TI which is the Pitch element size for this class is 1.
template <>
struct BasePitchPtr<const void> : public BaseBasePitchPtr<const void>
{
    using Base = BaseBasePitchPtr<const void>;
    using Ti   = typename Base::Ti;

    // Same constructors
    using BaseBasePitchPtr<const void>::BaseBasePitchPtr;
};

/**
 * Wrapper for a Pitch pointer and its stride size.
 *
 * @tparam T Pitch data type. This should be of array(2D) types.
 *
 * @note This PitchPtr class points to non-const type elements. There is another
 *       PitchPtr class that points to const type elements. It has identical
 *       class signature, but the compiler will make correct decisions to instantiate
 *       the correct PitchPtr instance based on whether its primitive data type is const
 *       for provided data type @b T.
 */
template <typename T, typename>
struct PitchPtr : public BasePitchPtr<T>
{
    using Base = BasePitchPtr<T>;
    using Ti   = typename Base::Ti;

    PitchPtr() = default;

    /**
    *  Constructor with data pointer and stride
    *
    *  @param ptrIn Data pointer of type @p T. This should be an Pitch allocated from CUDA.
    *               This pointer itself is constant.
    *  @param strideItemsIn Stride(actual row width) size of Pitch.
    */
    CUDA_BOTH_INLINE
    PitchPtr(Ti* const ptrIn, size_t const strideItemsIn)
        : BasePitchPtr<T>(ptrIn, strideItemsIn)
    {
    }

    /// Create an empty PitchPtr<T> instance. Its internal Pitch is empty.
    CUDA_BOTH_INLINE
    PitchPtr(std::nullptr_t) // clang-tidy NOLINT(google-explicit-constructor)
        : BasePitchPtr<T>(nullptr, 0)
    {
    }

    /**
    *  Creates a PitchPtr instance with const Pitch elements.
    *  @return A new PitchPtr\<const T\> instance. Its Pitch elements are const.
    */
    CUDA_BOTH_INLINE
    auto toConst() const -> PitchPtr<const T>
    {
        return PitchPtr<const T>(Base::getPtr(), Base::getStrideItems());
    }

    /**
    *  Same as toConst(). A convenient converter to convert non-const
    *  PitchPtr<T> instances into PitchPtr<const T> implicitly.
    */
    CUDA_BOTH_INLINE
    explicit operator PitchPtr<const T>() const noexcept // clang-tidy NOLINT
    {
        return toConst();
    }

    /**
    *  Create a PitchPtr\<T\> instance from raw pointers to Pitch data.
    *  @param p Data pointer to a Pitch.
    *  @param strideInBytes Stride size of this Pitch.
    *
    *  @return A PitchPtr\<T\> instance with given Pitch data pointer @p p and
    *          adjusted stride size.
    *  @note Returned PitchPtr\<T\> instance will adjust its stride size per
    *        is SIZE_TI automatically.
    */
    static auto fromBytePointer(void* p, size_t strideInBytes) -> PitchPtr<T>
    {
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return PitchPtr<T>(reinterpret_cast<Ti*>(p), strideInBytes / Base::SIZE_TI);
    }
};

/**
 * Wrapper for a Pitch pointer to const contents and its stride size.
 *
 * @tparam T Pitch data type. This should be of array(2D) types. Must be const.
 */
template <typename T>
struct PitchPtr<T, typename std::enable_if<std::is_const<typename std::remove_all_extents<T>::type>::value>::type>
    : public BasePitchPtr<T>
{
    using Base = BasePitchPtr<T>;
    using Ti   = typename Base::Ti;

    PitchPtr() = default;

    /**
    *  Constructor with data pointer and stride
    *  @param ptr_ Data pointer of type @p T. This should be an Pitch allocated from CUDA.
    *  @param strideItems_ Stride(actual row width) size of Pitch.
    */
    CUDA_BOTH_INLINE
    PitchPtr(Ti* ptr_, size_t strideItems_)
        : BasePitchPtr<T>(ptr_, strideItems_)
    {
    }

    /// Create an empty PitchPtr\<const T\> instance. Its internal Pitch is empty.
    CUDA_BOTH_INLINE
    PitchPtr(std::nullptr_t) // clang-tidy NOLINT
        : BasePitchPtr<T>(nullptr, 0)
    {
    }

    /**
    *  Creates a PitchPtr instance with const Pitch elements.
    *  @return A reference to calling instance itself, as it's a PitchPtr to const Pitch itself.
    */
    CUDA_BOTH_INLINE
    auto toConst() const -> const PitchPtr<const T>&
    {
        return *this;
    }

    /**
    *  Create a PitchPtr\<T\> instance from raw pointers to Pitch data.
    *  @param p Data pointer to a Pitch. It's a pointer to a const Pitch(its elements are const).
    *  @param strideInBytes Stride size of this Pitch.
    *  @return A PitchPtr\<T\> instance with given Pitch data pointer @p p and
    *          given stride size @p strideInBytes.
    *  @note Returned PitchPtr\<T\> instance will adjust its stride size per
    *        is SIZE_TI automatically.
    *  @note Returned PitchPtr instance will point to constant data elements.
    *        Element values cannot be modified via this returned PitchPtr instance.
    */
    static auto fromBytePointer(const void* p, size_t strideInBytes) -> PitchPtr<T>
    {
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return PitchPtr<T>(reinterpret_cast<const Ti*>(p), strideInBytes / Base::SIZE_TI);
    }
};

// Forward def

template <typename T, typename = void>
class DevicePitchPtr;

template <typename T>
class DevicePitchPtr<T, typename std::enable_if<std::is_const<T>::value>::type>;

/**
 * DevicePitchPtrBase. Base class of DevicePitchPtr. This is a wrapper for ::PitchPtr<T>
 *
 * @tparam T Pitch data type. This should be of array(2D) types, or void.
 */
template <typename T>
class DevicePitchPtrBase
{
public:
    /// Element data type for Pitch
    using Ti = typename std::remove_all_extents<T>::type;

    /// Create a DevicePitchPtr instance with empty Pitch.
    DevicePitchPtrBase()
        : m_data({nullptr, 0})
    {
        static_assert(std::is_array<T>::value || std::is_same<T, void>::value, "DevicePitchPtr can only be used with array types");
    }

    /**
    *  Create a DevicePitchPtrBase instance with given Pitch data pointer and stride size.
    *
    *  @param ptr Pitch data pointer. Should be pointing to an array. This should be
    *             Allocated by CUDA API cudaMallocPitch(). Note this pointer itself is const.
    *  @param strideItems Stride size of Pitch.
    */
    explicit DevicePitchPtrBase(Ti* const ptr, size_t const strideItems)
        : m_data({ptr, strideItems})
    {
        static_assert(std::is_array<T>::value || std::is_same<T, void>::value, "DevicePitchPtr can only be used with array types");
    }

    /**
    *  @brief Creates a DevicePitchPtrBase instance from a PitchPtr\<T\> instance.
    *         Pitch contents and stride size won't change
    *
    *  @param data A const PitchPtr\<T\> instance. Its pitch data is mutable/modifiable.
    */
    explicit DevicePitchPtrBase(PitchPtr<T> const data)
        : m_data(data)
    {
        static_assert(std::is_array<T>::value || std::is_same<T, void>::value, "DevicePitchPtr can only be used with array types");
    }

    /// Checks whether internal Pitch is empty/null
    explicit operator bool() const // clang-tidy NOLINT
    {
        return m_data.ptr != nullptr;
    }

    /**
    *  Replaces internal Pitch with given Pitch data pointer @p ptr and stride size @p strideItems
    *  @param ptr New Pitch data pointer
    *  @param strideItems Stride size for Pitch @p ptr.
    */
    void set(Ti* ptr, size_t strideItems)
    {
        m_data.ptr         = ptr;
        m_data.strideItems = strideItems;
    }

    /**
    *  Access internal Pitch
    *  @return Internal Pitch represented in ::PitchPtr<T>
    */
    auto get() const -> const PitchPtr<T>&
    {
        return m_data;
    }

    /// Return stride size of internal Pitch
    size_t getStrideBytes() const
    {
        return m_data.getStrideBytes();
    }

    /**
    *  Return a cudaResourceDesc descriptor on internal Pitch for a specified @p size.
    *  @param size Array dimension for external access of internal Pitch data.
    *  @return A cudaResourceDesc descriptor with type cudaResourceTypePitch2D and with
    *          its size equals to given array @p size.
    */
    inline cudaResourceDesc getCudaResourceDesc(Array2ui size) const
    {
        cudaResourceDesc resDesc{};
        resDesc.resType                  = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.desc         = cudaCreateChannelDesc<typename std::remove_const<Ti>::type>();     // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.devPtr       = const_cast<void*>(reinterpret_cast<const void*>(m_data.getPtr())); // clang-tidy NOLINT
        resDesc.res.pitch2D.pitchInBytes = m_data.getStrideItems() * BasePitchPtr<T>::SIZE_TI;                // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.width        = size[0];                                                           // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        resDesc.res.pitch2D.height       = size[1];                                                           // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
        return resDesc;
    }

    /**
    *  @brief Reinterpret cast to a DevicePitchPtr<T>.
    *
    *  @tparam TNew New data type to convert Pitch data into. Its data size might be
    *               different than that in Pitch from calling instance.
    *
    *  @return A DevicePitchPtr\<T\> instance containing original data in internal Pitch,
    *          with the stride size automatically adjusted.
    */
    template <class TNew>
    auto reinterpret() const -> DevicePitchPtr<TNew[]>
    {
        return DevicePitchPtr<TNew[]>(get().template reinterpret<TNew>());
    }

private:
    PitchPtr<T> m_data;
};

/**
 * @brief DevicePitchPtr - non-const version.
 *        Represents a pointer to 2D memory (with stride) in device (gpu) memory space.
 *        This should be the type of all variables in cpu space that point
 *        to gpu 2D memory.
 *
 * @tparam T Pitch data type. This could be an array or other data types.
 */
template <typename T, typename>
class DevicePitchPtr : public DevicePitchPtrBase<T>
{
public:
    using Base = DevicePitchPtrBase<T>;
    /// Element data type in Pitch. Might be different than data type of @p T.
    using Ti = typename DevicePitchPtrBase<T>::Ti;

    /// Construct an empty DevicePitchPtr
    DevicePitchPtr()
        : DevicePitchPtrBase<T>()
    {
    }

    /// Construct an empty DevicePitchPtr
    DevicePitchPtr(std::nullptr_t) // clang-tidy NOLINT
    {
    }

    /**
    *  Create a DevicePitchPtr instance with given Pitch data pointer and stride size.
    *
    *  @param ptr Pitch data pointer. Should be pointing to an array. This should be
    *             Allocated by CUDA API cudaMallocPitch(). A const pointer.
    *  @param strideItems Stride size of Pitch.
    */
    explicit DevicePitchPtr(Ti* const ptr, size_t const strideItems)
        : DevicePitchPtrBase<T>(ptr, strideItems)
    {
    }

    /**
    *  @brief Creates a DevicePitchPtr instance from a PitchPtr\<T\> instance.
    *         Pitch contents and stride size won't change.
    *
    *  @param data A PitchPtr\<T\> instance.
    */
    explicit DevicePitchPtr(PitchPtr<T> const data)
        : DevicePitchPtrBase<T>(data)
    {
    }

    /**
    *  Create a DevicePitchPtr\<T\> instance from raw pointers to Pitch data.
    *
    *  @param p Data pointer to a Pitch. It's a pointer to a Pitch.
    *  @param strideInBytes Stride size of this Pitch.
    *
    *  @return A DataPitchPtr\<T\> instance with given Pitch data pointer @p p and
    *          given stride size @p strideInBytes.
    *
    *  @note Returned DataPitchPtr\<T\> instance will adjust its stride size per
    *        is SIZE_TI automatically.
    */
    static auto fromBytePointer(void* p, size_t strideInBytes) -> DevicePitchPtr<T>
    {
        return DevicePitchPtr<T>(PitchPtr<T>::fromBytePointer(p, strideInBytes));
    }

    /// Creates a new DevicePitchPtr\<const T\> instance, where its internal Pitch is const.
    auto toConst() const -> DevicePitchPtr<const T>
    {
        return DevicePitchPtr<const Ti[]>(this->get().getPtr(), this->get().getStrideItems());
    }

    /**
    * @brief Same as toConst().
    *        A convenient converter to convert non-const <tt>DevicePitchPtr<T></tt>
	*        instances into <tt>DevicePitchPtr<const T></tt>. E.g., there is a function asking for
	*        <tt>DevicePitchPtr<const T></tt> as input argument, we can directly call this API with
	*        non-const <tt>DevicePitchPtr<T></tt> instances, and this operator will convert the input
	*        argument from type <tt>DevicePitchPtr<T></tt> into <tt>DevicePitchPtr<const T></tt> automatically.
    */
    explicit operator DevicePitchPtr<const T>() const // clang-tidy NOLINT
    {
        return toConst();
    }
};

/**
 * @brief DevicePitchPtr - const version
 *        Represents a pointer to 2D memory (with stride) in device (gpu) memory space.
 *        This should be the type of all variables in cpu space that point
 *        to gpu 2D memory.
 *
 * @tparam T Pitch data type. Must be a const type
 */
template <typename T>
class DevicePitchPtr<T, typename std::enable_if<std::is_const<T>::value>::type>
    : public DevicePitchPtrBase<T>
{
public:
    using Ti = typename DevicePitchPtrBase<T>::Ti;

    /// Construct an empty DevicePitchPtr
    DevicePitchPtr()
        : DevicePitchPtrBase<T>()
    {
    }

    /// Construct an empty DevicePitchPtr
    DevicePitchPtr(std::nullptr_t) // clang-tidy NOLINT
    {
    }

    /**
    *  Create a DevicePitchPtr instance with given Pitch data pointer and stride size.
    *
    *  @param ptr Pitch data pointer. Should be pointing to an array. This should be
    *             Allocated by CUDA API cudaMallocPitch(). Must be a pointer to const array
    *  @param strideItems Stride size of Pitch.
    */
    explicit DevicePitchPtr(Ti* ptr, size_t strideItems)
        : DevicePitchPtrBase<T>(ptr, strideItems)
    {
    }

    /**
    *  @brief Creates a DevicePitchPtr instance from a PitchPtr\<T\> instance.
    *         Pitch contents and stride size won't change
    *
    *  @param data A PitchPtr\<T\> instance. This PitchPtr must be wrapping a const Pitch
    */
    explicit DevicePitchPtr(PitchPtr<T> data)
        : DevicePitchPtrBase<T>(data)
    {
    }

    /**
    *  Creates a DevicePitchPtr instance from a PitchPtr instance.
    *  @param data A PitchPtr instance. Its internal Pitch has same element data type of @p T
    *              but its internal Pitch elements are not const.
    */
    explicit DevicePitchPtr(PitchPtr<typename std::remove_const<T>::type> data)
        : DevicePitchPtrBase<T>(data.getPtr(), data.getStrideItems())
    {
    }

    /**
    *  Creates a DevicePitchPtr instance from another DevicePitchPtr instance.
    *  @param data A DevicePitchPtr instance. Its internal Pitch has same element data type of @p T
    *              but its internal Pitch elements are not const.
    */
    explicit DevicePitchPtr(DevicePitchPtr<typename std::remove_const<T>::type> data)
        : DevicePitchPtr<T>(data.get())
    {
    }

    /**
    *  Create a DevicePitchPtr\<T\> instance from raw pointers to Pitch data.
    *
    *  @param p Data pointer to a Pitch. It's a pointer to a const Pitch(its elements are const).
    *  @param strideInBytes Stride size of this Pitch.
    *
    *  @return A DataPitchPtr\<T\> instance with given Pitch data pointer @p p and
    *          given stride size @p strideInBytes.
    *
    *  @note Returned DataPitchPtr\<T\> instance will adjust its stride size per
    *        is SIZE_TI automatically.
    */
    static auto fromBytePointer(const void* p, size_t strideInBytes) -> DevicePitchPtr<T>
    {
        return DevicePitchPtr<T>(PitchPtr<T>::fromBytePointer(p, strideInBytes));
    }

    /**
    * Creates a DevicePitchPtr instance with const Pitch elements.
    * @return A reference to calling instance itself, as it's a DevicePitchPtr to const Pitch itself.
    */
    auto toConst() const -> const DevicePitchPtr<T>&
    {
        return *this;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Implementation of DevicePitchPtr
/////////////////////////////////////////////////////////////////////////////////////////

template <>
inline cudaResourceDesc BaseBasePitchPtr<__half[]>::getCudaResourceDesc(Array2ui size) const
{
    cudaResourceDesc resDesc{};
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc         = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat); // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.devPtr       = m_ptr;                                                          // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.pitchInBytes = m_strideItems * sizeof(__half);                                 // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.width        = size[0];                                                        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.height       = size[1];                                                        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    return resDesc;
}

template <>
inline cudaResourceDesc DevicePitchPtrBase<__half[]>::getCudaResourceDesc(Array2ui size) const
{
    cudaResourceDesc resDesc{};
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc         = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat); // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.devPtr       = m_data.getPtr();                                                // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.pitchInBytes = m_data.getStrideItems() * sizeof(__half);                       // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.width        = size[0];                                                        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    resDesc.res.pitch2D.height       = size[1];                                                        // clang-tidy NOLINT(cppcoreguidelines-pro-type-union-access)
    return resDesc;
}

} // namespace features
} // namespace dw

#endif // DW_CORE_DEVICE_PTR_HPP_
