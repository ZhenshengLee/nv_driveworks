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

#ifndef DWSHARED_CORE_UNIQUEDEVICE_PTR_HPP_
#define DWSHARED_CORE_UNIQUEDEVICE_PTR_HPP_

#include "DevicePtr.hpp"
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <memory>
#include <type_traits>
#include <cuda_runtime.h>

namespace dw
{

namespace core
{

/**
 * Unified deleter for unique pointers to Device(GPU) memory.
 *
 * @note Logs errors instead of throwing.
 * @tparam T Resource data type stored by unique pointers to Device memory
 */
template <typename T>
struct CudaDeviceMemDeleter
{
    using Ti = typename std::remove_extent<T>::type;

    void operator()(Ti* const p) const
    {
        cudaError_t const res{cudaFree(p)};
        if (res != cudaSuccess)
        {
            // Note: cannot throw on a deleter, log error instead
            LOGSTREAM_ERROR(nullptr) << "Error freeing cuda pointer";
        }
    }
};

/**
 * Unified deleter for unique pointers to Host memory.
 *
 * @note Logs errors instead of throwing.
 *
 * @tparam T Resource data type stored by unique pointers to Host memory
 */
template <typename T>
struct CudaHostMemDeleter
{
    using Ti = typename std::remove_extent<T>::type;

    void operator()(Ti* const p) const
    {
        cudaError_t const res{cudaFreeHost(p)};
        if (res != cudaSuccess)
        {
            // Note: cannot throw on a deleter, log error instead
            LOGSTREAM_ERROR(nullptr) << "Error freeing cuda pointer";
        }
    }
};

/**
 * Equivalent of std::unique_ptr but for device (gpu) memory.
 *
 * @tparam T Data type to wrap with unique pointer
 */
template <typename T>
class UniqueDevicePtr
{
public:
    /**
     * Element data type for @p T. E.g., when T is <tt>VectorFixed\<int\></tt>, Ti
     * will be <tt>int</tt>.
     */
    using Ti = typename std::remove_all_extents<T>::type;

    /// Create an empty UniqueDevicePtr instance. Nullptr is wrapped
    UniqueDevicePtr()
        : m_ptr(nullptr)
    {
    }

    /**
     * Create an UniqueDevicePtr from DevicePtr<T> instance
     * @param ptr DevicePtr<T> instance. A pointer to a device memory block
     */
    explicit UniqueDevicePtr(DevicePtr<T> const ptr)
        : m_ptr(ptr.get())
    {
    }

    UniqueDevicePtr(UniqueDevicePtr&& _Right) = default;
    ~UniqueDevicePtr()                        = default;

    UniqueDevicePtr& operator=(UniqueDevicePtr&& _Right) = default;

    /// Reset the pointer. Content of internal pointer will be destructed.
    void reset()
    {
        m_ptr.reset();
    }
    /**
    *  @brief Replace the pointer with a given DevicePtr<T> instance
    *         The internal raw pointer inside this unique pointer will be replaced by
    *         the wrapped pointer of given DevicePtr<T> instance
    *  @param value DevicePtr<T> instance
    */
    void reset(DevicePtr<T> value)
    {
        m_ptr.reset(value.get());
    }

    /**
    * Releases the ownership of the managed object if any. get() returns nullptr after the call.
    * The caller is responsible for deleting the object.
    *
    * @return Pointer to the managed object or nullptr if there was no managed object,
    *         i.e. the value which would be returned by get() before the call.
    */
    auto release() -> DevicePtr<T>
    {
        return DevicePtr<T>(m_ptr.release());
    }

    /**
    * Whether internal pointer is empty.
    * @return True if empty, otherwise false is returned.
    */
    bool empty() const
    {
        return !static_cast<bool>(m_ptr);
    }

    /**
    * Convert UniqueDevicePtr<T> into DevicePtr<T>
    * @return A DevicePtr<T> instance with internal pointer equal to one wrapped by calling
    *         UniqueDevicePtr instance.
    */
    auto getDevicePtr() const -> DevicePtr<T>
    {
        return DevicePtr<T>(m_ptr.get());
    }

    /**
    * Shortcut method so we don't have to call unique.getDevicePtr().get() to get the pointer.
    * This allows writing obj.get() to get the pointer when obj is DevicePtr or UniqueDevicePtr.
    *
    * @return Element type pointer
    */
    auto get() const -> Ti*
    {
        return m_ptr.get();
    }

    UniqueDevicePtr(const UniqueDevicePtr&) = delete;
    auto operator=(const UniqueDevicePtr&) -> UniqueDevicePtr& = delete;

private:
    std::unique_ptr<Ti, CudaDeviceMemDeleter<Ti>> m_ptr;
};

/**
 * Allocates GPU memory for a single object
 *
 * @tparam T Data type to wrap with unique pointer
 * @return UniqueDevicePtr<T> instance with a type @p T instance inside.
 */
template <typename T>
auto makeUniqueDevice() -> UniqueDevicePtr<T>
{
    static_assert(!std::is_array<T>::value,
                  "makeUnique(Context *) cannot be called with for an array type.");

    using Ti = typename DevicePtr<T>::Ti;

    cudaError_t res;
    Ti* value;
    res = cudaMalloc<Ti>(&value, sizeof(Ti));
    if (res != cudaSuccess)
    {
        throw CudaException(res, "Cannot allocate cuda memory");
    }

    return UniqueDevicePtr<T>(DevicePtr<T>(value));
}

/**
 * Allocates GPU memory for an array of objects
 *
 * @tparam T Data type to wrap with unique pointer
 *
 * @param itemCount Array size in elements
 * @return UniqueDevicePtr<T> instance with an array of elements having data type @p T.
 */
template <typename T>
auto makeUniqueDevice(size_t const itemCount) -> UniqueDevicePtr<T>
{
    static_assert(std::is_array<T>::value,
                  "makeUnique(Context *, size_t) cannot be called with for non-array type.");

    using Ti = typename DevicePtr<T>::Ti;

    cudaError_t res;
    Ti* value;
    res = cudaMalloc<Ti>(&value, itemCount * sizeof(Ti));
    if (res != cudaSuccess)
    {
        throw CudaException(res, "Cannot allocate cuda memory");
    }

    return UniqueDevicePtr<T>(DevicePtr<T>(value));
}

/**
 * Equivalent of std::unique_ptr for 2D device memory.
 *
 * @tparam T Array type of Pitch data
 *
 * @note This class only works with Array types
 */
template <typename T>
class UniqueDevicePitchPtr
{
public:
    /// Element data type of Pitch data type @p T
    using Ti = typename std::remove_all_extents<T>::type;

    static_assert(std::is_array<T>::value, "UniqueDevicePitchPtr can only be used with array types");

    /**
    * Construct from specified raw Pitch data pointers and
    * stride size of Pitch
    * @param ptr Raw data pointer of Pitch
    * @param strideItems Stride size of Pitch
    */
    UniqueDevicePitchPtr(Ti* const ptr, size_t const strideItems)
        : m_ptr(ptr)
        , m_strideItems(strideItems)
    {
    }

    /**
    * Construct an empty UniqueDevicePitchPtr - with empty
    * Pitch and stride size equals to 0.
    */
    UniqueDevicePitchPtr()
        : m_ptr(nullptr)
        , m_strideItems(0)
    {
    }

    /**
    * Convert a DevicePitchPtr<T> instance to unique pointer
    *
    * @param ptr DevicePitchPtr<T> instance
    *
    * @note Pitch instance inside given @p ptr will be transferred to
    *       current instance, including its Pitch data pointer and
    *       stride size.
    */
    explicit UniqueDevicePitchPtr(DevicePitchPtr<T> const ptr) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : UniqueDevicePitchPtr(ptr.get().getPtr(), ptr.get().getStrideItems())
    {
    }

    /**
    * Construct from an existing @b rvalue-referenced UniqueDevicePitchPtr<T> instance
    *
    * @param _Right Another @b rvalue-referenced UniqueDevicePitchPtr<T> instance
    * @note Ownership of Pitch from @p _Right will be transferred to current instance.
    */
    UniqueDevicePitchPtr(UniqueDevicePitchPtr&& _Right)
        : m_ptr(std::move(_Right.m_ptr))
        , m_strideItems(_Right.m_strideItems)
    {
    }

    ~UniqueDevicePitchPtr() = default;

    /**
    * Transfers ownership of Pitch from another @b rvalue-referenced UniqueDevicePitchPtr<T> instance.
    *
    * @param _Right Another @b rvalue-referenced UniqueDevicePitchPtr<T> instance
    * @note Ownership of Pitch from @p _Right will be transferred to current instance.
    */
    auto operator=(UniqueDevicePitchPtr&& _Right) -> UniqueDevicePitchPtr&
    { // assign by moving _Right
        if (this != &_Right)
        { // different, do the move
            m_ptr         = std::move(_Right.m_ptr);
            m_strideItems = _Right.m_strideItems;
        }
        return (*this);
    }

    /// Resets the unique pointer, releases the managed Pitch.
    void reset()
    {
        m_ptr.reset(nullptr);
    }

    /**
    * Replaces internal Pitch to given one
    * @param value A Pitch described in DevicePitchPtr<T>.
    * @note Stride size of Pitch remains unchanged after operation
    */
    void reset(DevicePitchPtr<T> value)
    {
        m_ptr         = std::unique_ptr<Ti, CudaDeviceMemDeleter<Ti>>(value.get().getPtr());
        m_strideItems = value.get().getStrideItems();
    }

    /**
    * Releases unique pointer
    * @return Managed Pitch pointer described in DevicePitchPtr<T>.
    * @note Current UniqueDevicePitchPtr<T> instance will become invalid.
    *       This basically means to transfer the ownership of managed Pitch
    *       to the returned DevicePitchPtr<T> instance.
    */
    auto release() -> DevicePitchPtr<T>
    {
        return DevicePitchPtr<T>(PitchPtr<T>({m_ptr.release(), m_strideItems}));
    }

    /**
     * Whether managed Pitch data pointer is empty.
     * @return True if empty and false otherwise.
     */
    bool empty() const
    {
        return !static_cast<bool>(m_ptr);
    }

    /**
    * Access managed Pitch.
    * @return A PitchPtr<T> instance with managed Pitch pointer from current instance
    *         and stride size unchanged.
    */
    auto get() const -> PitchPtr<T>
    {
        return PitchPtr<T>({m_ptr.get(), m_strideItems});
    }

    /**
    * Access managed Pitch.
    * @return A DevicePitchPtr<T> instance with Pitch pointer from current instance
    *         and stride size unchanged.
    */
    auto getDevicePtr() const -> DevicePitchPtr<T>
    {
        return DevicePitchPtr<T>(PitchPtr<T>({m_ptr.get(), m_strideItems}));
    }

    /// Returns stride size of managed Pitch
    size_t getStrideBytes() const
    {
        return DevicePitchPtr<T>(get()).getStrideBytes();
    }

    /**
    * Creates a new DevicePitchPtr<const T> instance from current instance
    * @return A DevicePitchPtr<const T> instance with current Pitch pointer,
    *         so this returned DevicePitchPtr instance manages a const Pitch.
    */
    auto toConst() const -> DevicePitchPtr<const T>
    {
        return DevicePitchPtr<const T>(get());
    }

    /**
    * Swap the managed Pitch with another UniqueDevicePitchPtr instance.
    * This will swap the managed Pitch data pointer and Pitch stride size
    * between current instance and the instance provided as @p other
    *
    * @param other UniqueDevicePitchPtr<T> instance to swap with current instance
    */
    void swap(UniqueDevicePitchPtr& other)
    {
        m_ptr.swap(other.m_ptr);
        std::swap(m_strideItems, other.m_strideItems);
    }

    UniqueDevicePitchPtr(const UniqueDevicePitchPtr&) = delete;
    auto operator=(const UniqueDevicePitchPtr&) -> UniqueDevicePitchPtr& = delete;

private:
    std::unique_ptr<Ti, CudaDeviceMemDeleter<Ti>> m_ptr;
    size_t m_strideItems;
};

/**
 * Allocate 2D GPU memory
 *
 * @tparam T Element data type of Pitch
 *
 * @param size Array size in 2 dimensions
 * @return A unique pointer to allocated Pitch
 */
template <typename T>
auto makeUniqueDevice(Array2ui const& size) -> UniqueDevicePitchPtr<T>
{
    static_assert(std::is_array<T>::value,
                  "makeUnique(Vector2ui) cannot be called with for non-array type.");

    using Ti = typename DevicePitchPtr<T>::Ti;

    cudaError_t res;
    Ti* value;
    size_t strideBytes;
    res = cudaMallocPitch<Ti>(&value, &strideBytes, static_cast<uint64_t>(sizeof(Ti)) * size[0], static_cast<uint64_t>(size[1]));
    if (res != cudaSuccess)
    {
        throw CudaException(res, "Cannot allocate cuda pitch memory, bytes ",
                            static_cast<uint64_t>(sizeof(Ti)) * size[0] * size[1]);
    }
    return UniqueDevicePitchPtr<T>(DevicePitchPtr<T>(value, strideBytes / sizeof(Ti)));
}

/**
 * Allocate 2D GPU memory
 *
 * @tparam T Element data type for Pitch
 *
 * @param width Width of 2D array
 * @param height Height of 2D array
 * @return A unique pointer to allocated Pitch
 */
template <typename T>
auto makeUniqueDevice(size_t const width, size_t const height) -> UniqueDevicePitchPtr<T>
{
    return makeUniqueDevice<T>(Array2ui{static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
}

//////////////////////////////////////////////
// General cuda host memory
// cuda pinned memory and cuda mapped memory are allocated by
// the same cudaHostAlloc API and freed by cudaFreeHost API,
// the only difference is the flag during allocation, pinned
// memory uses cudaHostAllocPortable flag, while mapped memory
// uses cudaHostAllocMapped flag. So here use UniqueCudaHostPtr
// for all kinds of page-locked memory allocated by cudaHostAlloc
// with different allcation flag.
// UniquePinnedPtr and UniqueMappedPtr are 2 aliases for this
// general-purpose cuda page-locked memory type

/**
 * CUDA host memory unique pointer. This could be CUDA pinned memory
 * or CUDA mapped memory.
 *
 * @note ::makeUniqueCudaHost(uint32_t flag) could be called with
 *       different flags to allocate CUDA pinned or mapped memory blocks.
 *       cudaHostAllocPortable for pinned memory blocks and
 *       cudaHostAllocMapped for mapped memory blocks.
 *
 * @tparam T Data type of object to manage with unique pointer
 */
template <typename T>
using UniqueCudaHostPtr = std::unique_ptr<T, CudaHostMemDeleter<T>>;

/**
 * Allocate Host memory for non-array data types
 *
 * @tparam T Data type to allocate memory for
 *
 * @param flag cudaHostAlloc() flags.
 * @return An unique pointer(#UniqueCudaHostPtr<T>) to allocated
 *         CUDA host memory block
 */
template <typename T>
UniqueCudaHostPtr<T> makeUniqueCudaHost(uint32_t flag)
{
    static_assert(!std::is_array<T>::value,
                  "makeUniqueCudaHost() cannot be called with for array type.");

    using Ti = typename std::remove_all_extents<T>::type;
    Ti* value{nullptr};
    cudaError_t res{cudaHostAlloc(&value, sizeof(Ti), flag)};

    if (res != cudaSuccess)
    {
        throw CudaException(res, "Cannot allocate cuda host memory with flag ", flag);
    }
    // TODO(dwplc): compliant by exception, it is allowed to use explicit new function call to create an instance of it requires a custom deleter.
    // coverity[autosar_cpp14_a20_8_5_violation]
    return UniqueCudaHostPtr<T>(value);
}

/**
 * Allocate Host memory for arrays
 *
 * @tparam T Data type to allocate memory for. Must be an array type
 *
 * @param flag cudaHostAlloc() flags.
 * @param itemCount Array size
 * @return An unique pointer(#UniqueCudaHostPtr<T>) to allocated
 *         CUDA host array
 */
template <typename T>
UniqueCudaHostPtr<T> makeUniqueCudaHost(size_t const itemCount, uint32_t flag)
{
    static_assert(std::is_array<T>::value,
                  "makeUniqueCudaHost(size_t) cannot be called with for non-array type.");

    using Ti = typename std::remove_all_extents<T>::type;
    cudaError_t res;
    Ti* value;
    res = cudaHostAlloc(&value, itemCount * sizeof(Ti), flag);

    if (res != cudaSuccess)
    {
        throw CudaException(res, "Cannot allocate cuda host memory with flag", flag);
    }
    // TODO(dwplc): FP - compliant by exception, it is allowed to use explicit new function call to create an instance of it requires a custom deleter.
    // coverity[autosar_cpp14_a20_8_5_violation]
    return UniqueCudaHostPtr<T>(value);
}

//////////////////////////////////////////////
/**
 * Pinned host memory, alias of UniqueCudaHostPtr for easy using
 *
 * @tparam T Data type of managed pinned memory blocks by unique pointer
 */
template <typename T>
using UniquePinnedHostPtr = UniqueCudaHostPtr<T>;

/**
 * Allocate CUDA pinned host memory blocks
 *
 * @tparam T Data type to allocate pinned memory for.
 *
 * @return An unique pointer(#UniquePinnedHostPtr<T>) to allocated pinned memory
 *         for data type @p T
 */
template <typename T>
UniquePinnedHostPtr<T> makeUniquePinnedHost()
{
    return makeUniqueCudaHost<T>(cudaHostAllocPortable);
}

/**
 * Allocate CUDA pinned host memory for arrays
 *
 * @tparam T Element type of array to allocate pinned memory for.
 *
 * @param itemCount Array size
 * @return An unique pointer(#UniquePinnedHostPtr<T>) to allocated pinned memory
 *         for array with element data type @p Ti
 */
template <typename T>
UniquePinnedHostPtr<T> makeUniquePinnedHost(size_t const itemCount)
{
    return makeUniqueCudaHost<T>(itemCount, cudaHostAllocPortable);
}

//////////////////////////////////////////////
/**
 * Mapped host memory
 *
 * @tparam T Data type of managed mapped memory blocks by unique pointer
 */
template <typename T>
class UniqueMappedPtr : public UniqueCudaHostPtr<T>
{
    using Base = UniqueCudaHostPtr<T>;

public:
    UniqueMappedPtr() = default;

    /**
     * @brief Construct a new UniqueMappedPtr object from UniqueCudaHostPtr
     */
    explicit UniqueMappedPtr(Base&& base)
        : Base(std::move(base))
    {
    }

    /**
     * @brief Get GPU address of host mapped memory
     */
    template <typename T1 = T, std::enable_if_t<!std::is_array<T1>::value>* = nullptr>
    DevicePtr<T> getDevicePtr()
    {
        return MakeDevicePtr(this->get());
    }

    /**
     * @brief Get GPU address of host mapped memory
     */
    template <typename T1 = T, std::enable_if_t<std::is_array<T1>::value>* = nullptr>
    DevicePtr<T> getDevicePtr()
    {
        return MakeDeviceArray(this->get());
    }

    /**
     * @brief const version to get GPU address of host mapped memory
     */
    template <typename T1 = T, std::enable_if_t<!std::is_array<T1>::value>* = nullptr>
    DevicePtr<T const> getDevicePtr() const
    {
        return MakeDevicePtr(this->get());
    }

    /**
     * @brief const version to get GPU address of host mapped memory
     */
    template <typename T1 = T, std::enable_if_t<std::is_array<T1>::value>* = nullptr>
    DevicePtr<T const> getDevicePtr() const
    {
        return MakeDeviceArray(this->get());
    }
};

/**
 * Allocate CUDA mapped host memory blocks
 *
 * @tparam T Data type to allocate mapped memory for.
 *
 * @return An unique pointer(#dw::core::UniqueMappedPtr) to allocated mapped memory
 *         for data type @p T
 */
template <typename T>
UniqueMappedPtr<T> makeUniqueMapped()
{
    return UniqueMappedPtr<T>(std::move(makeUniqueCudaHost<T>(cudaHostAllocMapped)));
}

/**
 * Allocate CUDA mapped host memory for arrays
 *
 * @tparam T Element type of array to allocate mapped memory for.
 *
 * @param itemCount Array size
 * @return An unique pointer(#dw::core::UniqueMappedPtr) to allocated mapped memory
 *         for array with element data type @p T
 */
template <typename T>
UniqueMappedPtr<T> makeUniqueMapped(size_t itemCount)
{
    return UniqueMappedPtr<T>(std::move(makeUniqueCudaHost<T>(itemCount, cudaHostAllocMapped)));
}
}
} // namespace dw

#endif
