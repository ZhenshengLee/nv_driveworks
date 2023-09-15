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

#ifndef DW_CORE_BASEVECTORFIXED_HPP_
#define DW_CORE_BASEVECTORFIXED_HPP_

#include "../../UniqueSpan.hpp"

#include <utility>

namespace dw
{
namespace core
{

/// Iterator for VectorFixed
template <class T, bool IsConst>
class Iterator
{
public:
    /// const or non-const ptr to entries
    using TConditionalConst = typename std::conditional<IsConst, const void, void>::type;

    using value_type   = typename dw::core::copy_const<T, TConditionalConst>::type; ///< Type of element (including const)
    using element_type = T;                                                         ///< Type of element
    using pointer      = value_type*;                                               ///< Type of pointer to element
    using reference    = value_type&;                                               ///< Type of reference to element

    // definitions related to iterator_traits
    using iterator_category = std::random_access_iterator_tag; ///< Random-iterator tag
    using difference_type   = std::ptrdiff_t;                  ///< Type of interator diff

    /// create invalid iterator
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
    Iterator()
        : Iterator(span<value_type>(), 0)
    {
    }

    ~Iterator()                = default;            ///< destructor
    Iterator(Iterator&& other) = default;            ///< Move constructor
    Iterator& operator=(Iterator&& other) = default; ///< Move operator
    /// Conversion to const T
    /// Dummy template so this doesn't get defined for const iterators
    template <bool IsConst_ = IsConst, typename = typename std::enable_if<!IsConst_>::type>
    operator Iterator<T, true>() const // clang-tidy NOLINT
    {
        return Iterator<T, true>(m_entries, m_current);
    }

    /// create end-iterator
    explicit Iterator(span<value_type> entries) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : Iterator(entries, static_cast<ssize_t>(entries.size()))
    {
    }

    /// create iterator to a particular index
    Iterator(span<value_type> entries, ssize_t const index)
        : m_entries(std::move(entries))
        , m_current(index)
    {
    }

    /// copy constructor
    Iterator(const Iterator<T, IsConst>& other) = default;

    /// copy operator
    Iterator& operator=(const Iterator<T, IsConst>& other) = default;

    /// forward iterate (prefix)
    auto operator++() -> Iterator<T, IsConst>&
    {
        if (m_current < static_cast<ssize_t>(m_entries.size()))
        {
            ++m_current;
        }
        return *this;
    }

    /// backward iterate (prefix)
    auto operator--() -> Iterator<T, IsConst>&
    {
        if (m_current > 0)
        {
            --m_current;
        }
        else
        {
            m_current = -1;
        }
        return *this;
    }

    /// forward iterate (suffix)
    auto operator++(int32_t) -> Iterator<T, IsConst>
    {
        auto const before = *this;
        operator++();
        return before;
    }

    /// backward iterate (suffix)
    auto operator--(int32_t) -> Iterator<T, IsConst>
    {
        auto before = *this;
        operator--();
        return before;
    }

    /// forward iterate multi-step
    // TODO(dwplc): FP - basic numerical type "int" isn't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    template <typename TNumerical>
    auto operator+=(TNumerical const n) -> Iterator<T, IsConst>&
    {
        m_current += static_cast<ssize_t>(n);
        if ((m_current >= 0) && (static_cast<size_t>(m_current) > m_entries.size()))
        {
            m_current = static_cast<ssize_t>(m_entries.size());
        }
        return *this;
    }

    /// backward iterate multi-step
    // TODO(dwplc): FP - basic numerical type "int" isn't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    template <typename TNumerical>
    auto operator-=(TNumerical const n) -> Iterator<T, IsConst>&
    {
        if (m_current >= static_cast<ssize_t>(n))
        {
            m_current -= static_cast<ssize_t>(n);
        }
        else
        {
            m_current = -1;
        }
        return *this;
    }

    /// forward iterate multi-step, return new iterator
    // TODO(dwplc): FP - basic numerical type "int" isn't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    template <typename TNumerical>
    auto operator+(TNumerical const n) const -> Iterator<T, IsConst>
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto result = *this;
        result += n;
        return result;
    }

    /// backward iterate multi-step, return new iterator
    // TODO(dwplc): FP - basic numerical type "int" isn't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    template <typename TNumerical>
    auto operator-(TNumerical const n) const -> Iterator<T, IsConst>
    {
        auto result = *this;
        result -= n;
        return result;
    }

    /// difference of iterators
    template <bool OtherIsConst>
    ssize_t operator-(const Iterator<T, OtherIsConst>& other) const
    {
        return m_current - other.getCurrent();
    }

    // iterator comparison
    /// Inequality Operator
    template <bool OtherIsConst>
    bool operator!=(const Iterator<T, OtherIsConst>& other) const
    {
        // don't compare m_entries.size() to allow comparing during iterators even after changing the container,
        // for example needed when earase during iterate
        return (m_entries.data() != other.getEntries().data()) || (m_current != other.getCurrent());
    }

    /// Equality Operator
    template <bool OtherIsConst>
    bool operator==(const Iterator<T, OtherIsConst>& other) const
    {
        // don't compare m_entries.size() to allow comparing during iterators even after changing the container,
        // for example needed when earase during iterate
        return m_entries.data() == other.getEntries().data() && m_current == other.getCurrent();
    }

    /// Less-than Operator
    template <bool OtherIsConst>
    bool operator<(const Iterator<T, OtherIsConst>& other) const
    {
        if (m_entries.data() != other.getEntries().data())
        {
            throw ExceptionWithStackTrace("BaseVectorFixed Iterator: comparing iterators with of different vectors.\n");
        }

        return m_current < other.getCurrent();
    }

    /// Less-or-equal Operator
    template <bool OtherIsConst>
    bool operator<=(const Iterator<T, OtherIsConst>& other) const
    {
        if (m_entries.data() != other.getEntries().data())
        {
            throw ExceptionWithStackTrace("BaseVectorFixed Iterator: comparing iterators with of different vectors.\n");
        }

        return m_current <= other.getCurrent();
    }

    /// Greater-than Operator
    template <bool OtherIsConst>
    bool operator>(const Iterator<T, OtherIsConst>& other) const
    {
        if (m_entries.data() != other.getEntries().data())
        {
            throw ExceptionWithStackTrace("BaseVectorFixed Iterator: comparing iterators with of different vectors.\n");
        }

        return m_current > other.getCurrent();
    }

    /// Greater-or-equal Operator
    template <bool OtherIsConst>
    bool operator>=(const Iterator<T, OtherIsConst>& other) const
    {
        if (m_entries.data() != other.getEntries().data())
        {
            throw ExceptionWithStackTrace("BaseVectorFixed Iterator: comparing iterators with of different vectors.\n");
        }

        return m_current >= other.getCurrent();
    }

    /// iterator dereferencing
    auto operator*() const -> reference
    {
        return m_entries[static_cast<size_t>(m_current)];
    }

    /// get pointer to element
    auto getPtr() const -> pointer
    {
        return &(m_entries[static_cast<size_t>(m_current)]);
    }

    /// get pointer to element
    auto operator-> () const -> pointer
    {
        return getPtr();
    }

    /// get buffer of all entries
    auto getEntries() const -> span<value_type const>
    {
        return m_entries;
    }

    /// get index of iterator
    ssize_t getCurrent() const
    {
        return m_current;
    }

private:
    span<value_type> m_entries; ///< all entries
    ssize_t m_current;          ///< iterator index
};

/// This class implements a vector (similar to std::vector) that cannot be resized after construction
/// It should be used instead of std::vector to make sure that no runtime dynamic allocations are made
/// It simply mirrors all the methods of std::vector but adds checks in the methods that might
/// cause allocations.
///
/// The data of the vector is stored in: data, capacity, and size. There are different implementations
/// of the vector:
/// - StaticVector: data is owned and stored inside of the vector (usually used to create a vector on
///                the stack with a constexpr capacity).
/// - HeapVectorFixed: data is owned and stored in the heap.
/// - VectorRef: data is not owned. The vector receives data, capacity, and size as references and
///              modifies them as items are pushed/popped.
///
/// This base class is templated on the real vector type so the calls to size(), capacity(), and data()
/// will be inlined or optimized out when possible. This avoids virtual methods.
///
/// TStorage must implement:
///     /// The capacity of the buffer
///     /// Can be static constexpr
///     size_t capacity() const;
///
///     /// Reference to the size variable
///     size_t size() const;
///     size_t& size();
///
///     /// Pointer to buffer start
///     T *data();
template <typename T, class TStorage>
class BaseVectorFixed
{
public:
    using value_type      = T;                  ///< Type of values
    using size_type       = std::size_t;        ///< Type of size
    using difference_type = std::ptrdiff_t;     ///< Type of difference
    using reference       = value_type&;        ///< Type of reference
    using const_reference = value_type const&;  ///< Type of const reference
    using pointer         = value_type*;        ///< Type of pointer
    using const_pointer   = value_type const*;  ///< Type of const pointer
    using iterator        = Iterator<T, false>; ///< Type of iterator
    using const_iterator  = Iterator<T, true>;  ///< Type of const iterator

protected:
    /// Basic constructor
    explicit BaseVectorFixed(TStorage&& storage, char8_t const*&& hint)
        : m_storage(std::move(storage))
        , m_capacityHint(std::move(hint))
    {
    }

    /// Copies all elements from other container
    template <class S>
    BaseVectorFixed(TStorage&& storage, char8_t const*&& hint, BaseVectorFixed<T, S> const& other)
        : m_storage(std::move(storage))
        , m_capacityHint(std::move(hint))
    {
        push_back_range(make_span(other));
    }

    /// Moves all elements from other container
    template <class S>
    BaseVectorFixed(TStorage&& storage, char8_t const*&& hint, BaseVectorFixed<T, S>&& other)
        : m_storage(std::move(storage))
        , m_capacityHint(std::move(hint))
    {
        push_back_range_move(make_span(other));
        other.clear();
    }

public:
    BaseVectorFixed()
        : m_storage{}
        , m_capacityHint(nullptr)
    {
    }

    /// Copies all elements from other container
    template <class S>
    explicit BaseVectorFixed(const BaseVectorFixed<T, S>& other)
        : m_storage{}
        , m_capacityHint(other.getCapacityHint())
    {
        push_back_range(make_span(other));
    }

    /// Same as above but matches the exact type of this class
    BaseVectorFixed(const BaseVectorFixed& other)
        : m_storage{}
        , m_capacityHint(other.m_capacityHint)
    {
        push_back_range(make_span(other));
    }

    /// Moves all elements from other container
    template <class S>
    explicit BaseVectorFixed(BaseVectorFixed<T, S>&& other)
        : m_storage{}
        , m_capacityHint(other.getCapacityHint())
    {
        push_back_range_move(make_span(other));
        other.clear();
    }

    /// Moves all elements from other container
    BaseVectorFixed(BaseVectorFixed&& other)
        // TODO(dwplc): RFD - not moving is intentional here, for instance for static storage
        // coverity[autosar_cpp14_a12_8_4_violation]
        : m_storage{}
        , m_capacityHint(other.m_capacityHint)
    {
        push_back_range_move(make_span(other));
        other.clear();
    }

    BaseVectorFixed& operator=(const BaseVectorFixed& other) = default; ///< Copy operator
    BaseVectorFixed& operator=(BaseVectorFixed&& other) = default;      ///< Move operator
    ~BaseVectorFixed()                                  = default;      ///< Destructor

    /// Adds a string to the error messages thrown when the buffer is full
    /// It can be used to give the user a hint on how to control the capacity of this buffer
    /// E.g. buffer.allocate(MAX_ITEM_COUNT); buffer.setCapacityHint("MAX_ITEM_COUNT");
    void setCapacityHint(char8_t const* hint)
    {
        m_capacityHint = hint;
    }

    /// Return string that is stored as capacity hint (see setCapacityHint).
    char8_t const* getCapacityHint() const
    {
        return m_capacityHint;
    }

    /// Max number of elements this vector can contain
    size_t capacity() const { return m_storage.capacity(); }

    /// Number of elements this vector currently contains
    size_t size() const { return m_storage.size(); }

    /// Data buffer
    auto data() const -> const T* { return m_storage.data(); } ///< Get pointer to data
    auto data() -> T* { return m_storage.data(); }             ///< Get const pointer to data

    /// True if there are no elements in vector
    bool empty() const { return size() == 0U; }

    /// True if vector is full and no more elements can be added
    bool full() const { return size() == m_storage.capacity(); }

    /// Number of elements that can be added before it is full
    size_t available() const { return m_storage.capacity() - size(); }

    auto begin() -> iterator { return iterator(m_storage, 0); }                                                  ///< Get begin iterator
    auto end() -> iterator { return iterator(m_storage); }                                                       ///< Get end iterator
    auto begin() const -> const_iterator { return const_iterator(m_storage, 0); }                                ///< Get begin const-iterator
    auto end() const -> const_iterator { return const_iterator(make_span(m_storage.data(), m_storage.size())); } ///< Get end const-iterator

    auto front() -> T& { return (*this)[0]; }                     ///< Get reference to first element
    auto back() -> T& { return (*this)[size() - 1U]; }            ///< Get reference to last element
    auto front() const -> const T& { return (*this)[0]; }         ///< Get const reference to first element
    auto back() const -> const T& { return (*this)[size() - 1]; } ///< Get const reference to last element

    /// Destroy and remove all entries
    void clear()
    {
        // Destroy all
        for (auto& item : *this)
        {
            // TODO(dwplc): FP - constructor/destructor marked as C-style cast
            // coverity[autosar_cpp14_a5_2_2_violation]
            item.~T();
        }
        m_storage.size() = 0U;
    }

    /// Resize the vector to @a newSize. If this call grows the vector, new elements will be copy-constructed from
    /// @a value. If the current @ref size is larger than @a newSize, this vector will shrink and elements will be
    /// destroyed.
    void resize(size_type newSize, value_type const& value)
    {
        resizeImpl(newSize, [&value](pointer p) {
            new (p) value_type(value);
        });
    }

    /// Resize the vector to @a newSize. If this call grows the vector, new elements will be default-constructed in
    /// place.
    void resize(size_type newSize)
    {
        resizeImpl(newSize, [](pointer p) {
            new (p) value_type();
        });
    }

    /// Create an object directly on the storage at the back of the container if the container is not full.
    /// If return value is false, no element was added.
    template <typename... Args>
    bool emplace_back_maybe(Args&&... args)
    {
        if (size() >= m_storage.capacity())
        {
            return false;
        }
        else
        {
            try
            {
                span<T> const newStorageSpan{m_storage.data(), size() + 1U};
                // Copy construct in place
                new (&newStorageSpan[size()]) T(std::forward<Args>(args)...);
                m_storage.size() += 1U;
                return true;
            }
            catch (const dw::core::OutOfBoundsException&)
            {
                // This line shall never be reached because the bound
                // is checked in the entry if clause already.
                return false;
            }
        }
    }
    /// Push a new element at the back of the container if the container is not full
    /// If return value is false, no element was added
    bool push_back_maybe(T&& value)
    {
        return emplace_back_maybe(std::move(value));
    }

    /// Push a new element at the back of the container if the container is not full
    /// If return value is false, no element was added
    bool push_back_maybe(const T& value)
    {
        if (size() >= m_storage.capacity())
        {
            return false;
        }
        else
        {
            try
            {
                span<T> const newStorageSpan{m_storage.data(), size() + 1U};
                // Copy construct in place
                new (&newStorageSpan[size()]) T(value);
                m_storage.size() += 1U;
                return true;
            }
            catch (const dw::core::OutOfBoundsException&)
            {
                // This line shall never be reached because the bound
                // is checked in the entry if clause already.
                return false;
            }
        }
    }

    /// Create an object directly on the storage at the back of the container
    /// Returns a reference to the created object
    /// If the container is full an exception is thrown
    template <typename... Args>
    auto emplace_back(Args&&... args) -> T&
    {
        if (!emplace_back_maybe(std::forward<Args>(args)...))
        {
            if (m_capacityHint == nullptr)
            {
                throw BufferFullException("VectorFixed::emplace_back: full.");
            }
            else
            {
                // TODO(dwplc): FP - "Using basic numerical type "char" rather than a typedef that includes size and signedness information."
                // coverity[autosar_cpp14_a3_9_1_violation]
                throw BufferFullException("VectorFixed::emplace_back: full. "
                                          "Capacity is controlled by ",
                                          m_capacityHint);
            }
        }

        return back();
    }
    /// Push a new object at the back of the container
    /// Returns a reference to the created object
    /// If the container is full an exception is thrown
    auto push_back(T&& value) -> T&
    {
        return emplace_back(std::move(value));
    }
    /// Push a new object at the back of the container
    /// Returns a reference to the created object
    /// If the container is full an exception is thrown
    auto push_back(const T& value) -> T&
    {
        if (!push_back_maybe(value))
        {
            if (m_capacityHint == nullptr)
            {
                throw BufferFullException("VectorFixed::push_back: full.");
            }
            else
            {
                // TODO(dwplc): FP - "Using basic numerical type "char" rather than a typedef that includes size and signedness information."
                // coverity[autosar_cpp14_a3_9_1_violation]
                throw BufferFullException("VectorFixed::push_back: full. "
                                          "Capacity is controlled by ",
                                          m_capacityHint);
            }
        }

        return back();
    }

    /// Remove last element
    void pop_back()
    {
#if DW_RUNTIME_CHECKS()
        if (empty())
        {
            throw BufferEmptyException("VectorFixed::pop_back: empty");
        }
#endif

        T* const ptr{&back()};
        m_storage.size() -= 1U;
        // TODO(dwplc): FP - constructor/destructor marked as C-style cast
        // coverity[autosar_cpp14_a5_2_2_violation]
        ptr->~T();
    }

    /// Returns item at given index.
    auto inline operator[](size_t const idx) -> T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= size())
        {
#if !defined(__CUDACC__)
            throwVectorIndexOutOfBounds();
#else
            core::assertException<OutOfBoundsException>("VectorFixed: index is out of bounds");
#endif
        }
#endif

        return *(begin() + idx);
    }

    /// Returns const item at given index.
    auto inline operator[](size_t const idx) const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= size())
        {
#if !defined(__CUDACC__)
            throwVectorIndexOutOfBounds();
#else
            core::assertException<OutOfBoundsException>("VectorFixed: index is out of bounds");
#endif
        }
#endif

        return *(begin() + idx);
    }

    /// Returns item at given index.
    /// Always checks bounds.
    auto inline at(size_t const idx) -> T&
    {
        if (idx >= size())
        {
#if !defined(__CUDACC__)
            throwVectorIndexOutOfBounds();
#else
            core::assertException<OutOfBoundsException>("VectorFixed: index is out of bounds");
#endif
        }

        return *(begin() + idx);
    }

    /// Returns const item at given index.
    /// Always checks bounds.
    auto inline at(size_t const idx) const -> const T&
    {
        if (idx >= size())
        {
#if !defined(__CUDACC__)
            throwVectorIndexOutOfBounds();
#else
            core::assertException<OutOfBoundsException>("VectorFixed: index is out of bounds");
#endif
        }

        return *(begin() + idx);
    }

    /// Erases the given element by moving back() into it and then erasing back().
    /// This is the cheapest way to erase an element in the middle but changes the order of the elements.
    void erase_unordered(iterator const position)
    {
#if DW_RUNTIME_CHECKS()
        checkIteratorInBounds(position);
#endif

        if (position != begin() + size() - 1)
        {
            *position = std::move(back());
        }
        pop_back();
    }

    /// Erases the given element by moving back() into it and then erasing back().
    /// This is the cheapest way to erase an element in the middle but changes the order of the elements.
    void erase_unordered(size_t const position)
    {
        erase_unordered(begin() + position);
    }

    /// Erases the given element and relocates all the following elements up by one.
    /// This is more expensive than erase_unordered() but it guarantees the order of the elements is kept.
    void erase(iterator position)
    {
#if DW_RUNTIME_CHECKS()
        checkIteratorInBounds(position);
#endif
        iterator const itEnd = end();
        for (iterator itNext = position + 1; itNext != itEnd; ++itNext)
        {
            *position = std::move(*itNext);
            ++position;
        }
        pop_back();
    }

    /// Erase element at position
    void erase(size_t position)
    {
        erase(begin() + position);
    }

    /// Inserts a new element at the given position.
    /// Position must satisfy begin() <= position <= end().
    /// After return, the new element will be at position and all previous elements at or after position will have moved forward.
    /// Only checks argument if runtime checks are enabled.
    void insert(iterator position, T&& val)
    {
#if DW_RUNTIME_CHECKS()
        if (position < begin() || position > end())
        {
            throw OutOfBoundsException("VectorFixed: iterator is out of bounds");
        }
#endif

        if (position == end())
        {
            auto& ret = push_back(std::move(val));
            if (&ret != &back())
            {
                throw ExceptionWithStackTrace("VectorFixed::insert: unknown error");
            }
        }
        else
        {
            // extend by one element, move last element there
            auto& ret = push_back(std::move(back()));
            if (&ret != &back())
            {
                throw ExceptionWithStackTrace("VectorFixed::insert: unknown error");
            }

            // move remaining elements, starting at second last,
            // up to the element that is currently at 'position'
            for (iterator it = end() - 3; it >= position; it--)
            {
                *(it + 1) = std::move(*(it));
            }

            // move new element into 'position'
            *position = std::move(val);
        }
    }

    /// Inserts copy of element at position.
    void insert(iterator position, const T& val)
    {
        insert(position, std::move(static_cast<T>(val)));
    }

    /// Inserts element at position by moving.
    void insert(size_t position, T&& val)
    {
        insert(begin() + position, std::move(val));
    }

    /// Inserts copy of element at position.
    void insert(size_t position, const T& val)
    {
        insert(begin() + position, val);
    }

    /// Pushes a range of items in the back, copying each
    void push_back_range(span<const T> const items)
    {
        size_t const itemCount{items.size()};

        if (size() + itemCount > m_storage.capacity())
        {
            throw BufferFullException("VectorFixed cannot fit the passed range.");
        }

        // Copy construct in place
        size_t dstIdx{size()};
        span<T> const newStorageSpan{m_storage.data(), size() + itemCount};
        for (const T& item : items)
        {
            new (&newStorageSpan[dstIdx]) T(item); // Copy construct
            ++dstIdx;
        }

        m_storage.size() += itemCount;
    }

    /// Pushes a range of items in the back, copying each
    void push_back_range(std::initializer_list<T> const items)
    {
        push_back_range(make_span(items.begin(), items.size()));
    }

    /// Pushes a range of items in the back, moving each
    void push_back_range_move(span<T> const items)
    {
        size_t const itemCount{items.size()};

        if (size() + itemCount > m_storage.capacity())
        {
            throw BufferFullException("VectorFixed cannot fit the passed range.");
        }

        // Copy construct in place
        size_t dstIdx{size()};
        span<T> const newStorageSpan{m_storage.data(), size() + itemCount};
        for (T& item : items)
        {
            new (&newStorageSpan[dstIdx]) T(std::move(item)); // Move construct
            ++dstIdx;
        }

        m_storage.size() += itemCount;
    }

    /// Pushes a range of items in the back, copying each
    template <class TValue = T>
    void push_back_range(size_t const itemCount, TValue const value)
    {
        if (size() + itemCount > m_storage.capacity())
        {
            throw BufferFullException("VectorFixed cannot fit the passed range.");
        }

        // Copy construct in place
        size_t dstIdx = size();
        span<T> const newStorageSpan(m_storage.data(), size() + itemCount);
        for (size_t i = 0; i < itemCount; ++i)
        {
            new (&newStorageSpan[dstIdx]) T(value); // Copy construct
            ++dstIdx;
        }

        m_storage.size() += itemCount;
    }

    /// Return true if other has the same size and all
    /// elements at the corresponding indices are equal.
    template <class D2>
    bool isEqual(const BaseVectorFixed<T, D2>& other) const noexcept
    {
        try
        {
            if (size() != other.size())
            {
                return false;
            }

            for (size_t i = 0, iend = size(); i < iend; i++)
            {
                if (this->at(i) == other.at(i))
                { // Avoid using != operator
                    continue;
                }
                return false;
            }
        }
        catch (dw::core::OutOfBoundsException const& ex)
        {
            return false;
        }
        catch (std::exception const& ex)
        {
            return false;
        }
        catch (...)
        {
            return false;
        }

        return true;
    }

    /// Copies elements from a span.
    /// Warning: this operator will call the copy-constructor of T if the size of the current
    /// vector is less than the other vector. If T allocates in its constructor,
    /// this may violate the no-dynamic-memory rule.
    void copyFrom(span<const T> const other)
    {
        if (m_storage.capacity() < other.size())
        {
            throw BufferFullException("VectorFixed: cannot assign vector because destination "
                                      "capacity is less than the source size.");
        }

        size_t const minSize{std::min(size(), other.size())};

        // first copy into existing items
        for (size_t i{0U}; i < minSize; i++)
        {
            (*this)[i] = other[i];
        }

        // delete extra items
        while (size() > minSize)
        {
            pop_back();
        }

        // copy-construct missing items
        push_back_range(other + minSize);
    }

    /// Moves elements from a span
    /// Warning: this operator will call the move-constructor of T if the size of the current
    /// vector is less than the other vector. If T allocates in its constructor,
    /// this may violate the no-dynamic-memory rule.
    void moveFrom(span<T> const other)
    {
        if (m_storage.capacity() < other.size())
        {
            throw BufferFullException("VectorFixed: cannot assign vector because destination "
                                      "capacity is less than the source size.");
        }

        size_t const minSize{std::min(size(), other.size())};

        // first move into existing items
        for (size_t i{0U}; i < minSize; i++)
        {
            (*this)[i] = std::move(other[i]);
        }

        // delete extra items
        while (size() > minSize)
        {
            pop_back();
        }

        // move-construct missing items
        push_back_range_move(other + minSize);
    }

    /// Copy operator
    template <class S>
    auto operator=(const BaseVectorFixed<T, S>& other) -> BaseVectorFixed&
    {
        copyFrom(other);
        return *this;
    }

    /// Move operator
    template <class S>
    auto operator=(BaseVectorFixed<T, S>&& other) -> BaseVectorFixed&
    {
        moveFrom(other);
        return *this;
    }

    /// Equality operator
    template <class D2>
    bool operator==(const BaseVectorFixed<T, D2>& other) const
    {
        return isEqual(other);
    }

protected:
    /// Set storage
    void setStorage(TStorage& storage)
    {
        m_storage = std::move(storage);
    }

    /// Set storage
    void setStorage(TStorage const& storage)
    {
        m_storage = storage;
    }

    /// Get storage
    auto getStorage() -> TStorage&
    {
        return m_storage;
    }

    /// Get storage
    auto getStorage() const -> const TStorage&
    {
        return m_storage;
    }

    /// Check if iterator is valid
    void checkIteratorInBounds(const_iterator const position) const
    {
        if (position < begin() || position >= end())
        {
            throw OutOfBoundsException("VectorFixed: iterator is out of bounds");
        }
    }

private:
    TStorage m_storage;
    char8_t const* m_capacityHint;

    template <typename FConstruct>
    void resizeImpl(size_type newSize, const FConstruct&& construct)
    {
        if (newSize > m_storage.capacity())
        {
            if (m_capacityHint == nullptr)
            {
                throw BufferFullException("VectorFixed::resize: "
                                          "requested size (",
                                          newSize, ") is bigger than capacity (", m_storage.capacity(), ").");
            }
            else
            {
                throw BufferFullException("VectorFixed::resize: "
                                          "requested size is bigger than capacity. ",
                                          "Capacity is controlled by ", m_capacityHint);
            }
        }

        if (newSize > size())
        {
            // Add new items
            auto uninitializedStorage = span<T>(m_storage.data(), newSize).subspan(size());
            for (auto& unconstructedElem : uninitializedStorage)
            {
                construct(&unconstructedElem);
                m_storage.size() += 1;
            }

            if (size() != newSize)
            {
                throw OutOfBoundsException("???", size(), " vs ", newSize);
            }
        }
        else
        {
            while (newSize < size())
            {
                pop_back();
            }
        }
    }
};

} // namespace core
} // namespace dw

#endif
