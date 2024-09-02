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

#ifndef DWSHARED_CORE_HASHMAP_HPP_
#define DWSHARED_CORE_HASHMAP_HPP_

#include <memory>
#include <limits>
#include <functional>
#include <dwshared/dwfoundation/dw/core/language/Hash.hpp>
#include "../../VectorFixed.hpp"

namespace dw
{
namespace core
{

/*************************************************************************************************************/
/**
  * HashMap and HashSet
  *
  *
  * These are fixed sized containers that allow cache efficient element lookup with a key.
  *
  * It should be avoided to fully fill the container, as performance degrades due to hash collisions.
  *
  * Iterating the elements is supported, however if the usage relies on fast
  * iteration, it is recommended to use HashMapIterable / HashSetIterable instead, especially
  * if there are only few elements relative to the reserved capacity.
  * This also applies to the clear().
  */

// Helper class that enables to select the appropriate hashing function
template <class T>
struct HashSelector
{
    static typename std::conditional<std::is_enum<T>::value, EnumHash<T>, std::hash<T>>::type test(...);

    template <class U = T>
    static auto test(bool) -> PairHash<typename U::first_type, typename U::second_type>;

    using type = decltype(test(true)); // clang-tidy NOLINT(cppcoreguidelines-pro-type-vararg)
};

/// Default hash that can handle enums and pairs as well
template <class T>
using DefaultHash = typename HashSelector<T>::type;

/*************************************************************************************************************/
/**
 * Iterator for the hash container
 */
template <class TEntry, class TElement, bool IsConst>
class HashIterator
{
public:
    /// const or non-const ptr to entries
    using TConstRef = typename std::conditional<IsConst, const void, void>::type;

    using value_type = typename dw::core::copy_const<TElement, TConstRef>::type; ///< Type of element iterator is pointing to
    using pointer    = value_type*;                                              ///< Type of pointer to element
    using reference  = value_type&;                                              ///< Type of reference of element

    using entry         = typename dw::core::copy_const<TEntry, TConstRef>::type; ///< Type of entry
    using entry_pointer = entry*;                                                 ///< Type of poitner to entry

    // definitions related to iterator_traits
    using iterator_category = std::forward_iterator_tag; ///< Tag as forward iterator
    using difference_type   = std::ptrdiff_t;            ///< Type of iterator difference

    /// create invalid iterator
    HashIterator()
        : m_entries{}, m_current{}
    {
    }

    /// Conversion to const T
    /// Dummy template so this doesn't get defined for const iterators
    template <bool IsConst_ = IsConst, typename = typename std::enable_if<!IsConst_>::type>
    operator HashIterator<TEntry, TElement, true>() const // clang-tidy NOLINT
    {
        return toConst();
    }

    /// Conversion to const T
    /// Dummy template so this doesn't get defined for const iterators
    /// Avoid implicit conversation flagged by AUTOSAR rule A23-0-1
    template <bool IsConst_ = IsConst, typename = typename std::enable_if<!IsConst_>::type>
    HashIterator<TEntry, TElement, true> toConst() const
    {
        return HashIterator<TEntry, TElement, true>(m_entries, m_current);
    }

    /// create end-iterator
    explicit HashIterator(span<entry> entries) // NOLINT(cppcoreguidelines-pro-type-member-init) - FP about uninitialized members
        : HashIterator(entries, entries.size())
    {
    }

    /// create iterator to a particular index
    HashIterator(span<entry> entries, size_t const index)
        : m_entries(std::move(entries))
        , m_current(index)
    {
    }

    /// set iterator to next valid element, element order is not defined
    auto operator++() -> HashIterator&
    {
        bool done{m_current >= m_entries.size()};
        while (!done)
        {
            ++m_current;
            done = (m_current >= m_entries.size()) || (m_entries[m_current].has_value());
        }
        return *this;
    }

    // iterator comparison

    /// Inequality operator
    template <bool OtherIsConst>
    bool operator!=(const HashIterator<TEntry, TElement, OtherIsConst>& other) const
    {
        return (m_entries != other.getEntries()) || (m_current != other.getCurrent());
    }

    /// Equality operator
    template <bool OtherIsConst>
    bool operator==(const HashIterator<TEntry, TElement, OtherIsConst>& other) const
    {
        return m_entries == other.getEntries() && m_current == other.getCurrent();
    }

    // iterator dereferencing

    /// Dereference
    auto operator*() const -> reference
    {
        return *m_entries[m_current];
    }

    /// Get pointer
    auto operator-> () const -> pointer
    {
        return &(*m_entries[m_current]);
    }

    /// Get index
    size_t const& getCurrent() const
    {
        return m_current;
    }

    /// Get span of entries
    auto getEntries() const -> span<entry> const&
    {
        return m_entries;
    }

private:
    span<entry> m_entries; ///< Span of entries
    size_t m_current;      ///< Index the iterator is pointing to
};

/**
 * Hash set/map entry. Subset of Optional.
 * Re-implemented because Optional has functions marked as __device__, causing
 * nvcc to error out when it is parametrized by FixedString or pair types (as
 * used by the Hash containers) as they are not CUDA compatible.
 * TODO(lbrusatin, DRIV-7944): fix root cause and revert to using Optional
 */
template <class TElement>
class HashEntry
{
public:
    constexpr HashEntry() = default;

    /// Copy constructor
    // TODO(dwplc, AVDWPLC-844): Defining a copy or move constructor with side effects.
    // coverity[autosar_cpp14_a12_8_1_violation]
    inline HashEntry(const HashEntry& other)
    {
        if (other.has_value())
        {
            m_isValid = true;
            new (&m_valueBuffer) TElement(*other);
        }
    }

    /// Move constructor
    // TODO(dwplc, AVDWPLC-844): Defining a copy or move constructor with side effects.
    // coverity[autosar_cpp14_a12_8_1_violation]
    // coverity[autosar_cpp14_a12_8_4_violation]
    inline HashEntry(HashEntry&& other)
    {
        if (other.has_value())
        {
            m_isValid = true;
            new (&m_valueBuffer) TElement(std::move(*other));
        }
    }

    /// Value constructor
    inline HashEntry(const TElement& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
    {
        new (&m_valueBuffer) TElement(valueIn);
        m_isValid = true;
    }

    /// Move value constructor
    inline HashEntry(TElement&& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
    {
        new (&m_valueBuffer) TElement(std::move(valueIn));
        m_isValid = true;
    }

    /// Destructor
    inline ~HashEntry()
    {
        reset(); // makes sure value is destructed
    }

    /// True if entry contains a valid element
    inline explicit operator bool() const { return m_isValid; }

    /// True if entry contains a valid element
    inline bool has_value() const { return m_isValid; }

    /// Construct new element in-place
    template <typename... Args>
    inline void emplace(Args&&... args)
    {
        reset(); // destroy any previously existing value

        new (&m_valueBuffer) TElement{std::forward<Args>(args)...};
        m_isValid = true;
    }

    /// Move operator
    inline auto operator=(HashEntry&& other) -> HashEntry&
    {
        if (other.has_value())
        {
            *this = std::move(*other);
        }
        else
        {
            reset();
        }
        return *this;
    }

    /// Copy operator
    inline auto operator=(const HashEntry& other) -> HashEntry&
    {
        if (other.has_value())
        {
            *this = *other;
        }
        else
        {
            reset();
        }
        return *this;
    }

    /// Move element into this entry
    template <class U>
    inline auto operator=(U&& other)
        -> typename std::enable_if<
            std::is_same<U, TElement>::value,
            HashEntry&>::type
    {
        if (m_isValid)
        {
            bufferAsValue() = std::move(other); // clang-tidy NOLINT(bugprone-move-forwarding-reference) Ok to silence because U is always equal to T
        }
        else
        {
            new (&m_valueBuffer) TElement(std::move(other)); // clang-tidy NOLINT(bugprone-move-forwarding-reference) Ok to silence because U is always equal to T
            m_isValid = true;
        }

        return *this;
    }

    /// Copy element into this entry
    template <class U>
    inline auto operator=(const U& other)
        -> typename std::enable_if<
            std::is_same<U, TElement>::value,
            HashEntry&>::type
    {
        if (m_isValid)
        {
            bufferAsValue() = other;
        }
        else
        {
            new (&m_valueBuffer) TElement(other);
            m_isValid = true;
        }
        return *this;
    }

    /// Dereferencing return entry's element
    inline auto operator*() -> TElement&
    {
        return bufferAsValue();
    }

    /// Dereferencing return entry's element
    inline auto operator*() const -> const TElement&
    {
        return bufferAsValue();
    }

    /// If entry contains an element, destroy it. Set to invalid.
    inline void reset()
    {
        if (m_isValid)
        {
            // TODO(dwplc, AVDWPLC-844): Using traditional C-style cast in "(void)this->bufferAsValue()"
            // coverity[autosar_cpp14_a5_2_2_violation]
            bufferAsValue().~TElement();
        }
        m_isValid = false;
    }

private:
    inline auto bufferAsValue() -> TElement&
    {
        return *reinterpret_cast<TElement*>(&m_valueBuffer); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }
    inline auto bufferAsValue() const -> TElement const&
    {
        return *reinterpret_cast<TElement const*>(&m_valueBuffer); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    // use storage with same alignment as wrapped value type
    typename std::aligned_storage<sizeof(TElement), alignof(TElement)>::type m_valueBuffer = {};

    // place valid flag *after* value to not enlarge total type's size too much due to alignment requirements
    bool m_isValid = false;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Base class with common functioniality for HashMap and HashSet.
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

template <class TKey,            //!< type of hash keys
          class TElement_,       //!< type of container elements.
                                 //!  for HashSet this is the same as TKey,
                                 //!  for HashMap this is a std::pair<TKey,TValue>
          class TKeyFromElement, //!< function object class to get the key to an element,
                                 //!  for HashSet this returns the element itself,
                                 //!  for HashMap this returns the key of the key-value pair.
          class THash,           //!< hash function object class, creates a hash value from a key
          class TKeyEqual,       //!< function object class to test key equality
          class TStorage>        //!< Storage for elements
class BaseHashContainer
{
public:
    // to make the type accessible
    using key_type   = TKey;      ///< Type of hash keys
    using value_type = TElement_; ///< Type of container elements
    using TElement   = TElement_; ///< Type of container elements

    using Entry = HashEntry<TElement>; ///< Type of element slot on container

    // iterator type
    using iterator       = HashIterator<Entry, TElement, false>; ///< Type of iterator
    using const_iterator = HashIterator<Entry, TElement, true>;  ///< Type of const-iterator

protected:
    /// Basic constructor when storage needs no configuring
    BaseHashContainer() = default;

    /// Basic constructor
    explicit BaseHashContainer(TStorage&& storage)
        : m_storage(std::move(storage))
    {
    }

    /// Copies all elements from other container
    template <class S2>
    explicit BaseHashContainer(const BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>& other)
        : m_storage{}
    {
        insert_range(other.begin(), other.end());
    }

    /// Moves all elements from other container
    template <class S2>
    explicit BaseHashContainer(BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>&& other)
        : m_storage{}
    {
        insert_range_move(other.begin(), other.end());
        other.clear();
    }

    /// Copies all elements from other container
    template <class S2>
    BaseHashContainer(TStorage&& storage, const BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>& other)
        : m_storage(std::move(storage))
    {
        insert_range(dw::core::make_span(other));
    }

    /// Moves all elements from other container
    template <class S2>
    BaseHashContainer(TStorage&& storage, BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>&& other)
        : m_storage(std::move(storage))
    {
        insert_range_move(dw::core::make_span(other));
        other.clear();
    }

    /// Construction from initializer list
    BaseHashContainer(std::initializer_list<TElement> const ilist)
        : m_storage{}
    {
        for (TElement const entry : ilist)
        {
            if (!this->insert(entry))
            {
                if (this->full())
                {
                    throw BufferFullException("HashContainer: Cannot initialize from given list, capacity is too small");
                }
                else
                {
                    throw InvalidArgumentException("HashContainer: Element insertion failed. This could be caused by duplicate keys in the initializer list");
                }
            }
        }
    }

    /// Construction from storage and initializer list
    BaseHashContainer(TStorage&& storage, std::initializer_list<TElement> const ilist)
        : m_storage{std::move(storage)}
    {
        for (TElement const entry : ilist)
        {
            if (!this->insert(entry))
            {
                if (this->full())
                {
                    throw BufferFullException("HashContainer: Cannot initialize from given list, capacity is too small");
                }
                else
                {
                    throw InvalidArgumentException("HashContainer: Element insertion failed. This could be caused by duplicate keys in the initializer list");
                }
            }
        }
    }

public:
    /// Max number of elements this vector can contain
    size_t capacity() const { return m_storage.capacity(); }

    /// Number of elements this vector currently contains
    size_t size() const { return m_storage.size(); }

    /// True if there are no elements in vector
    bool empty() const { return size() == 0; }

    /// True if vector is full and no more elements can be added
    bool full() const { return size() == m_storage.capacity(); }

    /// Number of elements that can be added before it is full
    size_t available() const { return m_storage.capacity() - size(); }

    // iterators

    /// Get begin iterator
    auto begin() -> iterator
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto dataSpan = m_storage.data();
        iterator it{dataSpan, 0U};
        if ((dataSpan.size() > 0U) && (!dataSpan[0].has_value()))
        {
            ++it; // go to first used entry (look at implementation of operator++ in iterator)
        }
        return it;
    }

    /// Get begin const-iterator
    auto begin() const -> const_iterator
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto dataSpan = m_storage.data();
        const_iterator it{dataSpan, 0U};
        if (dataSpan.size() > 0U && !dataSpan[0].has_value())
        {
            ++it; // go to first used entry (look at implementation of operator++ in iterator)
        }
        return it;
    }

    /// Get end iterator
    auto end() -> iterator
    {
        return iterator(m_storage.data());
    }

    /// Get end const-iterator
    auto end() const -> const_iterator
    {
        return const_iterator(m_storage.data());
    }

    /// Returns an iterator to the element with given key, end iterator if none is found.
    template <class TKeyArg>
    auto find(TKeyArg&& key) -> iterator
    {
        size_t index{0U};
        bool const found{findIndex(&index, std::forward<TKeyArg>(key), false)};

        if (found && m_storage[index].has_value())
        {
            return iterator(m_storage.data(), index);
        }

        return end();
    }

    /// Returns an iterator to the element with given key, end iterator if none is found.
    template <class TKeyArg>
    auto find(TKeyArg&& key) const -> const_iterator
    {
        size_t index{0U};
        bool const found{findIndex(&index, std::forward<TKeyArg>(key), false)};

        if (found && m_storage[index].has_value())
        {
            return const_iterator(m_storage.data(), index);
        }

        return end();
    }

    /// True if container contains an element with given key.
    template <class TKeyArg>
    bool contains(TKeyArg&& key) const
    {
        return find(std::forward<TKeyArg>(key)) != end();
    }

    /// Return the number of elements in the container with given key.
    template <class TKeyArg>
    size_t count(TKeyArg&& key) const
    {
        return contains(std::forward<TKeyArg>(key)) ? 1 : 0;
    }

    /// insert element if there is no entry yet for the key of this element.
    /// Returns false if element was not inserted. This could happen if:
    /// - Container is full
    /// - No index could be found for the given element
    /// - There is already an entry for the key of the given element
    bool insert(TElement const& element)
    {
        if (size() >= capacity())
        {
            return false;
        }

        size_t index{0U};
        bool const found{findIndex(&index, *TKeyFromElement()(&element), false)};

        bool inserted{found};
        if (found)
        {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto& entry = m_storage[index];
            if (!entry.has_value())
            {
                // entry not there yet, add the new entry
                entry = element;
                m_storage.size() += 1U;
            }
            else
            {
                inserted = false;
            }
        }

        return inserted;
    }

    /// Insert with move semantics
    bool insert(TElement&& element)
    {
        if (size() >= capacity())
        {
            return false;
        }

        size_t index{0U};
        bool const found{findIndex(&index, *(TKeyFromElement()(&element)), false)};

        bool inserted{found};
        if (found)
        {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto& entry = m_storage[index];
            if (!entry.has_value())
            {
                // entry not there yet, add the new entry
                entry.emplace(std::move(element));
                m_storage.size() += 1U;
            }
            else
            {
                inserted = false;
            }
        }

        return inserted;
    }

    /// Copy all elements from begin to end iterator
    template <class TIterator>
    void insert_range(TIterator const beginRange, TIterator const endRange)
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        for (auto it = beginRange; it != endRange; ++it)
        {
            if (!insert(*it))
            {
                throw BufferFullException("HashContainer: insert_range could not insert");
            }
        }
    }

    /// Move all elements from begin to end iterator into container
    template <class TIterator>
    void insert_range_move(TIterator const beginRange, TIterator const endRange)
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        for (auto it = beginRange; it != endRange; ++it)
        {
            if (!insert(std::move(*it)))
            {
                throw BufferFullException("HashContainer: insert_range_move could not insert");
            }
        }
    }

    /// Erase element the given iterator refers to.
    /// Returns an iterator to the successor of the erased element in the container.
    auto erase(const_iterator it) -> iterator
    {
        // check that the iterator belongs to this container
        if (it.getEntries() != m_storage.data())
        {
            LOGSTREAM_WARN(this) << "HashContainer: tried to erase with invalid iterator.\n";
            return end();
        }

        size_t index{it.getCurrent()};
        bool erased{false};
        if (m_storage[index].has_value())
        {
            erased = eraseAtIndex(index);
            if (erased)
            {
                --m_storage.size();
            }
        }

        // Create new iterator at index to return.
        // eraseAtIndex fills up the gaps, so the next element will be
        // at that index now.
        iterator nextAfterErase{iterator(m_storage.data(), index)};
        if (!m_storage[index].has_value())
        {
            // If eraseAtIndex did not have an element to fill up the gap,
            // move forward to the next valid entry.
            ++nextAfterErase;
        }

        return nextAfterErase;
    }

    /// Erase element with given key.
    /// Returns true if an element has been erased, false otherwise.
    bool erase(const TKey& key)
    {
        size_t index{0U};
        bool const ok{findIndex(&index, key, false)};
        bool erased{false};
        if ((ok && (m_storage[index].has_value())) && (TKeyEqual()(*TKeyFromElement()(&(*m_storage[index])), key)))
        {
            // entry found, erase it
            erased = eraseAtIndex(index);
            if (erased)
            {
                m_storage.size() -= 1U;
            }
        }

        return erased;
    }

    /// Clear container.
    /// Destroys all contained elements and sets size to 0.
    void clear()
    {
        iterator const endIt{end()};
        if (m_storage.size() > 0U)
        {
            for (iterator it{begin()}; it != endIt; ++it)
            {
                m_storage[it.getCurrent()].reset();
            }

            m_storage.size() = 0U;
        }
    }

    /// Clear this container and copy all elements from other container into this.
    template <class S2>
    void copyFrom(const BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>& other)
    {
        clear();
        insert_range(other.begin(), other.end());
    }

    /// Clear this container and move all elements from other container into this.
    template <class S2>
    void moveFrom(BaseHashContainer<TKey, TElement, TKeyFromElement, THash, TKeyEqual, S2>&& other)
    {
        if (this != &other)
        {
            clear();
            insert_range_move(other.begin(), other.end());
            other.clear();
        }
    }

    /// Replace an existing element that has the same key as the provided
    /// element with the provided element.
    /// If no element with the same key exists an exception is thrown.
    void updateExistingEntry(TElement const& element)
    {
        size_t index{0U};
        bool const found{findIndex(&index, *TKeyFromElement()(&element), false)};
        if (!found)
        {
            throw InvalidStateException("HashContainer: entry to update not found.");
        }
        m_storage[index] = element;
    }

    /// If there are few elements in the HashMap, only iterate and reset
    /// the existing entries (for better efficiency)
    template <class TEntry, class _TKeyFromElement>
    void clear(TEntry const& entries, const _TKeyFromElement& keyFromElement)
    {
        for (uint32_t i{0U}; i < entries.size(); ++i)
        {
            clearExistingEntry(keyFromElement(&entries[i]));
        }
        if (size() > 0U)
        {
            throw InvalidStateException("HashContainer: Not all elements have been removed during clear!");
        }
    }

protected:
    /// Caller has to know that the entry is there. This can only be used with the goal of clearing all
    /// entries, as after this call the hash container structure is invalid!
    void clearExistingEntry(TKey const& key)
    {
        size_t const index{findExistingIndex(key)};
        if (!m_storage[index].has_value())
        {
            throw InvalidStateException("HashContainer::clearExistingEntry: trying to clear unused entry!");
        }
        m_storage[index].reset();
        m_storage.size() -= 1U;
    }

    /// Erase an entry at given index.
    /// Other entries that could have been inserted here, but were inserted
    /// somewhere else due to a hash collision need to be moved back to occupy the empty
    /// slot. Otherwise their index would not be found anymore in findIndex, as the search
    /// stops at empty slots.
    bool eraseAtIndex(size_t eraseIndex)
    {
        // check that there's a valid entry at given index
        if (!m_storage[eraseIndex].has_value())
        {
            return false;
        }

        // do at most capacity() moves, to avoid infinite loop at full container
        size_t maxMoveCount{capacity()};
        while (maxMoveCount > 0U) // can't use a for-loop due to AUTOSAR A6-5-1
        {
            // find last entry that can be moved to empty cell

            // erase element at given index.
            // the following findIndex call will stop here if it reaches this index
            m_storage[eraseIndex].reset();

            size_t movableIndex{std::numeric_limits<size_t>::max()};
            size_t nextIndex{eraseIndex};
            size_t maxIndexIterationCount{capacity()};
            while (maxIndexIterationCount > 0U) // can't use a for-loop due to AUTOSAR A6-5-1
            {
                // look at next entry of collision resolution policy
                nextIndex = getNextIndex(nextIndex);
                const Entry& nextEntry{m_storage[nextIndex]};

                // if it's an empty slot, there are no hash collisions, we can stop here
                if (!nextEntry.has_value())
                {
                    break;
                }

                // check at which index the next entry would end up, if it were to be reinserted
                size_t newIndex{std::numeric_limits<size_t>::max()};

                // a multimap/multiset would need to stop only at empty elements,
                // but for normal hash containers, it is also ok to stop an existing key already.
                constexpr bool FIND_EMPTY_SLOT{true};
                if (findIndex(&newIndex, *TKeyFromElement()(&(*nextEntry)), !FIND_EMPTY_SLOT))
                {
                    if (newIndex == eraseIndex)
                    {
                        // if the next entry would end up in the erased slot, we can move it there.
                        // So remember it and continue trying, such that in the end we'll just move
                        // the last movable one in the collision resolution chain.
                        movableIndex = nextIndex;
                        break;
                    }
                }

                --maxIndexIterationCount;
            }

            if (movableIndex == std::numeric_limits<size_t>::max())
            {
                break; // done
            }
            else
            {
                // copy the movable entry, and erase it at its old slot
                m_storage[eraseIndex] = std::move(m_storage[movableIndex]);
                eraseIndex            = movableIndex;
            }

            --maxMoveCount;
        }

        return true;
    }

    /// linear probing
    /// we could also try quadratic probing or double hashing
    /// trade-off between cache-locality and hash collisions
    inline size_t getNextIndex(size_t const index) const
    {
        return (index + 1U) % capacity();
    }

    /// find the index of a key into the hash table.
    /// index will be an empty slot where the key can be stored, or the index
    /// of an existing key in the hash table, if 'findEmptySlot' is false.
    template <class TKeyArg>
    bool findIndex(size_t* const index, TKeyArg const& key, bool const findEmptySlot) const
    {
        if (capacity() == 0U)
        {
            return false;
        }

        // start search at hash value of the key
        *index = THash()(key) % capacity();

        Entry const* entry{&(m_storage[*index])};
        size_t collisionCount{0U};
        while (((entry->has_value()) &&                                                   // slot not empty
                (findEmptySlot || (!TKeyEqual()(*TKeyFromElement()(&(*(*entry))), key)))) // check slot is occupied
                                                                                          // by the same key
               && (collisionCount < capacity()))                                          // avoid infinite loop if all entries are used
        {
            // hash collision, check next index
            *index = getNextIndex(*index);
            entry  = &m_storage[*index];
            ++(collisionCount);
        }

        bool found{true};
        if (collisionCount >= capacity())
        {
            found = false;
        }
        return found;
    }

    /// Find index of element with given key, with
    /// prior knowledge that such an element exists.
    size_t findExistingIndex(const TKey& key) const
    {
        // start search at hash value of the key
        size_t index{THash()(key) % capacity()};

        // Search for key, don't stop the search at unused slots,
        // as entries with colliding hashes might have been cleared already.
        // The caller has to know that the key exists in m_entries.
        Entry const* entry{&(m_storage[index])};
        size_t counter{0U};
        while (!(entry->has_value() && TKeyEqual()(*TKeyFromElement()(&(**entry)), key)))
        {
            index = getNextIndex(index);
            entry = &m_storage[index];
            ++counter;

            if (counter >= capacity())
            {
                throw InvalidStateException("HashContainer::findExistingIndex: index not found!");
            }
        }

        return index;
    }

protected:
    /// constant used to set container capacity when constructed with initializer list
    static constexpr float32_t INIT_LIST_LOAD_FACTOR{1.4F};

    auto storage() -> TStorage& { return m_storage; }             ///< Get reference to storage
    auto storage() const -> TStorage const& { return m_storage; } ///< Get const reference to storage

private:
    TStorage m_storage; ///< storage
};

/************************************************************************************************************/
/**
 * Function object used in the HashContainer base class
 * to access the key from a HashMap element
 */
template <class TKey, class TElement>
class HashMapKeyFromElement
{
public:
    /// Key is first of element
    const TKey* operator()(TElement const* const element) const
    {
        // for HashMap TElement has to be a key-value pair
        return &element->first;
    }
};

/*************************************************************************************************************/
/**
 * HashMap
 */
template <class TKey,
          class TValue,
          class THash,
          class TKeyEqual,
          class TStorage>
class BaseHashMap : public BaseHashContainer<TKey,
                                             std::pair<TKey, TValue>,
                                             HashMapKeyFromElement<TKey, std::pair<TKey, TValue>>,
                                             THash,
                                             TKeyEqual,
                                             TStorage>
{
public:
    /// Type of base class
    using Base = BaseHashContainer<TKey, std::pair<TKey, TValue>, HashMapKeyFromElement<TKey, std::pair<TKey, TValue>>, THash, TKeyEqual, TStorage>;

    using mapped_type    = TValue;                        ///< Type of value
    using iterator       = typename Base::iterator;       ///< Type of iterator
    using const_iterator = typename Base::const_iterator; ///< Type of const iterator

    using Base::end;
    using Base::find;
    using Base::findIndex;
    using Base::insert;
    using Base::storage;

protected:
    // Inherit all constructors
    using Base::Base;

public:
    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, throw an exception.
    template <class TKeyArg>
    auto at(TKeyArg&& key) -> TValue&
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const iter = find(std::forward<TKeyArg>(key));
        if (iter == end())
        {
            throw dw::core::OutOfBoundsException("HashMap: Key does not exist");
        }
        return iter->second;
    }

    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, throw an exception.
    template <class TKeyArg>
    auto at(TKeyArg&& key) const -> const TValue&
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const iter = find(std::forward<TKeyArg>(key));
        if (iter == end())
        {
            throw OutOfBoundsException("HashMap: Key does not exist");
        }

        return iter->second;
    }

    /// HashMap specific operator.
    /// Insert an element in-place if there is no entry yet for the key of this element.
    /// Returns a pair consisting of an iterator to the inserted element, or the already-existing element,
    /// and a bool denoting whether the insertion took place.
    ///
    /// Insertion does not happen if:
    /// - Container is full
    /// - No index could be found for the given element
    /// - There is already an entry for the key of the given element
    template <class TKeyArg, class... Args>
    auto emplace(TKeyArg&& key, Args&&... args) -> std::pair<iterator, bool>
    {
        size_t index{0U};
        bool found{false};
        bool inserted{false};

        found = findIndex(&index, key, false);
        if (found)
        {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto& entry = storage()[index];
            if (!entry.has_value())
            {
                // entry not there yet, add the new entry
                entry.emplace(std::piecewise_construct,
                              std::forward_as_tuple(key),
                              std::forward_as_tuple(std::forward<Args>(args)...));
                storage().size() += 1U;
                inserted = true;
            }
        }

        iterator element{end()};

        if (found)
        {
            element = iterator(storage().data(), index);
        }

        return {element, inserted};
    }

    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, create an entry this key with a default constructed value.
    template <class TKeyArg>
    auto operator[](TKeyArg&& key) -> TValue&
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto const res = emplace(std::forward<TKeyArg>(key));
        if (res.first == end())
        {
            throw OutOfBoundsException("HashMap: Index not found");
        }
        return res.first->second;
    }
};

/*************************************************************************************************************/
/**
 * function object used in the HashContainer base class
 * to access the key from a HashSet element
 */
template <class TKey>
class HashSetKeyFromElement
{
public:
    /// key is the element itself
    const TKey* operator()(const TKey* const element) const
    {
        return element;
    }
};

/*************************************************************************************************************/
/**
 * HashSet
 */
template <class TKey,
          class THash,
          class TKeyEqual,
          class TStorage>
using BaseHashSet = BaseHashContainer<TKey, TKey, HashSetKeyFromElement<TKey>, THash, TKeyEqual, TStorage>;

} // namespace core
} // namespace dw

#endif // DW_CORE_HASHMAP_HPP_
