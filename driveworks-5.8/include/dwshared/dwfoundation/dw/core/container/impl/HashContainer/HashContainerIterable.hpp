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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_HASHMAPITERABLE_HPP_
#define DW_CORE_HASHMAPITERABLE_HPP_

#include "BaseHashContainer.hpp"
#include "HeapHashContainer.hpp"
#include "StaticHashContainer.hpp"

namespace dw
{
namespace core
{

/*************************************************************************************************************/
/**
 * HashMapIterable and HashSetIterable
 *
 *
 * These are fixed sized containers that allow cache efficient iteration over their elements,
 * while at the same time allowing quick key lookups.
 *
 * It should be avoided to fully fill the container, as performance degrades due to hash collisions.
 *
 * Use these containers instead of HashMap and HashSet,
 * when the container is iterated and/or cleared often.
 * The performance difference is most significant if the number of elements
 * stored in the container is small relative to the reserved capacity.
 *
 * If the focus is on insertion and lookup, it is better to use HashMap and HashSet.
 */

/*************************************************************************************************************/
/**
 * Base class containing shared functionality of HashMapIterable and HashSetIterable.
 * Keeps an array of TElements that can be iterated over, and at the same time
 * stores a HashMap with pointers to the keys and elements for quick access.
 */
template <class TKey,
          class TElement,        //!< type of container elements.
                                 //!  for HashSet this is the same as TKey,
                                 //!  for HashMap this is a std::pair<TKey,TValue>
          class TKeyFromElement, //!< function object class to get the key to an element,
                                 //!  for HashSet this returns the element itself,
                                 //!  for HashMap this returns the key of the key-value pair.
          class THash     = DefaultHash<TKey>,
          class TKeyEqual = std::equal_to<TKey>>
class HashContainerIterable
{
public:
    using key_type   = TKey;     ///< Type of keys
    using value_type = TElement; ///< Type of stored elements

    // container type where the objects are stored
    using TElementContainer = HeapVectorFixed<TElement>; ///< Type of element container

    // iterator for a particular instance
    using iterator       = typename TElementContainer::iterator;       ///< Type of iterator
    using const_iterator = typename TElementContainer::const_iterator; ///< Type of const-iterator

    /// Default constructor
    HashContainerIterable() = default;

    /// Destructor
    ~HashContainerIterable() = default;

    /// Copy operator
    HashContainerIterable& operator=(const HashContainerIterable& other)
    {
        m_entries = other.m_entries;

        // TKey* pointers must be re-generated for new entries
        m_hashMap.clear();

        for (size_t i = 0U; i < m_entries.size(); ++i)
        {
            std::pair<const TKey*, size_t> const newEntry((TKeyFromElement()(&m_entries[i])), i);
            static_cast<void>(m_hashMap.insert(newEntry));
        }

        return *this;
    }

    /// Move constructor
    HashContainerIterable(HashContainerIterable&& other) = default;

    /// Move operator
    HashContainerIterable& operator=(HashContainerIterable&& other) = default;

private:
    /// Construction from map capacity and storage capacity (map should be bigger for better performance)
    HashContainerIterable(size_t const mapCapacity, size_t const entriesCapacity)
        : m_hashMap(mapCapacity)
        , m_entries(entriesCapacity)
    {
    }

public:
    /// Construction from capacity
    explicit HashContainerIterable(size_t const capacity)
        : HashContainerIterable(capacity, capacity)
    {
    }

    /// Construction from initializer lists
    HashContainerIterable(const std::initializer_list<TElement>& ilist)
        : HashContainerIterable(static_cast<size_t>(std::ceil(INIT_LIST_LOAD_FACTOR * static_cast<float32_t>(ilist.size()))), ilist.size())
    {
        for (const auto& entry : ilist)
        {
            if (!this->insert(entry))
            {
                throw InvalidArgumentException("HashMap: Duplicate keys in the initializer list");
            }
        }
    }

    /// Copy constructor
    /// Copies entries, regenerates hash
    HashContainerIterable(const HashContainerIterable& other)
        : m_hashMap(other.m_hashMap.capacity())
        , m_entries(other.m_entries)
    {
        for (size_t i = 0U; i < m_entries.size(); ++i)
        {
            std::pair<const TKey*, size_t> const newEntry((TKeyFromElement()(&m_entries[i])), i);
            static_cast<void>(m_hashMap.insert(newEntry));
        }
    }

    /// Allocate storage
    void allocate(size_t const capacity)
    {
        m_hashMap.allocate(capacity);
        m_entries.allocate(capacity);
    }

    // iterators

    /// Get begin iterator
    auto begin() -> iterator
    {
        return m_entries.begin();
    }

    /// Get begin const iterator
    auto begin() const -> const_iterator
    {
        return m_entries.begin();
    }

    /// Get end iterator
    auto end() -> iterator
    {
        return m_entries.end();
    }

    /// Get end const interator
    auto end() const -> const_iterator
    {
        return m_entries.end();
    }

    /// Returns an iterator to the element with given key, end iterator if none is found.
    auto find(const TKey& key) -> iterator
    {
        iterator result = end();

        auto const internalIt = m_hashMap.find(&key);
        if (internalIt != m_hashMap.end())
        {
            result = m_entries.begin() + internalIt->second;
        }

        return result;
    }

    /// Returns an const iterator to the element with given key, end iterator if none is found.
    auto find(const TKey& key) const -> const_iterator
    {
        const_iterator result = end();

        auto const internalIt = m_hashMap.find(&key);
        if (internalIt != m_hashMap.end())
        {
            result = m_entries.begin() + internalIt->second;
        }

        return result;
    }

    /// Returns true if container contains element with given key.
    bool contains(const TKey& key) const
    {
        return find(key) != end();
    }

    /// insert element if there is no entry yet for the key of this element.
    /// Returns false if element was not inserted. This could happen if:
    /// - Container is full
    /// - No index could be found for the given element
    /// - There is already an entry for the key of the given element
    bool insert(const TElement& newEntry)
    {
        bool inserted = false;
        if (size() < capacity())
        {
            // using address should work, because m_hashMap compares the dereferenced pointers
            auto const it = m_hashMap.find(TKeyFromElement()(&newEntry));

            if (it == m_hashMap.end())
            {
                // not there yet, insert
                static_cast<void>(m_entries.push_back(newEntry));
                std::pair<const TKey*, size_t> const newLookupEntry((TKeyFromElement()(&m_entries.back())),
                                                                    m_entries.size() - 1U);
                inserted = m_hashMap.insert(newLookupEntry);

                if (!inserted)
                {
                    LOGSTREAM_ERROR(nullptr) << "HashMapIteratable: Something went wrong, didn't insert\n";
                    m_entries.pop_back();
                }
            }
        }

        return inserted;
    }

    /// Insert with move semantics
    bool insert(TElement&& newEntry)
    {
        bool inserted = false;
        if (size() < capacity())
        {
            // using address should work, because m_hashMap compares the dereferenced pointers
            auto const it = m_hashMap.find(TKeyFromElement()(&newEntry));

            if (it == m_hashMap.end())
            {
                // not there yet, insert
                static_cast<void>(m_entries.push_back(std::move(newEntry)));
                std::pair<const TKey*, size_t> const newLookupEntry((TKeyFromElement()(&m_entries.back())),
                                                                    m_entries.size() - 1U);
                inserted = m_hashMap.insert(newLookupEntry);

                if (!inserted)
                {
                    LOGSTREAM_ERROR(nullptr) << "HashMapIteratable: Something went wrong, didn't insert\n";
                    m_entries.pop_back();
                }
            }
        }

        return inserted;
    }

    /// Erase element the given iterator refers to.
    /// Returns an iterator to the successor of the erased element in the container.
    // TODO(dwplc): FP: const qualifier already placed on right hand side of a typedef or using name
    // coverity[autosar_cpp14_a7_1_3_violation]
    auto erase(const_iterator const it) -> const iterator
    {
        dw::core::Optional<iterator> nextAfterErase = eraseKey(*TKeyFromElement()(it.operator->()));

        if (!nextAfterErase)
        {
            nextAfterErase = end();
            LOGSTREAM_WARN(this) << "HashContainerIterable: tried to erase with invalid iterator.\n";
        }

        return *nextAfterErase;
    }

    /// Erase element with given key.
    /// Returns true if an element has been erased, false otherwise.
    bool erase(const TKey& key)
    {
        dw::core::Optional<iterator> const nextAfterErase = eraseKey(key);
        bool const erased                                 = nextAfterErase != dw::core::NULLOPT;
        return erased;
    }

    /// Erase element with given key.
    /// Returns an iterator to the successor of the erased element in the container,
    /// if an element has been erased, `dw::core::NULLOPT` otherwise.
    auto eraseKey(const TKey& key) -> dw::core::Optional<iterator>
    {
        // return dw::core::NULLOPT if key is not erased
        dw::core::Optional<iterator> nextAfterErase = dw::core::NULLOPT;

        auto const it = m_hashMap.find(&key);
        if (it != m_hashMap.end())
        {
            const TKey* keyToErase = it->first;
            size_t elementToErase  = it->second;
            static_cast<void>(m_hashMap.erase(keyToErase));

            // m_entries cannot be empty here, because elementToErase points to an element in m_entries
            if (elementToErase != m_entries.size() - 1U)
            {
                // we're erasing some element in the middle of m_entries,
                // so replace erased entry with last
                m_entries[elementToErase] = std::move(m_entries.back());
                m_entries.pop_back();

                // .. and update the stored pointer in the hash_map
                m_hashMap.updateExistingEntry(std::pair<const TKey*, size_t>(keyToErase, elementToErase));

                // next element is now in the same place
                nextAfterErase = m_entries.begin() + elementToErase;
            }
            else
            {
                // element to erase is last entry, so just remove it
                m_entries.pop_back();

                // erased last element, so there is no next
                nextAfterErase = end();
            }
        }

        return nextAfterErase;
    }

    /// Get current number of elements in container
    inline size_t size() const
    {
        return m_hashMap.size();
    }

    /// Get container capacity
    inline size_t capacity() const
    {
        return m_hashMap.capacity();
    }

    /// Clear container, that is destroy and remove all elements.
    void clear()
    {
        size_t const entryCount = m_entries.size();
        if (entryCount < static_cast<size_t>(std::floor(0.2F * static_cast<float32_t>(capacity()))))
        {
            m_hashMap.clear(m_entries, TKeyFromElement());
        }
        else
        {
            // otherwise iterate the whole memory
            m_hashMap.clear();
        }
        m_entries.clear();
    }

    /// Swap with other container
    void swap(HashContainerIterable& other)
    {
        std::swap(m_hashMap, other.m_hashMap);
        std::swap(m_entries, other.m_entries);
    }

protected:
    /**
     * Functor to get the hash value of a key.
     */
    class Hash
    {
    public:
        /// Returns the hash value of a key.
        size_t operator()(const TKey* const key) const
        {
            return THash()(*key);
        }
    };

    /**
     * Functor to compare keys for equality.
     */
    class KeyEqual
    {
    public:
        /// Compares keys for equality
        bool operator()(const TKey* const x, const TKey* const y) const
        {
            return TKeyEqual()(*x, *y);
        }
    };

    /// Get HashMap with Key-Index pairs
    auto hashMap() -> HeapHashMap<const TKey*, size_t, Hash, KeyEqual>&
    {
        return m_hashMap;
    }

    /// Get const HashMap with Key-Index pairs
    auto hashMap() const -> HeapHashMap<const TKey*, size_t, Hash, KeyEqual> const&
    {
        return m_hashMap;
    }

    /// Get element container
    auto entries() -> TElementContainer&
    {
        return m_entries;
    }

    /// Get const element container
    auto entries() const -> TElementContainer const&
    {
        return m_entries;
    }

private:
    /// HashMap that maps from a key to the index of the element in the element containers
    HeapHashMap<const TKey*, size_t, Hash, KeyEqual> m_hashMap;

    /// Element container that can be iterated efficiently
    TElementContainer m_entries;

    /// HashMap capacity factor
    static constexpr float32_t INIT_LIST_LOAD_FACTOR = 1.4F;
};

/*************************************************************************************************************/
/**
 * HashMapIterable
 *
 * HashMap container that is efficient to iterate.
 */
template <class TKey,
          class TValue,
          class THash     = DefaultHash<TKey>,
          class TKeyEqual = std::equal_to<TKey>>
class HashMapIterable : public HashContainerIterable<TKey,
                                                     std::pair<TKey, TValue>,
                                                     HashMapKeyFromElement<TKey, std::pair<TKey, TValue>>,
                                                     THash,
                                                     TKeyEqual>
{
public:
    /// Type of base class
    using Base = HashContainerIterable<TKey,
                                       std::pair<TKey, TValue>,
                                       HashMapKeyFromElement<TKey, std::pair<TKey, TValue>>,
                                       THash,
                                       TKeyEqual>;

    using iterator       = typename Base::iterator;       ///< Type of iterator
    using const_iterator = typename Base::const_iterator; ///< Type of const iterator
    using Base::Base;

    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, create an entry this key with a default constructed value.
    auto operator[](const TKey& key) -> TValue&
    {
        auto const it = this->hashMap().find(&key);
        if (it == this->hashMap().end())
        {
            std::pair<TKey, TValue>& newEntry = this->entries().emplace_back(key, TValue());
            bool const inserted = this->hashMap().insert(std::pair<const TKey*, size_t>(&newEntry.first,
                                                                                        this->entries().size() - 1U));

            if (!inserted)
            {
                throw ExceptionWithStackTrace("HashMapIterable: new entry inserted, but lookup table wasn't able to do an insertion!");
            }

            return newEntry.second;
        }
        else
        {
            return this->entries()[it->second].second;
        }
    }

    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, throw an exception.
    auto at(const TKey& key) -> TValue&
    {
        iterator const iter = this->find(key);
        if (iter == this->end())
        {
            throw OutOfBoundsException("Key does not exist");
        }

        return iter->second;
    }

    /// HashMap specific operator:
    /// Access value for given key.
    /// If key is not in the HashMap, throw an exception.
    auto at(const TKey& key) const -> const TValue&
    {
        const_iterator const iter = this->find(key);
        if (iter == this->end())
        {
            throw OutOfBoundsException("Key does not exist");
        }

        return iter->second;
    }

    /// HashMap specific operator:
    /// Insert element in-place if there is no entry yet for the key of this element.
    /// Returns a pair consisting of an iterator to the inserted element, or the already-existing element
    /// if no insertion happened, and a bool denoting whether the insertion took place.
    /// - Container is full
    /// - No index could be found for the given element
    /// - There is already an entry for the key of the given element
    template <class... Args>
    auto emplace(const TKey& key, Args&&... args) -> std::pair<iterator, bool>
    {
        iterator it   = this->end();
        bool found    = false;
        bool inserted = false;

        if (this->size() < this->capacity())
        {
            auto internalIt = this->hashMap().find(&key);
            found           = internalIt != this->hashMap().end();
            if (!found)
            {
                this->entries().emplace_back(std::piecewise_construct,
                                             std::forward_as_tuple(key),
                                             std::forward_as_tuple(std::forward<Args>(args)...));

                // inserted is last element
                this->hashMap().insert(std::pair<const TKey*, size_t>(&this->entries().back().first,
                                                                      this->entries().size() - 1));

                inserted = true;
            }
            else
            {
                it = this->entries().begin() + internalIt->second;
            }
        }

        return {it, inserted};
    }
};

/*************************************************************************************************************/
/**
 * HashSetIterable
 * 
 * HashSet container that is efficient to iterate.
 */
template <class TKey,
          class THash     = DefaultHash<TKey>,
          class TKeyEqual = std::equal_to<TKey>>
using HashSetIterable     = HashContainerIterable<TKey,
                                              TKey,
                                              HashSetKeyFromElement<TKey>,
                                              THash,
                                              TKeyEqual>;
} // namespace core
} // namespace dw

#endif // DW_CORE_HASHMAPITERABLE_HPP_
