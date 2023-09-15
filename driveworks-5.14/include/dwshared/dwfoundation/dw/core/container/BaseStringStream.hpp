/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2019-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DWSHARED_CORE_BASESTRINGSTREAM_HPP_
#define DWSHARED_CORE_BASESTRINGSTREAM_HPP_

#include <iostream>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

namespace dw
{
namespace core
{

/// Input/Output streambuf using an existing BaseString as the storage model
/**
 * To construct an 'ostream' with a 'BaseString' as character storage, use the
 * following pattern:
 * @code
 * BaseString<char8_t, 512> myStorage;
 * BaseStringStreamBuf<512, char8_t> strmBuf(&myStorage);
 * std::ostream ostrm{&strmBuf};
 * ostrm << "Hello world!";
 * @endcode
 *
 * At which point 'myStorage' contains the null terminated string
 * "Hello world!".
 *
 * NOTE: BufferSize template parameter is only required because we store a
 * pointer to the BaseString providing storage and there is no un-templated
 * base class providing the interface we need.
 */
template <size_t BufferSize, typename CharT,
          class Traits = std::char_traits<CharT>>
class BaseStringStreamBuf : public std::basic_streambuf<CharT, Traits>
{
public:
    /// Construction from string storage pointer
    explicit BaseStringStreamBuf(BaseString<BufferSize, CharT>* storage)
        : m_storage(storage)
    {
        this->reset();
    }

    /// Destructor
    ~BaseStringStreamBuf() override = default;

    /// Reset
    void reset()
    {
        m_storage->resize(0);
        this->setp(m_storage->data(), m_storage->data() + BufferSize - 1);
        this->setg(m_storage->data(), m_storage->data(), m_storage->data());
    }

protected:
    /// illegal operation, throws an exception
    BaseStringStreamBuf<BufferSize, CharT, Traits>*
    setbuf(CharT*, std::streamsize) override
    {
        throw std::runtime_error("Illegal BaseStringStreamBuf::setbuf()");
    }

    /// override of std::basic_streambuf<CharT, Traits>::xsputn
    std::streamsize xsputn(const CharT* s, std::streamsize count) override
    {
        // Update the string size according to the number of characters we
        // are going to write (also writes the null terminator). Note that
        // we must do this *before* writing to the buffer because resize()
        // modifies the content between the current size and the new size.
        // Note further that if 'size() + count > capacity' then the resize()
        // function here will silently truncate, and 'BaseT::xsputn() will'
        // will 'put()' a subset of 'count' matching the truncated size.
        m_storage->resize(m_storage->size() + static_cast<uint32_t>(count));

        using BaseT            = std::basic_streambuf<CharT, Traits>;
        std::streamsize retval = this->BaseT::xsputn(s, count);

        // Update the end-pointer for the read stream, now that potentially new
        // content has been added to the storage model.
        this->setg(m_storage->data(), this->gptr(), (m_storage->data() + m_storage->size()));
        return retval;
    }

    /// string storage
    BaseString<BufferSize, CharT>* m_storage;
};

/// Provide storage for stringstreams
/**
 * This class only exists because BaseStringStream needs to construct it's
 * storage and streambuf before using them to construct the ostream interface.
 * BaseStringStream inherits from this class first ensuring the proper order of
 * construction/destruction.
 */
template <size_t BufferSize, typename CharT,
          class Traits = std::char_traits<CharT>>
class BaseStringStreamBase
{
protected:
    BaseStringStreamBase()
        : m_streambuf(&m_storage)
    {
    }

    virtual ~BaseStringStreamBase() = default;

public:
    /// Return a const reference to the underlying string storage.
    const BaseString<BufferSize, CharT>& str() const
    {
        return m_storage;
    }

protected:
    BaseString<BufferSize, CharT> m_storage;                    ///< character storage
    BaseStringStreamBuf<BufferSize, CharT, Traits> m_streambuf; ///< stream buffer
};

/// Input/Output stream backed by a BaseString storage model
/**
 * Works like std::stringstream. For example:
 * @code
 * BaseStringStream<100> stream;
 * stream << "hello" << " world" << "!\n";
 * BaseString<13> str = stream.str();  // contains "Hello world!\n"
 * @endcode
 */
template <size_t BufferSize, typename CharT,
          class Traits = std::char_traits<CharT>>
class BaseStringStream : public BaseStringStreamBase<BufferSize, CharT, Traits>,
                         public std::basic_iostream<CharT, Traits>
{
public:
    BaseStringStream()
        : std::basic_iostream<CharT, Traits>(&this->m_streambuf)
    {
    }

    ~BaseStringStream() override = default;

    /// Reset stream and clear.
    void reset()
    {
        this->m_streambuf.reset();
        this->clear();
    }
};

template <size_t N>
using FixedStringStream = BaseStringStream<N, char8_t>;

} // namespace core
} // namespace dw

#endif // DW_CORE_BASESTRINGSTREAM_HPP_
