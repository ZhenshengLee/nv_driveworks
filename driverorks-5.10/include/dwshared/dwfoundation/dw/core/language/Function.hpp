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
// SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_CORE_FUNCTION_HPP_
#define DWSHARED_CORE_FUNCTION_HPP_

#include <type_traits>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/language/CheckClassSize.hpp>

namespace dw
{
namespace core
{

/// A replacement class for std::function that can store, copy, and invoke any
/// Callable target -- functions, lambda expressions, bind expressions, or other
/// function objects.
///
/// It has the following advantages
///     - Unlike std::function this function does not allocate any memory on the heap.  By
///       default Function allocates a 32 byte buffer to hold any memory needed
///
/// Note:
///     It is the responsibility of the developer to ensure that any memory captured by a
///     Function object remains valid for the lifetime of that Function object.

//
// Usage examples:
//     int32_t myFancyFunction(int32_t fancy);
//     dw::core::Function<int32_t(int32_t)> f1 = myFancyFunction;
//     auto t = f1(10);
//
//     class X
//     {
//         float32_t myFancyFunction(float32_t f, float32_t g) {return f * g;}
//         int32_t operator()(int32_t x) {return x + 2;}
//     };
//
//     X x;
//     dw::core::Function<int32_t(int32_t)> f2 = x;
//     auto i = f2(14);
//
//     dw::core::Function<float32_t(float32_t,float32_t)> f3 = std::bind(&X::myFancyFunction,&x,std::placeholders::_1);
//     float32_t f = f3(10.0f,15.0f);
//
//     int32_t x = 5, y = 10, z = 15, w = 20;
//     dw::core::Function<void(int32_t)> f4 = [&x](int32_t y){ x += y;};
//     f4(3);
//
//     dw::core::Function<void(int32_t),40> f5 = [&x,&y,&z,&w](int32_t mul)
//     {
//         x *= mul;
//         y *= mul;
//         z *= mul;
//         w *= mul;
//     };
//     f5(2);

// TODO(dwplc): RFD - Required for template specialization below. This generic version is intentionally undefined, non-specialized instances shall result in a "undefined" compile error.
// coverity[autosar_cpp14_m3_2_3_violation]
template <typename F, size_t Size = 32>
class Function;

template <typename Ret, typename... Args, size_t Size>
class Function<Ret(Args...), Size>
{
public:
    Function() = default; // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)

    Function(std::nullptr_t) // clang-tidy NOLINT(google-explicit-constructor)
    {
    }

    template <typename Functor>
    Function(Functor&& f) // clang-tidy NOLINT
    {
        create(std::forward<Functor>(f));
    }

    ~Function()
    {
        reset();
    }

    Function(const Function& other)
    {
        copy(other);
    }

    Function(Function& other)
    {
        copy(other);
    }

    Function(Function&& other)
    {
        move(std::move(other));
    }

    bool operator==(std::nullptr_t)
    {
        return *this == Function(nullptr);
    }

    bool operator!=(std::nullptr_t)
    {
        return *this != Function(nullptr);
    }

    Function& operator=(std::nullptr_t)
    {
        reset();
        return *this;
    }

    Function& operator=(const Function& other)
    {
        if (this != &other)
        {
            reset();
            copy(other);
        }

        return *this;
    }

    Function& operator=(Function& other)
    {
        if (this != &other)
        {
            reset();
            copy(other);
        }

        return *this;
    }

    Function& operator=(Function&& other)
    {
        if (this != &other)
        {
            reset();
            move(std::move(other));
        }

        return *this;
    }

    template <typename Functor>
    Function& operator=(Functor&& f)
    {
        reset();
        create(std::forward<Functor>(f));

        return *this;
    }

    operator bool() const { return m_vtable.call != nullptr; } // clang-tidy NOLINT

    inline auto operator()(Args... args) const -> Ret
    {
        if (m_vtable.call != nullptr)
        {
            return m_vtable.call(&m_storage, std::forward<Args>(args)...);
        }
        else
        {
            throw ExceptionWithStackTrace("Function: Tried to call an empty function");
        }
    }

private:
    template <typename Functor>
    void create(Functor&& f)
    {
        using functorType = typename std::decay<Functor>::type;
        dw::core::checkClassSizeLessThan<functorType, Size>();

        new (&m_storage) functorType(std::forward<Functor>(f));

        m_vtable.call     = &callImplementation<functorType const>;
        m_vtable.destruct = &destructImplementation<functorType>;
        m_vtable.copy     = &copyImplementation<functorType>;
        m_vtable.move     = &moveImplementation<functorType>;
    }

    void reset()
    {
        auto destruct = m_vtable.destruct;
        if (destruct != nullptr)
        {
            m_vtable = VTable();
            destruct(&m_storage);
        }
    }

    void copy(const Function& other)
    {
        if (other.m_vtable.copy != nullptr)
        {
            other.m_vtable.copy(&other.m_storage, &m_storage);
            m_vtable = other.m_vtable;
        }
    }

    void move(Function&& other)
    {
        if (other.m_vtable.move != nullptr)
        {
            other.m_vtable.move(&other.m_storage, &m_storage);
            m_vtable = other.m_vtable;
            other.reset();
        }
    }

    template <typename Functor>
    static auto callImplementation(const void* const functor, Args&&... args) -> Ret
    {
        auto& f = *static_cast<const Functor*>(functor);
        return f(std::forward<Args>(args)...);
    }

    template <typename Functor>
    static void destructImplementation(void* const functor)
    {
        static_cast<Functor*>(functor)->~Functor();
    }

    template <typename Functor>
    static void copyImplementation(void const* const src, void* const dest)
    {
        new (dest) Functor(*static_cast<Functor const*>(src));
    }

    template <typename Functor>
    static void moveImplementation(void* const src, void* const dest)
    {
        new (dest) Functor(std::move(*static_cast<Functor*>(src)));
    }

    struct VTable
    {
        Ret (*call)(const void*, Args&&...) = nullptr;
        void (*destruct)(void*) = nullptr;
        void (*copy)(const void*, void*) = nullptr;
        void (*move)(void*, void*)       = nullptr;
    };

    using Storage = typename std::aligned_storage<Size>::type;

    VTable m_vtable;
    Storage m_storage{};
};
}
}

#endif //DW_CORE_FUNCTION_HPP_
