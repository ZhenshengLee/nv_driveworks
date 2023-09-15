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
// SPDX-FileCopyrightText: Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/language/CheckClassSize.hpp>

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

/// @brief Type-independent part of @c Function.
/// @tparam Size size of storage for the functor object in bytes.
template <size_t Size>
class FunctionBase
{
public:
    bool has_function() const
    {
        return m_vtable.call != nullptr;
    }

    explicit operator bool() const
    {
        return has_function();
    }

    bool operator==(std::nullptr_t) const
    {
        return !has_function();
    }

    bool operator!=(std::nullptr_t) const
    {
        return has_function();
    }

protected:
    FunctionBase() = default;

    FunctionBase(const FunctionBase& other)
    {
        copy(other);
    }

    FunctionBase(FunctionBase&& other)
    {
        move(other);
    }

    FunctionBase& operator=(const FunctionBase& other)
    {
        if (this != &other)
        {
            reset();
            copy(other);
        }
        return *this;
    }

    FunctionBase& operator=(FunctionBase&& other)
    {
        if (this != &other)
        {
            reset();
            move(other);
        }
        return *this;
    }

    ~FunctionBase()
    {
        reset();
    }

    void reset()
    {
        if (m_vtable.objectLifecycle != nullptr)
        {
            m_vtable.objectLifecycle(ObjectLifecycleOperation::DESTRUCTOR, &m_storage, nullptr);
        }
        m_vtable = {};
    }

    void copy(const FunctionBase& other)
    {
        if (other.m_vtable.objectLifecycle != nullptr)
        {
            other.m_vtable.objectLifecycle(ObjectLifecycleOperation::COPY, &m_storage, &other.m_storage);
        }
        m_vtable = other.m_vtable;
    }

    void move(FunctionBase& other)
    {
        if (other.m_vtable.objectLifecycle != nullptr)
        {
            // Cast other from mutable to const ref. The opposite conversion is done in 'lifeCycleImplementation()'.
            // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            auto* other_storage = const_cast<const void*>(static_cast<void*>(&other.m_storage));
            other.m_vtable.objectLifecycle(ObjectLifecycleOperation::MOVE, &m_storage, other_storage);
        }
        m_vtable = other.m_vtable;
        other.reset();
    }

    enum class ObjectLifecycleOperation : int32_t
    {
        DESTRUCTOR,
        COPY,
        MOVE,
    };

    using TypeErasedFunction = void (*)();
    // Signature (operation, This functor object (To), Other functor object (From))
    using ObjectLifeCycleFunction = void (*)(ObjectLifecycleOperation, void*, const void*);

    struct VTable
    {
        /// Type-erased implementation of Function::operator()
        TypeErasedFunction call{nullptr};
        /// Function implementing ctor, dtor, copy for the underlying functor object
        ObjectLifeCycleFunction objectLifecycle{nullptr};
    };

    VTable m_vtable{};
    std::aligned_storage_t<Size> m_storage{};
};

template <typename Ret, typename... Args, size_t Size>
class Function<Ret(Args...), Size> : public FunctionBase<Size>
{
private:
    using Base = FunctionBase<Size>;

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

    // Need to declare them explicitly to avoid a conflict with the ctor from a functor
    Function(const Function&) = default;
    Function(Function&)       = default;
    Function(Function&&)      = default;

    // Need to declare them explicitly to avoid a conflict with the assignment from a functor
    Function& operator=(const Function&) = default;
    Function& operator=(Function&) = default;
    Function& operator=(Function&&) = default;

    ~Function() = default;

    Function& operator=(std::nullptr_t)
    {
        Base::reset();
        return *this;
    }

    template <typename Functor>
    Function& operator=(Functor&& f)
    {
        Base::reset();
        create(std::forward<Functor>(f));

        return *this;
    }

    inline Ret operator()(Args... args) const
    {
        if (this->m_vtable.call == nullptr)
        {
            throw ExceptionWithStackTrace("Function: Tried to call an empty function");
        }
        // It's OK to reinterpret_cast m_vtable.call. The opposite cast is done in 'createImpl()'.
        if (this->m_vtable.objectLifecycle != nullptr)
        {
            // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return reinterpret_cast<FunctorFunction>(this->m_vtable.call)(&this->m_storage, std::forward<Args>(args)...);
        }
        else
        {
            // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return reinterpret_cast<StaticFunction>(this->m_vtable.call)(std::forward<Args>(args)...);
        }
    }

private:
    // Avoid extra reference indirection for cheap-to-copy types.
    template <typename Type>
    using ForwardByValueOrRef = std::conditional_t<std::is_scalar<Type>::value, Type, Type&&>;

    using FunctorFunction          = Ret (*)(const void*, ForwardByValueOrRef<Args>...);
    using StaticFunction           = Ret (*)(Args...);
    using ObjectLifecycleOperation = typename Base::ObjectLifecycleOperation;
    using TypeErasedFunction       = typename Base::TypeErasedFunction;

    template <typename Functor>
    void createImpl(const Functor& func, /* IsStaticFunctionConvertible= */ std::true_type)
    {
        // It's OK to reinterpret_cast the function ptr here. The opposite cast is done in 'operator()'.
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        this->m_vtable.call            = reinterpret_cast<TypeErasedFunction>(static_cast<StaticFunction>(func));
        this->m_vtable.objectLifecycle = nullptr;
    }

    template <typename Functor>
    void createImpl(Functor&& func, /* IsStaticFunctionConvertible= */ std::false_type)
    {
        using FunctorType = std::decay_t<Functor>;
        dw::core::checkClassSizeLessThan<FunctorType, Size>();

        // It's OK to reinterpret_cast the function ptr here. The opposite cast is done in 'operator()'.
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        this->m_vtable.call            = reinterpret_cast<TypeErasedFunction>(&callImplementation<FunctorType>);
        this->m_vtable.objectLifecycle = &lifeCycleImplementation<FunctorType>;
        new (&this->m_storage) FunctorType(std::forward<Functor>(func));
    }

    template <typename Functor>
    void create(Functor&& f)
    {
        // It looks like a bug in clang-tidy, it reports "prefer transparent functors 'greater<>'".
        // clang-tidy NOLINTNEXTLINE(modernize-use-transparent-functors)
        using IsStaticFunctionConvertible = std::is_convertible<std::decay_t<Functor>, StaticFunction>;
        createImpl(std::forward<Functor>(f), IsStaticFunctionConvertible{});
    }

    template <typename Functor>
    static void lifeCycleImplementation(ObjectLifecycleOperation operation, void* thisObj, const void* otherObj)
    {
        switch (operation)
        {
        case ObjectLifecycleOperation::DESTRUCTOR:
            static_cast<Functor*>(thisObj)->~Functor();
            break;
        case ObjectLifecycleOperation::COPY:
            new (thisObj) Functor(*static_cast<const Functor*>(otherObj));
            break;
        case ObjectLifecycleOperation::MOVE:
            // It's OK to do const to mutable ref cast here. The opposite conversion is done in 'move()' method.
            // Otherwise one has to use one more function argument, what is an unneeded overhead.
            // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            new (thisObj) Functor(std::move(*static_cast<Functor*>(const_cast<void*>(otherObj))));
            break;
        default:
            throw ExceptionWithStackTrace("Function: Unexpected lifecycle operation");
            break;
        }
    }

    template <typename Functor>
    static Ret callImplementation(const void* const functor, ForwardByValueOrRef<Args>... args)
    {
        auto& f = *static_cast<const Functor*>(functor);
        return f(std::forward<Args>(args)...);
    }
};
}
}

#endif //DW_CORE_FUNCTION_HPP_
