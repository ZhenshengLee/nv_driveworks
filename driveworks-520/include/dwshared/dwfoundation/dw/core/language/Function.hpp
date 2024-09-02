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
// SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * \defgroup function_base_group Function Base Group
 * @{
 */

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

/**
 * @brief Function class template.
 * @tparam Size  Default size of storage for the functor object in bytes.
*/
template <typename F, size_t Size = 32>
class Function;

/**
 * @brief Type-independent part of @c Function class.
 * @tparam Size size of storage for the functor object in bytes.
*/
template <size_t Size>
class FunctionBase
{
public:
    /**
     * @brief Determine if the function wrapper is null.
     *
     * Access specifier: public
     *
     * @retval true if the function object contains a target,
     * @retval false if the function object does not contain a target.
    */
    bool has_function() const
    {
        return m_vtable.call != nullptr;
    }

    /**
     * @brief Overload operator (), returns has_function(). Determine if the function wrapper has a target.
     *
     * Access specifier: public
     *
     * @retval true if this %function object contains a target,
     * @retval false if it is empty.
    */
    explicit operator bool() const
    {
        return has_function();
    }

    /**
     * @brief Overload operator == nullptr_t, returns !has_function(). Compares a polymorphic function object wrapper against 0
     * (the NULL pointer).
     *
     * Access specifier: public
     *
     * @retval true if the wrapper has no target,
     * @retval false if the wrapper has target.
     *
    */
    bool operator==(std::nullptr_t) const
    {
        return !has_function();
    }

    /**
     * @brief Overload operator != nullptr_t, returns has_function(). Compares a polymorphic function object wrapper against 0
     * (the NULL pointer).
     *
     * Access specifier: public
     *
     * @retval false if the wrapper has no target,
     * @retval true if the wrapper has target.
     *
    */
    bool operator!=(std::nullptr_t) const
    {
        return has_function();
    }

protected:
    /**
     * @brief Default constructor. (protected)
     *
     * Access specifier: protected
    */
    FunctionBase() = default;

    /**
     * @brief Copy constructor, calls this.copy(other)
     *
     * Access specifier: protected
     *
     * @param[in] other another reference to a FunctionBase object
    */
    FunctionBase(const FunctionBase& other)
    {
        copy(other);
    }

    /**
     * @brief Move constructor, calls this.move(other)
     *
     * Access specifier: protected
     *
     * @param[in] other another reference to a FunctionBase object
    */
    // coverity[autosar_cpp14_a12_8_4_violation] RFD Pending: TID-2319
    FunctionBase(FunctionBase&& other)
    {
        move(other);
    }

    /**
     * @brief Assignment operator. Assign to this object another const FunctionBase object.
     * Check if "other" reference is not the same of "this",then calls reset() and copy(other).
     *
     * Access specifier: protected 
     *
     * @param[in] other another reference to a FunctionBase object
     * @return Pointer to "this"
    */
    // coverity[autosar_cpp14_a12_8_2_violation] Deviation Record: AV-DriveWorks_Core_Library-dwfoundation_core_language-SWSADR-0001
    FunctionBase& operator=(const FunctionBase& other)
    {
        if (this != &other)
        {
            if (m_vtable.objectLifecycle != nullptr)
            {
                m_vtable.objectLifecycle(ObjectLifecycleOperation::DESTRUCTOR, &m_storage, nullptr, nullptr);
            }
            copy(other);
        }
        return *this;
    }

    /**
     * @brief Move assignment operator. Assign to this object another not-const FunctionBase object.
     * Check if "other" reference is not the same of "this",then calls reset() and copy(other).
     *
     * Access specifier: protected
     *
     * @param[in] other another reference to a FunctionBase object
     * @return Pointer to "this"
    */
    // coverity[autosar_cpp14_a12_8_2_violation] Deviation Record: AV-DriveWorks_Core_Library-dwfoundation_core_language-SWSADR-0001
    FunctionBase& operator=(FunctionBase&& other)
    {
        if (this != &other)
        {
            if (m_vtable.objectLifecycle != nullptr)
            {
                m_vtable.objectLifecycle(ObjectLifecycleOperation::DESTRUCTOR, &m_storage, nullptr, nullptr);
            }
            move(other);
        }
        return *this;
    }

    /**
     * @brief Class destructor, just calls reset()
     *
     * Access specifier: protected
    */
    ~FunctionBase()
    {
        reset();
    }

    /**
     * @brief Reset the underlying function lifecycle (ObjectLifecycleOperation::DESTRUCTOR) and m_vtable.
     *
     * Access specifier: protected
    */
    void reset()
    {
        if (m_vtable.objectLifecycle != nullptr)
        {
            m_vtable.objectLifecycle(ObjectLifecycleOperation::DESTRUCTOR, &m_storage, nullptr, nullptr);
        }
        m_vtable = {};
    }

    /**
     * @brief Change other function lifecycle to ObjectLifecycleOperation::COPY and copy it and m_vtable from the other object.
     *
     * Access specifier: protected
     *
     * @param[in] other Another reference to a FunctionBase object
    */
    void copy(const FunctionBase& other)
    {
        if (other.m_vtable.objectLifecycle != nullptr)
        {
            other.m_vtable.objectLifecycle(ObjectLifecycleOperation::COPY, &m_storage, &other.m_storage, nullptr);
        }
        m_vtable = other.m_vtable;
    }

    /**
     * @brief Change other function lifecycle to ObjectLifecycleOperation::MOVE and copy it and m_vtable from the other object.
     * The other FunctionBase object shall be reset after the move operations.
     *
     * Access specifier: protected
     *
     * @param[in] other Another reference to a FunctionBase object
    */
    void move(FunctionBase& other)
    {
        if (other.m_vtable.objectLifecycle != nullptr)
        {
            other.m_vtable.objectLifecycle(ObjectLifecycleOperation::MOVE, &m_storage, nullptr, &other.m_storage);
        }
        m_vtable = other.m_vtable;
        other.reset();
    }

    /**
     * @brief State of the underlying functor object
    */
    enum class ObjectLifecycleOperation : int32_t
    {
        /// Destructor function
        DESTRUCTOR,
        /// Copy function
        COPY,
        /// Move function
        MOVE,
    };

    /**
     * @brief Type-erased function pointer.
    */
    using TypeErasedFunction = void (*)();
    /**
     * @brief Signature of function in the vtable (operation, This functor object (To), Other functor object (From), Other functor object (mutable From))
    */
    using ObjectLifeCycleFunction = void (*)(ObjectLifecycleOperation, void*, const void*, void*);

    /**
     * @brief Functor virtual table structure
    */
    struct VTable
    {
        /// Type-erased implementation of Function::operator()
        TypeErasedFunction call{nullptr};
        /// Function implementing ctor, dtor, copy for the underlying functor object
        ObjectLifeCycleFunction objectLifecycle{nullptr};
    };

    /// The virtual table used by functor object.
    VTable m_vtable{};
    /// The size for the storage of the functor object.
    std::aligned_storage_t<Size> m_storage{};
};

/**@}*/

/**
 * \defgroup function_group Function Group
 * @{
 */

/**
 * @brief A replacement class for std::function that can store, copy, and invoke any
 *        Callable target -- functions, lambda expressions, bind expressions, or other
 *        function objects.
 * 
 *        It has the following advantages
 *              - Unlike std::function this function does not allocate any memory on the heap.  By
 *                default Function allocates a 32 byte buffer to hold any memory needed
*/
template <typename Ret, typename... Args, size_t Size>
class Function<Ret(Args...), Size> : public FunctionBase<Size>
{
private:
    using Base = FunctionBase<Size>;

public:
    /**
     * @brief Default constuctor
     *
     * Access specifier: public
    */
    Function() = default; // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)

    /**
     * @brief nullptr_t constructor (empty)
     * 
     * Access specifier: public
    */
    Function(std::nullptr_t) // clang-tidy NOLINT(google-explicit-constructor)
    {
    }

    /**
     * @brief Functor constructor. Call this.create() passing the functor.
     *
     * Access specifier: public
     *
     * @param[in] f The Functor used to create the Function object.
    */
    template <typename Functor, std::enable_if_t<!std::is_same<std::remove_cv_t<Functor>, Function>::value>* = nullptr>
    Function(Functor&& f) // clang-tidy NOLINT
    {
        create(std::forward<Functor>(f));
    }

    /**
     * @brief Default copy constructors.
     * Need to declare them explicitly to avoid a conflict with the ctor from a functor.
     *
     * Access specifier: public
    */
    Function(const Function&) = default;
    /**
     * @brief Default copy constructors.
     * Need to declare them explicitly to avoid a conflict with the ctor from a functor.
     *
     * Access specifier: public
    */
    Function(Function&) = default;
    /**
     * @brief Default move constructors.
     * Need to declare them explicitly to avoid a conflict with the ctor from a functor.
     *
     * Access specifier: public
    */
    Function(Function&&) = default;

    /**
     * @brief Default copy assignment operators.
     * Need to declare them explicitly to avoid a conflict with the assignment from a functor.
     *
     * Access specifier: public
    */
    Function& operator=(const Function&) = default;
    /**
     * @brief Default copy assignment operators.
     * Need to declare them explicitly to avoid a conflict with the assignment from a functor.
     *
     * Access specifier: public
    */
    Function& operator=(Function&) = default;
    /**
     * @brief Default move assignment operators.
     * Need to declare them explicitly to avoid a conflict with the assignment from a functor.
     *
     * Access specifier: public
    */
    Function& operator=(Function&&) = default;

    /**
     * @brief Default destructor.
     *
     * Access specifier: public
    */
    ~Function() = default;

    /**
     * @brief Nullptr assignment operator, calls FunctionBase<Size>::reset().
     *
     * Access specifier: public
     *
     * @return Pointer to this.
    */
    Function& operator=(std::nullptr_t)
    {
        Base::reset();
        return *this;
    }

    /**
     * @brief Assignment operator, calls FunctionBase<Size>::reset() and this.create() passing the functor.
     *
     * Access specifier: public
     *
     * @param[in] f The Functor used to create the Function object.
     * @return Pointer to this.
    */
    template <typename Functor>
    Function& operator=(Functor&& f)
    {
        Base::reset();
        create(std::forward<Functor>(f));

        return *this;
    }

    /**
     * @brief Overload of () operator. Calls m_vtable.call() passing the provided arguments.
     *
     * @throw ExceptionWithStackTrace if m_vtable.call is null.
     *
     * Access specifier: public
    */
    inline Ret operator()(Args... args) const
    {
        if (this->m_vtable.call == nullptr)
        {
            throw ExceptionWithStackTrace("Function: Tried to call an empty function");
        }
        // It's OK to reinterpret_cast m_vtable.call. The opposite cast is done in 'createImpl()'.
        if (this->m_vtable.objectLifecycle != nullptr)
        {
            // coverity[autosar_cpp14_m5_2_6_violation] RFD Pending: TID-2324
            // coverity[cert_exp39_c_violation] RFD Pending: TID-2462
            return reinterpret_cast<FunctorFunction>(this->m_vtable.call)(&this->m_storage, std::forward<Args>(args)...); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }
        else
        {
            // coverity[autosar_cpp14_m5_2_6_violation] RFD Pending: TID-2324
            // coverity[cert_exp39_c_violation] RFD Pending: TID-2462
            return reinterpret_cast<StaticFunction>(this->m_vtable.call)(std::forward<Args>(args)...); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }
    }

private:
    /// Avoid extra reference indirection for cheap-to-copy types.
    template <typename Type>
    using ForwardByValueOrRef = std::conditional_t<std::is_scalar<Type>::value, Type, Type&&>;

    /// FunctorFunction type definition
    using FunctorFunction = Ret (*)(const void*, ForwardByValueOrRef<Args>...);
    /// FunctorFunction type definition
    using StaticFunction = Ret (*)(Args...);
    /// ObjectLifecycleOperation type definition
    using ObjectLifecycleOperation = typename Base::ObjectLifecycleOperation;
    /// TypeErasedFunction type definition
    using TypeErasedFunction = typename Base::TypeErasedFunction;

    /**
     * @brief Assign m_vtable.call and m_vtable.objectLifeCycle from the passed functor.
     * This function is called by this.create() if passed functor is convertible to StaticFunction.
     *
     * Access specifier: private
     *
     * @param[in] func The functor to pass.
    */
    template <typename Functor>
    void createImpl(const Functor& func, /* IsStaticFunctionConvertible= */ std::true_type)
    {
        // It's OK to reinterpret_cast the function ptr here. The opposite cast is done in 'operator()'.
        // coverity[autosar_cpp14_m5_2_6_violation] RFD Pending: TID-2324
        this->m_vtable.call            = reinterpret_cast<TypeErasedFunction>(static_cast<StaticFunction>(func)); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        this->m_vtable.objectLifecycle = nullptr;
    }

    /**
     * @brief Assign m_vtable.call and m_vtable.objectLifeCycle from the passed functor.
     * This function is called by this.create() if passed functor is not convertible to StaticFunction.
     *
     * Access specifier: private
     *
     * @param[in] func The functor to pass.
    */
    template <typename Functor>
    void createImpl(Functor&& func, /* IsStaticFunctionConvertible= */ std::false_type)
    {
        // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
        using FunctorType = std::decay_t<Functor>;
        dw::core::checkClassSizeLessThan<FunctorType, Size>();

        // It's OK to reinterpret_cast the function ptr here. The opposite cast is done in 'operator()'.
        // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
        // coverity[autosar_cpp14_m5_2_6_violation] RFD Pending: TID-2324
        this->m_vtable.call            = reinterpret_cast<TypeErasedFunction>(&callImplementation<FunctorType>); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        this->m_vtable.objectLifecycle = &lifeCycleImplementation<FunctorType>;
        // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
        new (&this->m_storage) FunctorType(std::forward<Functor>(func));
    }

    /**
     * @brief Populate the Function object with information from the passed functor.
     * Calls one among two different versions of createImpl(), depending on the result of the check std::is_convertible<std::decay_t<Functor>, StaticFunction>
     *
     * Access specifier: private
     *
     * @param[in] f The functor to pass.
    */
    template <typename Functor>
    void create(Functor&& f)
    {
        // It looks like a bug in clang-tidy, it reports "prefer transparent functors 'greater<>'".
        // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
        using IsStaticFunctionConvertible = // clang-tidy NOLINTNEXTLINE(modernize-use-transparent-functors)
            std::is_convertible<std::decay_t<Functor>, StaticFunction>;
        createImpl(std::forward<Functor>(f), IsStaticFunctionConvertible{});
    }

    /**
     * @brief Returns the correct function, depending on the passed operation. Called by createImpl() function.
     *
     * Access specifier: private
     *
     * @param[in] operation The object life cycle operation to execute.
     * @param[in] thisObj The pointer to this object.
     * @param[in] otherObjConst The pointer to another const object for copy.
     * @param[in] otherObjMutable The pointer to another mutable object for move.
     *
     * @throw ExceptionWithStackTrace if @b thisObj , @b otherObjConst or @b otherObjMutable is null
    */
    template <typename Functor>
    static void lifeCycleImplementation(ObjectLifecycleOperation operation, void* thisObj, const void* otherObjConst, void* otherObjMutable)
    {
        switch (operation)
        {
        case ObjectLifecycleOperation::DESTRUCTOR:
            if (!thisObj)
            {
                throw ExceptionWithStackTrace("Function: nullptr argument");
            }
            // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            static_cast<Functor*>(thisObj)->~Functor();
            break;
        case ObjectLifecycleOperation::COPY:
            if (!thisObj || !otherObjConst)
            {
                throw ExceptionWithStackTrace("Function: nullptr argument");
            }
            // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
            new (thisObj) Functor(*static_cast<const Functor*>(otherObjConst));
            break;
        case ObjectLifecycleOperation::MOVE:
            if (!thisObj || !otherObjMutable)
            {
                throw ExceptionWithStackTrace("Function: nullptr argument");
            }
            // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
            new (thisObj) Functor(std::move(*static_cast<Functor*>(otherObjMutable)));
            break;
        default:
            throw ExceptionWithStackTrace("Function: Unexpected lifecycle operation");
        }
    }

    /**
     * @brief Returns the type-erased implementation of functor. Called by createImpl() function.
    */
    template <typename Functor>
    static Ret callImplementation(const void* const functor, ForwardByValueOrRef<Args>... args)
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        // coverity[autosar_cpp14_a5_1_7_violation] FP: nvbugs/4299504
        auto& f = *static_cast<const Functor*>(functor);
        return f(std::forward<Args>(args)...);
    }
};

/**@}*/

} // core
} // dw

#endif //DW_CORE_FUNCTION_HPP_
