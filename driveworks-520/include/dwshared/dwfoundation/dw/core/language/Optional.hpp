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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CUDA_OPTIONAL_HPP_
#define DW_CUDA_OPTIONAL_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>
#include <dwshared/dwfoundation/dw/core/ConfigChecks.h>

#include "cxx14.hpp"

namespace dw
{
namespace core
{

/** \defgroup nullopt_struct NullOpt Structure
 *  @{
 */

/// Disengaged state indicator type and value (analog to 'std::nullopt_t' / 'std::nullopt')
struct NullOpt
{
    struct Init
    {
        constexpr explicit Init() = default;
    };
    constexpr explicit NullOpt(Init){};
};

/// Definition of null optional
constexpr NullOpt NULLOPT{NullOpt::Init{}};

/** @}*/

/**
 * @brief This class replaces std::optional because 'std::optional' doesn't support CUDA.
 *        The class template BaseOptional manages an optional contained value, i.e. a value that may or may not be present.
 *        Any instance of BaseOptional<T> at any given point in time either contains a value or does not contain a value.
*/
// If we ever switch to C++17 in CUDA and it supports std::optional, this could be removed.
// TODO(danielh): replace all uses of std::optional with this type and remove 3rdparty optional lib.
template <class T, template <class> class Derived>
class BaseOptional
{
public:
    using value_type = T;

    /**
     * @brief Create a disengaged optional.
     *
     * Access specifier: public
    */
    constexpr BaseOptional() = default; // clang-tidy NOLINT(cppcoreguidelines-pro-type-member-init)

    /**
     * @brief Create a disengaged optional. This is equivalent to the default constructor.
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE constexpr BaseOptional(NullOpt) noexcept // clang-tidy NOLINT(google-explicit-constructor)
        : BaseOptional()
    {
    }

    /**
     * @brief Copy constructor.
     *
     * Access specifier: public
     *
     * @param[in] other Another reference to a BaseOptional object
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    // coverity[autosar_cpp14_a12_8_1_violation] RFD Pending: TID-2319
    CUDA_BOTH_INLINE BaseOptional(const BaseOptional& other)
    {
        if (other.has_value())
        {
            m_isValid = other.m_isValid;
            new (&m_valueBuffer) T(*other);
        }
    }

    /**
     * @brief Move constructor.
     *
     * Access specifier: public
     *
     * @param[in] other Another reference to a BaseOptional object
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    // coverity[autosar_cpp14_a12_8_4_violation] RFD Pending: TID-2319
    // coverity[autosar_cpp14_a12_8_1_violation] RFD Pending: TID-2319
    CUDA_BOTH_INLINE BaseOptional(BaseOptional&& other)
    {
        if (other.has_value())
        {
            m_isValid = other.m_isValid;
            new (&m_valueBuffer) T(std::move(*other));
        }
    }

    /**
     * @brief Constructor for BaseOptional.
     * It initializes @b valueIn to the buffer and sets m_isValid to true.
     *
     * Access specifier: public
     *
     * @param[in] valueIn The value to pass
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE BaseOptional(const T& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
    {
        new (&m_valueBuffer) T(valueIn);
        m_isValid = true;
    }

    /**
     * @brief Move constructor for BaseOptional.
     * It sets @b valueIn to the buffer and sets m_isValid to true.
     *
     * Access specifier: public
     *
     * @param[in] valueIn The value to pass
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE BaseOptional(T&& valueIn) // clang-tidy NOLINT(google-explicit-constructor)
    {
        new (&m_valueBuffer) T(std::move(valueIn));
        m_isValid = true;
    }

    /**
     * @brief Determine if the object is valid. Operator () overload. Returns m_isValid.
     *
     * Access specifier: public
     *
     * @return true if the object is valid,
     *         false otherwises.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE explicit operator bool() const { return m_isValid; }
    /**
     * @brief Determine if the object has the value.
     *
     * Access specifier: public
     *
     * @return true if the object has the value,
     *         false otherwises.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE bool has_value() const { return m_isValid; }

    /**
     * @brief Operator -> overload. Call assetValueDebug() and return &bufferAsValue().
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator-> () -> T*
    {
        assertValueDebug();
        return &bufferAsValue();
    }
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator-> () const -> const T*
    {
        assertValueDebug();
        return &bufferAsValue();
    }

    /**
     * @brief Operator * overload. Just return bufferAsValue().
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator*() -> T&
    {
        return bufferAsValue();
    }
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator*() const -> const T&
    {
        return bufferAsValue();
    }

    /**
     * @brief Return the value of the object.
     * Call assertValue() and returns bufferAsValue().
     *
     * Access specifier: public
     *
     * @return The value of the object.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto value() -> T&
    {
        assertValue();
        return bufferAsValue();
    }
    /**
     * @brief Return the const value of the object.
     * Call assertValue() and returns bufferAsValue().
     *
     * Access specifier: public
     *
     * @return The const value of the object.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto value() const -> const T&
    {
        assertValue();
        return bufferAsValue();
    }

    /**
     * @brief If m_isValid is true return bufferAsValue(), otherwise return default_value.
     *
     * Access specifier: public
    */
    template <class U>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto value_or(U&& default_value) const -> T
    {
        if (m_isValid)
        {
            return bufferAsValue();
        }
        else
        {
            return static_cast<T>(std::forward<U>(default_value));
        }
    }

    /**
     * @brief Return the pointer to the object. If the object is invalid, return nullptr.
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto ptr() -> T* { return m_isValid ? &bufferAsValue() : nullptr; }
    /**
     * @brief Return the const pointer to the object. If the object is invalid, return nullptr.
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto ptr() const -> T const* { return m_isValid ? &bufferAsValue() : nullptr; }

    /**
     * @brief Destroy any previously existing value and create a new value from args.
     *
     * Access specifier: public
    */
    template <typename... Args>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto emplace(Args&&... args) -> BaseOptional&
    {
        reset(); // destroy any previously existing value

        new (&m_valueBuffer) T{std::forward<Args>(args)...};
        m_isValid = true;

        return *this;
    }

    /**
     * @brief Copy assignment using null optional.
     * Just call reset() and return pointer to this object.
     *
     * Access specifier: public
     *
     * @return *this
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator=(NullOpt) -> BaseOptional&
    {
        reset();
        return *this;
    }

    /**
     * @brief Move assignment using a valid BaseOptional object.
     *
     * Access specifier: public
     * 
     * @param[in] other Another BaseOptional object
     *
     * @return *this
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator=(BaseOptional&& other) -> BaseOptional&
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

    /**
     * @brief Copy assignment using a valid BaseOptional.
     *
     * Access specifier: public
     * 
     * @param[in] other Another const BaseOptional object
     *
     * @return *this
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator=(const BaseOptional& other) -> BaseOptional&
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

    template <class U>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator=(U&& other)
        -> typename std::enable_if<
            std::is_same<U, T>::value,
            BaseOptional&>::type
    {
        if (m_isValid)
        {
            bufferAsValue() = std::move(other); // clang-tidy NOLINT(bugprone-move-forwarding-reference) Ok to silence because U is always equal to T
        }
        else
        {
            new (&m_valueBuffer) T(std::move(other)); // clang-tidy NOLINT(bugprone-move-forwarding-reference) Ok to silence because U is always equal to T
            m_isValid = true;
        }

        return *this;
    }

    /**
     * @brief Assignment operator overload, used to assign a value to the optional variable.
     *
     * Access specifier: public
     *
     * @param[in] other Another const BaseOptional object
     *
     * @return *this
    */
    template <class U>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto operator=(const U& other)
        -> typename std::enable_if<
            std::is_same<U, T>::value,
            BaseOptional&>::type
    {
        if (m_isValid)
        {
            bufferAsValue() = other;
        }
        else
        {
            new (&m_valueBuffer) T(other);
            m_isValid = true;
        }
        return *this;
    }

    /**
     * @brief Reset the object.
     * If m_isValid is true, call bufferAsValue().~T() and set m_Valid to false.
     *
     * Access specifier: public
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void reset()
    {
        if (m_isValid)
        {
            // coverity[autosar_cpp14_a5_2_2_violation] FP: nvbugs/3384338
            bufferAsValue().~T();
        }
        m_isValid = false;
    }

    // --------------------------------------------------------
    // Monadic functional operations (based on C++ standard proposal p0798R3 - Monadic operations for std::optional).
    // Three operations are provided: and_then(), or_else(), and transform().
    // All operations act on *constant* Optional instances, and always return a new Optional instance *by value*.
    //
    //

    /* These allow chaining of operations depending on the content of an Optional like this:
     *
     * @code
     *     trySomeOperationReturningOptionalFloat() // represents an operation that could fail, returning an Optional
     *    .transform([](float32_t const& valueAsFloat) {
     *        // maps the Optional's value to a new integer type, *if* initial operation did not fail, otherwise an empty Optional is propagated
     *        // .transform doesn't need to return an Optional, but can't be a non-returning void function
     *        return castValueToInt(valueAsFloat);
     *    })
     *    .and_then([](int32_t const& otherValueAsInt) {
     *        // a non-void function - returned Optional is non-empty if input is non-empty
     *        makeOptional(otherValueAsInt + otherValueAsInt);
     *    })
     *    .and_then([](int32_t const& otherValueAsInt) {
     *        // a non-returning void function - returned Optional is filled MonoState if input is non-empty
     *        log(otherValueAsInt);
     *    })
     *    .or_else([]() {
     *        // this is only called in case any of the previous operations failed - returns a filled optional with a default value (bool in this case)
     *        return makeOptional(true);
     *    })
     *    .or_else([]() {
     *        // a terminal function - this is never called in this example, as the previous .or_else() already returns a non-empty Optional.
     *        // Is usually used to, e.g., log errors / set fail state
     *        log("error"); });
     * @endcode
     */

    // TODO(@janickm): extend with rvalue variants to move wrapped values into returned instances

    /**
     * @brief For the case of a non-Monostate optional, returns NULLOPT if the Optional is NULLOPT, otherwise calls non-void unaryFunction
     *        with the wrapped value and returns the function's Optional as a result (can be filled / unfilled).
     *        This replicates rust's Optional<T>.and_then(), some languages call this operation flatmap.
    */
    template <class UnaryFunction,
              typename U      = T,
              typename Result = typename std::result_of<UnaryFunction(T const&)>::type,
              typename        = typename std::enable_if<!std::is_same<Result, void>::value>::type,
              typename std::enable_if<!std::is_same<U, Monostate>::value, U>::type...>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto and_then(UnaryFunction&& unaryFunction) const -> Result
    {
        static_assert(IsSpecialization<Result, Derived>::value, "and_then: Result not a specialization of current Optional type");
        if (m_isValid)
        {
            return std::forward<UnaryFunction>(unaryFunction)(bufferAsValue());
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief Returns a MonoState 'NULLOPT' if the Optional is 'NULLOPT', otherwise calls unaryFunction with void result type on the wrapped value.
     *        Returns a filled MonoState optional if the input was filled, otherwise a MonoState-typed 'NULLOPT'.
     *        This is useful if there is no natural return type for 'and_then' (e.g., if processing has finished), but subsequent 'or_else' calls
     *        still need to be performed based on this Optional's validity state.
    */
    template <class UnaryFunction,
              typename U      = T,
              typename Result = typename std::result_of<UnaryFunction(T const&)>::type,
              typename        = typename std::enable_if<std::is_same<Result, void>::value>::type,
              typename std::enable_if<!std::is_same<U, Monostate>::value, U>::type...>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto and_then(UnaryFunction&& unaryFunction) const -> Derived<Monostate>
    {
        if (m_isValid)
        {
            std::forward<UnaryFunction>(unaryFunction)(bufferAsValue());
            return Monostate{};
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief For the case of a Monostate optional, returns NULLOPT if the Optional is NULLOPT, otherwise calls a generator function and returns the
     *        function's Optional as a result (can be filled / unfilled)
    */
    template <class GeneratorFunction,
              typename U      = T,
              typename Result = typename std::result_of<GeneratorFunction()>::type,
              typename        = typename std::enable_if<!std::is_same<Result, void>::value>::type,
              typename std::enable_if<std::is_same<U, Monostate>::value, U>::type...>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto and_then(GeneratorFunction&& generatorFunction) const -> Result
    {
        static_assert(IsSpecialization<Result, Derived>::value, "and_then: Result not a specialization of current Optional type");
        if (m_isValid)
        {
            return std::forward<GeneratorFunction>(generatorFunction)();
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief For the case of a Monostate optional, returns NULLOPT if the Optional is NULLOPT, otherwise calls the void function and returns
     *        a filled Monostate optional.
     *        This is useful if there is no natural return type for 'and_then' (e.g., if processing has finished), but subsequent 'or_else' calls
     *        still need to be performed based on this Optional's validity state.
    */
    template <class Function,
              typename U      = T,
              typename Result = typename std::result_of<Function()>::type,
              typename        = typename std::enable_if<std::is_same<Result, void>::value>::type,
              typename std::enable_if<std::is_same<U, Monostate>::value, U>::type...>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto and_then(Function&& function) const -> Derived<Monostate>
    {
        if (m_isValid)
        {
            std::forward<Function>(function)();
            return Monostate{};
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief Returns a copy of the Optional if not 'NULLOPT', otherwise calls the provided generatorFunction to generate
     *        Optional of same type (can be filled / unfilled).
    */
    template <class GeneratorFunction,
              typename Result = typename std::result_of<GeneratorFunction()>::type,
              typename        = typename std::enable_if<!std::is_same<Result, void>::value>::type>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto or_else(GeneratorFunction&& generatorFunction) const -> Derived<T>
    {
        static_assert(IsSpecialization<Result, Derived>::value, "or_else: Result of not a specialization of current Optional type");
        if (!m_isValid)
        {
            return std::forward<GeneratorFunction>(generatorFunction)();
        }
        else
        {
            return bufferAsValue();
        }
    }

    /**
     * @brief Call a terminal function if the Optional is 'NULLOPT'. Useful to handle error of previously failed operations.
    */
    template <class Function>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto
    or_else(Function&& function) const -> typename std::enable_if<
        std::is_same<typename std::result_of<Function()>::type, void>::value>::type
    {
        if (!m_isValid)
        {
            std::forward<Function>(function)();
        }
    }

    /**
     * @brief Terminal no-op overload of 'or_else' in case no further action is required on the final Optional value of the processing chain.
     *        This is useful to prevent coverity coverage rule AUTOSAR C++14 A0-1-2 / static_cast<void> on a processing chain that doesn't
     *        require a terminal 'or_else'.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void
    or_else() const
    {
    }

    /**
     * @brief Returns 'NULLOPT' if the Optional is 'NULLOPT', otherwise calls the provided unaryFunction on the value of the Optional,
     *        and returns the result as a new filled Optional wrapping this value.
     *        This is similar to non-void 'and_then', with the difference that return value is not required to be of Optional type.
    */
    template <class UnaryFunction,
              typename Result = typename std::result_of<UnaryFunction(T const&)>::type>
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto transform(UnaryFunction&& unaryFunction) const -> Derived<Result>
    {
        static_assert(!std::is_same<Result, void>::value, "transform: Function can't be void");
        if (m_isValid)
        {
            return std::forward<UnaryFunction>(unaryFunction)(bufferAsValue());
        }
        else
        {
            return {};
        }
    }

protected:
    /**
     * @brief Destructor for BaseOptional.
     * It calls reset() to release the object.
     *
     * Access specifier: protected
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE ~BaseOptional()
    {
        reset(); // makes sure value is destructed
    }

private:
    /**
     * @brief If not a valid optional, call the function handleInvalidAccess of the derived class.
     *
     * Access specifier: private
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void assertValue() const
    {
        if (!m_isValid)
        {
            // CRTP-based delegation to derived class on how to handle invalid optional access
            static_cast<Derived<T> const*>(this)->handleInvalidAccess();
        }
    }
    /**
     * @brief If not a valid optional, call the function handleInvalidAccess of the derived class.
     *        This shall apply only for Debug BUILDS.
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void assertValueDebug() const
    {
#if DW_RUNTIME_CHECKS()
        assertValue();
#endif
    }

    /**
     * @brief Returns the stored optional value (m_valueBuffer)
     *
     * Access specifier: private
     *
     * @return The reference of the optional value
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto bufferAsValue() -> T&
    {
        return *reinterpret_cast<T*>(&m_valueBuffer); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE auto bufferAsValue() const -> T const&
    {
        return *reinterpret_cast<T const*>(&m_valueBuffer); // clang-tidy NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    /// Use storage with same alignment as wrapped value type
    typename std::aligned_storage<sizeof(T), alignof(T)>::type m_valueBuffer = {};

    /// Place valid flag *after* value to not enlarge total type's size too much due to alignment requirements
    bool m_isValid = false;
};

/**
 * @brief Generic exception indicating optional was accessed when empty.
*/
class BadOptionalAccess : public ExceptionWithStackTrace
{
    using ExceptionWithStackTrace::ExceptionWithStackTrace;

public:
    static constexpr char8_t const* MESSAGE{"Optional is empty"};
};

/**
 * @brief Default variant of generic optional (doesn't allocate any additional state)
*/
template <class T>
class Optional : public BaseOptional<T, Optional>
{
    // Use inherited constructors
    using Base = BaseOptional<T, Optional>;
    using Base::Base;

public:
    /**
     * @brief This function is used to handle invalid access.
     * It raises an exception with the specific message.
     *
     * Access specifier: public
     * @throws BadOptionalAccess to handle invalid access
    */
    // coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
    CUDA_BOTH_INLINE void handleInvalidAccess() const
    {
        core::assertException<BadOptionalAccess>(BadOptionalAccess::MESSAGE);
    }
};

/**
 * @brief Relational operators on other optionals (same behaviour as std::optional)
*/
template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
    // 'operator==' for optionals "inherits" the 'noexcept' property from the enclosed type T
    noexcept(noexcept(std::declval<T>() == std::declval<T>()))
{
    if (lhs.has_value() != rhs.has_value())
    {
        return false;
    }
    if (!lhs.has_value())
    {
        return true;
    }

    return *lhs == *rhs;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
{
    return !(lhs == rhs);
}

/**
 * @brief Relational operators on values (same behaviour as std::optional)
*/
template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(BaseOptional<T, D> const& lhs, T const& rhs)
{
    return lhs.has_value() ? *lhs == rhs : false;
}
template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(T const& lhs, BaseOptional<T, D> const& rhs)
{
    return rhs == lhs;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(BaseOptional<T, D> const& lhs, T const& rhs)
{
    return !(lhs == rhs);
}
template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(T const& lhs, BaseOptional<T, D> const& rhs)
{
    return rhs != lhs;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(BaseOptional<T, D> const& lhs, dw::core::NullOpt)
{
    return lhs == Optional<T>{NULLOPT};
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator==(dw::core::NullOpt, BaseOptional<T, D> const& rhs)
{
    return rhs == Optional<T>{NULLOPT};
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(BaseOptional<T, D> const& lhs, dw::core::NullOpt)
{
    return lhs != Optional<T>{NULLOPT};
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator!=(dw::core::NullOpt, BaseOptional<T, D> const& rhs)
{
    return rhs != Optional<T>{NULLOPT};
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
{
    // https://eel.is/c++draft/optional.relops
    // Returns: If !y, false; otherwise, if !x, true; otherwise *x < *y.
    return !rhs.has_value() ? false : (lhs.has_value() ? lhs.value() < rhs.value() : true);
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<(BaseOptional<T, D> const& lhs, const T& rhs)
{
    return lhs.has_value() ? lhs.value() < rhs : true;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<(const T& lhs, BaseOptional<T, D> const& rhs)
{
    return !rhs.has_value() ? false : lhs < rhs.value();
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<=(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
{
    // https://eel.is/c++draft/optional.relops
    // Returns: If !x, true; otherwise, if !y, false; otherwise *x <= *y.
    return !lhs.has_value() ? true : (rhs.has_value() ? lhs.value() <= rhs.value() : false);
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<=(BaseOptional<T, D> const& lhs, const T& rhs)
{
    return !lhs.has_value() ? true : lhs.value() <= rhs;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator<=(const T& lhs, BaseOptional<T, D> const& rhs)
{
    return rhs.has_value() ? lhs <= rhs.value() : false;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
{
    // https://eel.is/c++draft/optional.relops
    // Returns: If !x, false; otherwise, if !y, true; otherwise *x > *y.
    return !lhs.has_value() ? false : (rhs.has_value() ? lhs.value() > rhs.value() : true);
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>(BaseOptional<T, D> const& lhs, const T& rhs)
{
    return !lhs.has_value() ? false : lhs.value() > rhs;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>(const T& lhs, BaseOptional<T, D> const& rhs)
{
    return rhs.has_value() ? lhs > rhs.value() : true;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>=(BaseOptional<T, D> const& lhs, BaseOptional<T, D> const& rhs)
{
    // https://eel.is/c++draft/optional.relops
    // Returns: If !y, true; otherwise, if !x, false; otherwise *x >= *y.
    return !rhs.has_value() ? true : (lhs.has_value() ? lhs.value() >= rhs.value() : false);
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>=(BaseOptional<T, D> const& lhs, const T& rhs)
{
    return lhs.has_value() ? lhs.value() >= rhs : false;
}

template <typename T, template <class> class D>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool operator>=(const T& lhs, BaseOptional<T, D> const& rhs)
{
    return !rhs.has_value() ? true : lhs >= rhs.value();
}

/**
 * @brief Creates an engaged Optional wrapping the provided value
*/
template <class T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr auto makeOptional(T&& v) -> Optional<typename std::decay<T>::type>
{
    return Optional<typename std::decay<T>::type>(std::forward<T>(v));
}

/**
 * @brief Creates a UnitOptional wrapping a stateless Monostate value.
*/
/// The Optional's engagement-state is equal to the provided boolean flag.
using UnitOptional = Optional<Monostate>;
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE UnitOptional makeUnitOptional(bool const state)
{
    return state ? UnitOptional(Monostate{}) : UnitOptional{};
}

// This type instance is already explicitly instantiated to verify via static analysis
// (explicitly use an instance unlikely used in practice)
extern template class Optional<void*>;

} // namespace core
} // namespace dw

#endif // DW_CUDA_OPTIONAL_HPP_
