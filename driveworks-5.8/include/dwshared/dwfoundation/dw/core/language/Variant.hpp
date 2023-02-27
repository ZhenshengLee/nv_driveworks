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
// Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DWSHARED_DW_CORE_VARIANT_HPP_
#define DWSHARED_DW_CORE_VARIANT_HPP_

#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/platform/CompilerSpecificMacros.hpp>

#include <type_traits>

#include "cxx14.hpp"
#include "cxx17.hpp"
#include "impl/VariantImpl.hpp"

namespace dw
{
namespace core
{
namespace detail
{

struct VariantPrivacyBypass;

} // namespace dw::core::detail

template <typename... Types>
class Variant;

/// @{
/// Check if @c T is a @ref Variant type. If it is, this will derive from @c std::true_type; otherwise this will derive
/// from @c std::false_type.
template <typename T>
using IsVariant = IsSpecialization<T, Variant>;

/// @see IsVariant
template <typename T>
constexpr bool IS_VARIANT_V = IsVariant<T>::value;
/// @}

/// Thrown from @ref Variant operations which attempt to access an instance with a different type than required or one
/// which is @ref Variant::valueless_by_exception.
class BadVariantAccess : public ExceptionWithStackTrace
{
public:
    using ExceptionWithStackTrace::ExceptionWithStackTrace;

    BadVariantAccess();

    ~BadVariantAccess() noexcept override;

    /// The default message of exceptions.
    static char8_t const* MESSAGE;
};

using detail::VariantSizeType;

/// The index of a @ref Variant in the invalid state.
///
/// @see Variant::valueless_by_exception
using detail::VARIANT_NPOS;

/// @{
/// If @c T is a @c Variant, this type contains a member constant @c value with the number of alternatives.
///
/// @see VARIANT_SIZE_V
template <typename T>
struct VariantSize
{
};

template <typename... Types>
struct VariantSize<Variant<Types...>> : std::integral_constant<VariantSizeType, sizeof...(Types)>
{
};

template <typename T>
struct VariantSize<T const> : VariantSize<T>
{
};

/// @see VariantSize
template <typename T>
constexpr VariantSizeType VARIANT_SIZE_V = VariantSize<T>::value;
/// @}

/// @{
/// Get the type at the zero-based @c Index of the @c TVariant. If @c TVariant is a @ref Variant and @c Index is a valid
/// index, there will be a member type named @c type with this type. This transformation trait is @c const preserving.
template <VariantSizeType Index, typename TVariant>
struct VariantAlternative
{
};

template <VariantSizeType Index, typename... Types>
struct VariantAlternative<Index, Variant<Types...>> : detail::VariantNthType<Index, Types...>
{
};

template <VariantSizeType Index, typename... Types>
struct VariantAlternative<Index, Variant<Types...> const> : detail::VariantNthType<Index, Types const...>
{
};

/// @see VariantAlternative
template <VariantSizeType Index, typename TVariant>
using VariantAlternativeT = typename VariantAlternative<Index, TVariant>::type;
/// @}

/// A type-safe discriminated union. Like types defined by the @c union keyword, the same memory region is used to store
/// the possible types. Unlike a @c union, access to a variant of a specific type is always checked at runtime and the
/// special members of construction, assignment, and destruction are automatically synthesized correctly as-needed.
///
/// @note
/// This class is as close to compatible with @c std::variant while staying AUTOSAR compliant. In general, this means
/// the implementation is slightly more restrictive than the @c std::variant implementation you would find in the
/// Standard Library. For example, @c Types with destructors which throw result in unspecified behavior for operations
/// like converting move assignments; in this implementation, @c Types are simply not allowed to have destructors which
/// can throw.
template <typename... Types>
class Variant
    : private detail::VariantImpl<Types...>
{
    using BaseType = detail::VariantImpl<Types...>;

    using SelfType = Variant;

    // NOTE: This should remain private -- see the comment on detail::VariantTraits for more information.
    using Traits = typename BaseType::Traits;

    friend struct detail::VariantPrivacyBypass;

public:
    using SizeType = VariantSizeType;

public:
    /// The default constructor is only enabled if the first type of @c Types has a default constructor. This operation
    /// is never trivial, as the @ref index field must always be initialized.
    Variant() = default;

    /// The copy constructor is enabled only if all @c Types have a copy constructor. This constructor is trivial if all
    /// types have trivial copy constructors.
    Variant(Variant const&) = default;

    /// The move constructor is enabled if all @c Types have a move constructor or, for those that do not, there is a
    /// copy constructor. This constructor is trivial if all types have trivial move constructors.
    Variant(Variant&&) = default;

    /// @{
    /// Construct the variant as the alternative with the given @c Index using the specified @a args. This function only
    /// participates in overload resolution when the requested type is constructible from the provided @a args.
    template <SizeType Index,
              typename... Args,
              typename = std::enable_if_t<std::is_constructible<VariantAlternativeT<Index, SelfType>, Args...>::value>>
    CUDA_BOTH_INLINE constexpr explicit Variant(in_place_index_t<Index> tag, Args&&... args) noexcept(std::is_nothrow_constructible<VariantAlternativeT<Index, SelfType>, Args...>::value)
        : BaseType(tag, std::forward<Args>(args)...)
    {
    }

    template <SizeType Index,
              typename U,
              typename... Args,
              typename = std::enable_if_t<std::is_constructible<VariantAlternativeT<Index, SelfType>,
                                                                std::initializer_list<U>&,
                                                                Args...>::value>>
    CUDA_BOTH_INLINE constexpr explicit Variant(in_place_index_t<Index> tag, std::initializer_list<U> initList, Args&&... args) noexcept(std::is_nothrow_constructible<VariantAlternativeT<Index, SelfType>, std::initializer_list<U>&, Args...>::value)
        : BaseType(tag, initList, std::forward<Args>(args)...)
    {
    }
    /// @}

    /// @{
    /// Construct the variant as the @c T alternative using the specified @a args. This function only participates in
    /// overload resolution when the requested type is constructible from the provided @a args and when @c T appears
    /// exactly once in the @c Types list.
    template <typename T,
              typename... Args,
              typename       = std::enable_if_t<std::is_constructible<T, Args...>::value>,
              SizeType Index = Traits::template IndexOfAlternative<T>::value>
    CUDA_BOTH_INLINE constexpr explicit Variant(in_place_type_t<T>, Args&&... args) noexcept(std::is_nothrow_constructible<T, Args...>::value)
        : BaseType(in_place_index<Index>, std::forward<Args>(args)...)
    {
    }

    template <typename T,
              typename U,
              typename... Args,
              typename       = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value>,
              SizeType Index = Traits::template IndexOfAlternative<T>::value>
    CUDA_BOTH_INLINE constexpr explicit Variant(in_place_type_t<T>, std::initializer_list<U> initList, Args&&... args) noexcept(std::is_nothrow_constructible<T, std::initializer_list<U>&, Args...>::value)
        : BaseType(in_place_index<Index>, initList, std::forward<Args>(args)...)
    {
    }
    /// @}

    /// Convert from a given @a src. This function only participates in overload resolution when the decayed form of
    /// @c T is neither a @c Variant, a specialization of @c in_place_type_t or @c in_place_index_t, and there is only
    /// one converting constructor in @c Types which accepts a @c T.
    template <typename T,
              typename TSelectedTraits = typename Traits::template AlternativeFromConstruction<T>,
              typename TSelected       = typename TSelectedTraits::type>
    CUDA_BOTH_INLINE constexpr Variant(T&& src) noexcept(std::is_nothrow_constructible<TSelected, T&&>::value) // clang-tidy NOLINT(google-explicit-constructor)
        : BaseType(in_place_index<TSelectedTraits::INDEX>, std::forward<T>(src))
    {
    }

    /// The copy-assignment operator is enabled if all @c Types have a copy-constructor and a copy-assignment operator.
    /// The requirment on having a copy-constructor comes from assignment on type-changing operators, which are
    /// implemented by destroying the current value, then copy-constructing the new value. This operator is trivial if
    /// all types are trivially destructible, trivially copy-constructible, and trivially copy-assignable.
    Variant& operator=(Variant const&) = default;

    /// The move-assignment operator is enabled if all @c Types have a move-constructor and a move-assignment operator
    /// or if the copy-assignment operator is trivial. The move-based move assignment logic happens for the same reason
    /// as copy-assignment, except with move. This operator is trivial if all types are trivially destructible,
    /// trivially move-constructible, and trivially move-assignable.
    ///
    /// A trivial copy-assignment enables move-assignment on non-movable types because a move-assignment can be
    /// trivially implemented with a trivial copy. If this seems odd, remember that a type with a trivial copy which is
    /// unmoveable does not make a lot of sense in the first place.
    Variant& operator=(Variant&&) = default;

    /// Assign from a given @a src. This function only participates in overload resolution when the decayed form of
    /// @c T is neither a @c Variant, a specialization of @c in_place_type_t or @c in_place_index_t, and there is only
    /// one converting assignment operator in @c Types which accepts a @c T.
    template <typename T,
              typename TSelectedTraits = typename Traits::template AlternativeFromAssignment<T>,
              typename TSelected       = typename TSelectedTraits::type>
    CUDA_BOTH_INLINE Variant& operator =(T&& src)
    {
        BaseType::emplace(detail::VariantIndex<TSelectedTraits::INDEX>(), std::forward<T>(src));
        return *this;
    }

    /// The destructor is trivial if all alternative @c Types have trivial destructors.
    ~Variant() = default;

    /// @return the zero-based index of the alternative currently contained in this instance or @ref VARIANT_NPOS if the
    ///         instance is @ref valueless_by_exception.
    CUDA_BOTH_INLINE constexpr SizeType index() const noexcept
    {
        return BaseType::index();
    }

    /// Check if this instance is valueless. This can happen in cases where a type-changing assignment or @ref emplace
    /// throws an exception after the destruction of the previous value.
    CUDA_BOTH_INLINE constexpr bool valueless_by_exception() const noexcept
    {
        return index() == VARIANT_NPOS;
    }

    /// @{
    /// Construct the alternative with the given @c Index using the provided @a args through direct initialization. If
    /// there is a value, it is destroyed before attempting to initialize the new type (even in cases when the current
    /// @ref index is identical to @c Index). If the constructor of the new type throws, this instance will become
    /// @ref valueless_by_exception. This function only participates in overload resolution when the requested type is
    /// constructible from the provided @a args.
    template <SizeType Index, typename... Args>
    CUDA_BOTH_INLINE std::enable_if_t<std::is_constructible<VariantAlternativeT<Index, SelfType>, Args...>::value,
                                      VariantAlternativeT<Index, SelfType>&>
    emplace(Args&&... args)
    {
        return BaseType::emplace(detail::VariantIndex<Index>(), std::forward<Args>(args)...);
    }

    template <SizeType Index, typename U, typename... Args>
    CUDA_BOTH_INLINE std::enable_if_t<std::is_constructible<VariantAlternativeT<Index, SelfType>, std::initializer_list<U>&, Args...>::value,
                                      VariantAlternativeT<Index, SelfType>&>
    emplace(std::initializer_list<U> initList, Args&&... args)
    {
        return BaseType::emplace(detail::VariantIndex<Index>(), initList, std::forward<Args>(args)...);
    }
    /// @}

    /// @{
    /// Construct the alternative with the given @c T using the provided @a args through direct initialization. If there
    /// is a value, it is destroyed before attempting to initialize the new type (even in cases where @ref index holds
    /// the specified @c T). If the constructor of the new type throws, this instance will become
    /// @ref valueless_by_exception. This function only participates in overload resolution when the requested type
    /// appears exactly once in @c Types list and is constructible from the provided @a args.
    template <typename T, typename... Args, VariantSizeType Index = Traits::template IndexOfAlternative<T>::value>
    CUDA_BOTH_INLINE std::enable_if_t<std::is_constructible<T, Args...>::value, T&> emplace(Args&&... args)
    {
        return BaseType::emplace(detail::VariantIndex<Index>(), std::forward<Args>(args)...);
    }

    template <typename T, typename U, typename... Args, VariantSizeType Index = Traits::template IndexOfAlternative<T>::value>
    CUDA_BOTH_INLINE std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value, T&>
    emplace(std::initializer_list<U> initList, Args&&... args)
    {
        return BaseType::emplace(detail::VariantIndex<Index>(), initList, std::forward<Args>(args)...);
    }
    /// @}

    /// Swap the contents of this instance with @a other.
    CUDA_BOTH_INLINE void swap(Variant& other)
    {
        BaseType::swap(other);
    }
};

extern template class Variant<Monostate>;

namespace detail
{

/// This class gives users access to the implementation of @ref Variant, bypassing the fact that @c detail::VariantImpl
/// is a private base. Since all state lives in the implementation, operations on the implementation are safe -- privacy
/// is used to restrict operations to a useful subset rather than for preservation of invariants. Even so, users should
/// not use this.
struct VariantPrivacyBypass
{
    template <typename... Types>
    CUDA_BOTH_INLINE static constexpr VariantImpl<Types...> const& toImpl(Variant<Types...> const& src) noexcept
    {
        return src;
    }

    template <typename... Types>
    CUDA_BOTH_INLINE static constexpr VariantImpl<Types...>& toImpl(Variant<Types...>& src) noexcept
    {
        return src;
    }
};

} // namespace dw::core::detail

/// @{
/// Get the value of @a src as the type specified by @c Index if the current alternative matches @c Index.
///
/// @return a pointer to the value if @c src.index() is @c Index or @c nullptr if it is not.
template <VariantSizeType Index, typename... Types>
CUDA_BOTH_INLINE constexpr std::add_pointer_t<VariantAlternativeT<Index, Variant<Types...>>> get_if(Variant<Types...>* src) noexcept
{
    return detail::VariantPrivacyBypass::toImpl(*src).getIf(detail::VariantIndex<Index>{});
}

template <VariantSizeType Index, typename... Types>
CUDA_BOTH_INLINE constexpr std::add_pointer_t<VariantAlternativeT<Index, Variant<Types...>> const> get_if(Variant<Types...> const* src) noexcept
{
    return detail::VariantPrivacyBypass::toImpl(*src).getIf(detail::VariantIndex<Index>{});
}
/// @}

/// @{
/// Get the value of @a src as @c T if @c T is the current alternative. This overload is only enabled if @c T appears
/// exactly once in the @c Types list.
///
/// @return a pointer to the value if @c T is the current alternative or @c nullptr if it is not.
template <typename T, typename... Types, VariantSizeType Index = detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value>
CUDA_BOTH_INLINE constexpr std::add_pointer_t<T> get_if(Variant<Types...>* src) noexcept
{
    return get_if<Index>(src);
}

template <typename T, typename... Types, VariantSizeType Index = detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value>
CUDA_BOTH_INLINE constexpr std::add_pointer_t<T const> get_if(Variant<Types...> const* src) noexcept
{
    return get_if<Index>(src);
}
/// @}

/// @{
/// Get the value of @a src as the type specified by the @c Index.
///
/// @throw BadVariantAccess if the @ref Variant::index does not match the requested @c Index.
template <VariantSizeType Index, typename... Types>
CUDA_BOTH_INLINE constexpr VariantAlternativeT<Index, Variant<Types...>> const& get(Variant<Types...> const& src)
{
    return detail::VariantPrivacyBypass::toImpl(src).get(detail::VariantIndex<Index>{});
}

template <VariantSizeType Index, typename... Types>
CUDA_BOTH_INLINE constexpr VariantAlternativeT<Index, Variant<Types...>>& get(Variant<Types...>& src)
{
    return detail::VariantPrivacyBypass::toImpl(src).get(detail::VariantIndex<Index>{});
}

template <VariantSizeType Index, typename... Types>
CUDA_BOTH_INLINE constexpr VariantAlternativeT<Index, Variant<Types...>>&& get(Variant<Types...>&& src)
{
    return std::move(get<Index>(src));
}
/// @}

/// @{
/// Get the value of @a src as the specified type @c T. This overload is only enabled if @c T appears exactly once in
/// the @c Types list.
///
/// @throw BadVariantAccess if the variant is not currently holding the @c T alternative.
template <typename T, typename... Types, VariantSizeType Index = detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value>
CUDA_BOTH_INLINE constexpr T const& get(Variant<Types...> const& src)
{
    return get<Index>(src);
}

template <typename T, typename... Types, VariantSizeType Index = detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value>
CUDA_BOTH_INLINE constexpr T& get(Variant<Types...>& src)
{
    return get<Index>(src);
}

template <typename T, typename... Types, VariantSizeType Index = detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value>
CUDA_BOTH_INLINE constexpr T&& get(Variant<Types...>&& src)
{
    return std::move(get<Index>(src));
}
/// @}

/// Test if the given @a value holds the type @c T. This function is only enabled if @c T appears exactly once in the
/// @ref Variant type list.
template <typename T, typename... Types>
CUDA_BOTH_INLINE constexpr std::enable_if_t<meta::COUNT_V<std::is_same<T, Types>...> == 1U, bool>
holds_alternative(Variant<Types...> const& value)
{
    // WAR in the above signature: you may think to replace "meta::COUNT_V..."  with "...::ALTERNATIVES_MATCHING...".
    // That works, but in QNX the compiler will complain about ALTERNATIVES_MATCHING not being a function
    // template...which is a fact, but also somehow an error to q++.  Sigh.
    return value.index() == detail::VariantTraits<Types...>::template IndexOfAlternative<T>::value;
}

/// @{
/// Invoke a @a visitor function with the active type in the @a value variant.
///
/// @tparam R The result type. If unspecified, this type will be deduced from the @c std::common_type of applying the
///           @a visitor to all alternatives of the @a value.
///
/// @throw BadVariantAccess if @a value is @ref Variant::valueless_by_exception.
///
/// @note
/// The equivalent @c std::visit function overload for @c std::variant allows for an arbitrary number of instances
/// instead of a single @a value (as this implementation is limited to). This ability has not been implemented because
/// it is difficult to write and has not yet been needed.
template <typename R, typename FVisitor, typename TVariant>
CUDA_BOTH_INLINE std::enable_if_t<IS_VARIANT_V<std::decay_t<TVariant>>, R> visit(FVisitor&& visitor, TVariant&& value)
{
    if (value.valueless_by_exception())
    {
        detail::throwBadVariantAccess("visit: Attempt to visit valueless variant");
    }

    return detail::VariantPrivacyBypass::toImpl(value)
        .template visitIndexed<R>([&](auto tag) -> R {
            constexpr VariantSizeType INDEX = std::decay_t<decltype(tag)>::value;

            return std::forward<FVisitor>(visitor)(get<INDEX>(std::forward<TVariant>(value)));
        });
}

template <typename FVisitor, typename TVariant>
CUDA_BOTH_INLINE detail::VariantVisitResultT<FVisitor, TVariant> visit(FVisitor&& visitor, TVariant&& value)
{
    using ReturnType = detail::VariantVisitResultT<FVisitor, TVariant>;

    return visit<ReturnType>(std::forward<FVisitor>(visitor), std::forward<TVariant>(value));
}
/// @}

template <typename... Types>
CUDA_BOTH_INLINE void swap(Variant<Types...>& lhs, Variant<Types...>& rhs) noexcept(noexcept(lhs.swap(rhs)))
{
    lhs.swap(rhs);
}
} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_DW_CORE_VARIANT_HPP_*/
