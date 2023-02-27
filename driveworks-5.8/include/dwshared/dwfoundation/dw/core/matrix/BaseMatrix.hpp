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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_MATRIX_BASEMATRIX_HPP_
#define DW_CORE_MATRIX_BASEMATRIX_HPP_

#include <type_traits>

#include <dw/core/platform/CompilerSpecificMacros.hpp>
#include <dw/core/language/BasicTypes.hpp>
#include <dw/core/platform/Limits.hpp>
#include <dw/core/utility/Compare.hpp>
#include <dw/core/utility/Base2.hpp>
#include <dw/core/ConfigChecks.h>
#include <dw/core/language/Optional.hpp>
#include <dw/core/container/Span.hpp>
#include <dw/core/container/StaticSpan.hpp>
#include <dw/core/platform/SIMD.hpp>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

// TODO(dwplc): FP - ## is only used once per declaration, and Type is template argument which can't be enclosed in parentheses
// coverity[autosar_cpp14_m16_0_6_violation]
// coverity[autosar_cpp14_m16_3_1_violation]
#define MATRIX_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
    using Matrix1##TypeSuffix  = Matrix<Type, 1, 1>;     \
    using Vector1##TypeSuffix  = Vector<Type, 1>;        \
    using Matrix2##TypeSuffix  = Matrix<Type, 2, 2>;     \
    using Vector2##TypeSuffix  = Vector<Type, 2>;        \
    using Matrix3##TypeSuffix  = Matrix<Type, 3, 3>;     \
    using Vector3##TypeSuffix  = Vector<Type, 3>;        \
    using Matrix4##TypeSuffix  = Matrix<Type, 4, 4>;     \
    using Vector4##TypeSuffix  = Vector<Type, 4>;        \
    using Matrix5##TypeSuffix  = Matrix<Type, 5, 5>;     \
    using Vector5##TypeSuffix  = Vector<Type, 5>;        \
    using Matrix6##TypeSuffix  = Matrix<Type, 6, 6>;     \
    using Vector6##TypeSuffix  = Vector<Type, 6>;        \
    using Matrix34##TypeSuffix = Matrix<Type, 3, 4>;

// decide whether to engage CPU-side matrix improvements
#if DW_ENABLE_ARM_SIMD
#define DW_MATRIX_CPU_OPTS 1
#else
#define DW_MATRIX_CPU_OPTS 0
#endif

namespace dw
{
namespace core
{

/// Stuff in the detail namespace isn't meant to be used outside of the local headers.
/// They are implementation details
namespace detail
{
/// Helper struct to determine the type of the expression a*b
template <class A, class B>
struct MultiplyResult
{
    // TODO(dwplc): FP - basic numerical type "int" or "unsigned int" aren't used here
    // coverity[autosar_cpp14_a3_9_1_violation]
    using type = decltype(std::declval<A>() * std::declval<B>());
};

/// Helper struct to determine the type of the expression a+b
template <class A, class B>
struct AddResult
{
    using type = decltype(std::declval<A>() + std::declval<B>());
};
} // namespace detail

////////////////////////////////////////////////////////////////////////
// Forward defines
////////////////////////////////////////////////////////////////////////
/// This is our main dense small-size Matrix class for cuda
/// It implements basic matrix-matrix and matrix-scalar operations (*,+,-,/)
/// Its memory is aligned to make sure that memory load-store can be vectorized.
///

// Note: nvcc (as of cuda 9.1) has a limitation that it won't vectorize matrix access
// if elements are assigned individually. Thus, all operations compute each column in local
// variables (registers) and then assign the entire column at once. This ensures that vector
// load/store is used by nvcc. See: http://nvbugs/2052647
// TODO(dwplc): [autosar_cpp14_a12_0_1_violation] declare all others of these five special member functions, it
// would meet another clang-tidy issue: error: pass by value and use std::move [modernize-pass-by-value,-warnings-as-errors]
// Left a TODO here need to fix.
// TODO(dwplc): FP - this is a forward declaration,
// coverity[autosar_cpp14_m3_2_3_violation]
// coverity[autosar_cpp14_a12_0_1_violation]
template <class T, size_t ROW_COUNT, size_t COLUMN_COUNT>
class Matrix;

// TODO(dwplc): FP - this is a forward declaration
// coverity[autosar_cpp14_m3_2_3_violation]
template <class T, size_t DIM_>
class SymmetricMatrix;

// TODO(dwplc): FP - this is a forward declaration
// coverity[autosar_cpp14_m3_2_3_violation]
template <class T, size_t DIM, size_t UB>
class SymmetricBandedMatrix;

// TODO(dwplc): FP - this is a forward declaration
// coverity[autosar_cpp14_m3_2_3_violation]
template <class T, size_t Dim_>
class DiagonalMatrix;

// TODO(dwplc): FP - this is a forward declaration
// coverity[autosar_cpp14_m3_2_3_violation]
template <class T, size_t N, bool IS_LOWER>
class TriangularMatrix;

////////////////////////////////////////////////////////////////
/// Storage classes and storage to matrix link
///
/// The matrix storage determines which elements are stored and which are not.
/// It also provides the conversion between (row,col) and linear array indexing.
///
namespace detail
{
template <class T, size_t ROW_COUNT, size_t COLUMN_COUNT>
struct DenseMatrixStorage;

template <class T, size_t DIM>
struct SymmetricMatrixStorage;

template <class T, size_t DIM, size_t UB>
struct SymmetricBandedMatrixStorage;

template <class T, size_t DIM>
struct DiagonalMatrixStorage;

template <class T, size_t N, bool IS_LOWER>
struct TriangularMatrixStorage;

/// Return the type of a matrix given the storage type.
/// Set the scalar type of the result to T2
template <class TStorage, class T2 = typename TStorage::Scalar>
struct MatrixFromStorage;

template <class T, size_t ROW_COUNT, size_t COLUMN_COUNT, class T2>
struct MatrixFromStorage<DenseMatrixStorage<T, ROW_COUNT, COLUMN_COUNT>, T2>
{
    using type = Matrix<T2, ROW_COUNT, COLUMN_COUNT>;
};

template <class T, size_t DIM, class T2>
struct MatrixFromStorage<SymmetricMatrixStorage<T, DIM>, T2>
{
    using type = SymmetricMatrix<T2, DIM>;
};

template <class T, size_t DIM, size_t UB, class T2>
struct MatrixFromStorage<SymmetricBandedMatrixStorage<T, DIM, UB>, T2>
{
    using type = SymmetricBandedMatrix<T2, DIM, UB>;
};

template <class T, size_t DIM, class T2>
struct MatrixFromStorage<DiagonalMatrixStorage<T, DIM>, T2>
{
    using type = DiagonalMatrix<T2, DIM>;
};
template <class T, size_t DIM, bool IS_LOWER, class T2>
struct MatrixFromStorage<TriangularMatrixStorage<T, DIM, IS_LOWER>, T2>
{
    using type = TriangularMatrix<T2, DIM, IS_LOWER>;
};
} // namespace detail

//////////////////////////////////////////////////////
/// Aliases for commonly used matrix types
///
template <class T, size_t ROW_COUNT>
using Vector = Matrix<T, ROW_COUNT, 1>;

template <class T, size_t ColCount>
using RowVector = Matrix<T, 1, ColCount>;

MATRIX_MAKE_TYPEDEFS_ALL_SIZES(uint8_t, ub)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(int8_t, b)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(uint16_t, us)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(int16_t, s)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(uint32_t, ui)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(int32_t, i)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(size_t, sz)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(dwFloat16_t, h)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(float32_t, f)
MATRIX_MAKE_TYPEDEFS_ALL_SIZES(float64_t, d)

// TODO(dwplc): FP - this is a forward declaration
// coverity[autosar_cpp14_m3_2_3_violation]
template <class Ta, class Tb, size_t ROWS, size_t COLS>
CUDA_BOTH_INLINE auto
dot(const Matrix<Ta, ROWS, COLS>& a,
    const Matrix<Tb, ROWS, COLS>& b) -> typename detail::MultiplyResult<Ta, Tb>::type;

////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////

//Alignment helper structs
namespace detail
{

/// Determines what should be the alignment for a type of the given size.
/// It avoids padding but enforces alignment when the size is already a power of two
///
/// Note: limit to 16 byte alignment because of the warning
///       The ABI for passing parameters with 32-byte alignment has changed in GCC 4.6
///       Which seems to lead to interoperability bugs between gcc and nvcc even with
///       higher versions. See: https://github.com/ComputationalRadiationPhysics/picongpu/issues/1553
template <size_t size>
struct ByteAlignmentHelper
{
    // clang-format off
    static constexpr size_t VALUE = ((size/16UL)*16UL==size) ? 16 :
                                    ((size/8UL)*8UL==size) ? 8 :
                                    ((size/4UL)*4UL==size) ? 4 :
                                    ((size/2UL)*2UL==size) ? 2 : 1;
    // clang-format on
};

template <class TScalar, size_t ELEMENT_COUNT>
struct AlignmentHelper
{
    static constexpr size_t VALUE = util::max(std::alignment_of<TScalar>::value, ByteAlignmentHelper<sizeof(TScalar) * ELEMENT_COUNT>::VALUE);
};

/////////////////////////////////////////////////////////////////////////////////////
/// Storage class for a dense matrix
/// BaseMatrix receives a storage type that determines which elements are stored and how.
/// The storage class must have:
///     constexpr size_t ELEMENT_COUNT: number of elements that are physically stored
///     constexpr bool isElementStored(row, col): returns true if the given element is stored phyiscally
///     constexpr size_t indexFromRowCol(row, col): transforms 2D indexing into 1D linear indexing
///     T dataArray[ELEMENT_COUNT]: the actual elements stored.
///
/// All must be constexpr so they can be optimized out at compile-time.
///
template <class T, size_t ROW_COUNT, size_t COLUMN_COUNT>
struct DenseMatrixStorage // clang-tidy NOLINT - only here because clang-tidy crashes on AutoDiff.cu
{
    using Scalar                          = T;
    static constexpr size_t ELEMENT_COUNT = COLUMN_COUNT * ROW_COUNT;
    static constexpr size_t ALIGNMENT     = detail::AlignmentHelper<T, ELEMENT_COUNT>::VALUE;

    /// Indicates if the element is stored
    /// This is trivial for this class but makes sense because
    /// DiagonalMatrix and SymmetricMatrix have the same API
    CUDA_BOTH_INLINE
    static constexpr bool isElementStored(size_t const row, size_t const col) { return (row < ROW_COUNT) && (col < COLUMN_COUNT); }

    /// Converts the 2d index into a linear index inside dataArray
    CUDA_BOTH_INLINE
    static constexpr size_t indexFromRowCol(size_t const row, size_t const col)
    {
        size_t res{0};
        if ((row < ROW_COUNT) && (col < COLUMN_COUNT))
        {
            res = col * ROW_COUNT + row;
        }
#if DW_RUNTIME_CHECKS()
        else
        {
#if !defined(__CUDACC__)
            throwMatrixIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>(
                "DenseMatrixStorage::indexFromRowCol(): Tried to get index outside of matrix");
#endif
        }
#endif
        return res;
    }

    DenseMatrixStorage() = default;

    /// Allow construction with storage of different shape so we can reshape matrices
    template <size_t R2, size_t C2>
    explicit DenseMatrixStorage(const DenseMatrixStorage<T, R2, C2>& other)
    {
        static_assert(R2 * C2 == ELEMENT_COUNT, "The element count must match");
        for (size_t i = 0; i < ELEMENT_COUNT; ++i)
        {
            m_dataArray[i] = other[i];
        }
    }

    CUDA_BOTH_INLINE
    auto operator[](std::size_t const idx) -> T&
    {
#if DW_RUNTIME_CHECKS()
#if !defined(__CUDACC__)
        if (idx >= ELEMENT_COUNT)
        {
            throwMatrixIndexOutOfBounds();
        }
#else
        assertExceptionIf<OutOfBoundsException>(idx < ELEMENT_COUNT, "DenseMatrixStorage::operator[]: Tried to get index outside of matrix");
#endif
#endif
        return m_dataArray[idx];
    }

    CUDA_BOTH_INLINE
    auto operator[](std::size_t const idx) const -> const T&
    {
#if DW_RUNTIME_CHECKS()
#if !defined(__CUDACC__)
        if (idx >= ELEMENT_COUNT)
        {
            throwMatrixIndexOutOfBounds();
        }
#else
        assertExceptionIf<OutOfBoundsException>(idx < ELEMENT_COUNT, "DenseMatrixStorage::operator[]: Tried to get index outside of matrix");
#endif
#endif
        return m_dataArray[idx];
    }

    CUDA_BOTH_INLINE
    auto begin() -> T*
    {
        return m_dataArray.begin();
    }
    CUDA_BOTH_INLINE
    auto begin() const -> const T*
    {
        return m_dataArray.begin();
    }
    CUDA_BOTH_INLINE
    auto end() -> T*
    {
        return m_dataArray.end();
    }
    CUDA_BOTH_INLINE
    auto end() const -> const T*
    {
        return m_dataArray.end();
    }
    CUDA_BOTH_INLINE
    auto data() -> T*
    {
        return m_dataArray.data();
    }
    CUDA_BOTH_INLINE
    auto data() const -> const T*
    {
        return m_dataArray.data();
    }

private:
    AlignedArray<T, ELEMENT_COUNT, ALIGNMENT> m_dataArray;
};
} // namespace detail

///
/// Base Matrix Class Definition
///
// TODO(dwplc): [autosar_cpp14_a12_0_1_violation] declare all others of these five special member functions, it
// would meet another clang-tidy issue: error: pass by value and use std::move [modernize-pass-by-value,-warnings-as-errors]
// Left a TODO here need to fix.
template <class T, size_t ROW_COUNT_, size_t COLUMN_COUNT_, class TStorage>
class BaseMatrix
{
public:
    using Scalar                          = T;
    static constexpr size_t ROW_COUNT     = ROW_COUNT_;
    static constexpr size_t COLUMN_COUNT  = COLUMN_COUNT_;
    static constexpr size_t ELEMENT_COUNT = TStorage::ELEMENT_COUNT;
    static constexpr size_t ALIGNMENT     = TStorage::ALIGNMENT;

    static constexpr bool IS_VECTOR = (ROW_COUNT == 1) || (COLUMN_COUNT == 1);

    static_assert(ROW_COUNT > 0, "ROW_COUNT must be > 0");
    static_assert(COLUMN_COUNT > 0, "COLUMN_COUNT must be > 0");
    static_assert(ROW_COUNT <= (std::numeric_limits<size_t>::max() / COLUMN_COUNT), "ROW_COUNT * COLUMN_COUNT must not overflow!");
    static_assert(ROW_COUNT * COLUMN_COUNT <= (std::numeric_limits<size_t>::max() - ROW_COUNT), "ROW_COUNT * COLUMN_COUNT + ROW_COUNT must not overflow!");

    using TColumn = Vector<T, ROW_COUNT>;
    using TRow    = RowVector<T, COLUMN_COUNT>;

    /// This is the real type of the matrix for *this
    /// This can be used for the methods that must return another matrix of the same type.
    /// E.g. TDerived BaseMatrix::abs() const;
    using TDerived = typename detail::MatrixFromStorage<TStorage>::type;

    /// The dense equivalent to this class.
    using TDense = Matrix<T, ROW_COUNT, COLUMN_COUNT>;

    /// Set all the matrix elements to zero.
    CUDA_BOTH_INLINE
    void setZero()
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            m_data[i] = T{};
        }
    }

    /// Set all the matrix elements to one.
    CUDA_BOTH_INLINE
    void setOnes()
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            // TODO(dwplc): FP - report C-style casts when T = bool.
            // coverity[autosar_cpp14_a5_2_2_violation]
            m_data[i] = static_cast<T>(1);
        }
    }

    /// Set to identity matrix.
    CUDA_BOTH_INLINE
    void setIdentity()
    {
        setZero();
        constexpr size_t MAXI = ROW_COUNT < COLUMN_COUNT ? ROW_COUNT : COLUMN_COUNT;
        for (size_t i = 0; i < MAXI; i++)
        {
            atRef(i, i) = static_cast<T>(1);
        }
    }

    explicit operator T() const
    {
        static_assert(ROW_COUNT == 1 && COLUMN_COUNT == 1, "Can only convert to scalar if it is 1x1 matrix");
        return at(0);
    }

    /// Set a single column of the matrix.
    CUDA_BOTH_INLINE
    void setColumn(size_t const idx, TColumn const& value)
    {
        // TODO(danielh): this reinterpret allows setting an entire column and enables vectorized operations in nvcc
        //                However this leads the compiler to optimize the code wrong due to breakage of the strict aliasing rule
        //                The resulting code will not be correct. See commit: a72e4aabf94bebb06959c30f0efd81ae1c31115a
        // code-comment " auto &columnData = *reinterpret_cast<TColumn*>(m_data + ROW_COUNT*idx);"
        // code-comment " columnData = value;"

        // Assign each value, will not be vectorized by nvcc
        for (size_t j = 0; j < ROW_COUNT; j++)
        {
            atRef(j, idx) = value[j];
        }
    }

    /// Return a single column of the matrix.
    CUDA_BOTH_INLINE
    auto getColumn(size_t const idx) const -> TColumn
    {
        TColumn res;
        for (size_t j = 0; j < ROW_COUNT; j++)
        {
            res[j] = atInternal(j, idx);
        }
        return res;
    }

    /// Set a single row of the matrix.
    CUDA_BOTH_INLINE
    void setRow(size_t idx, TRow const& value)
    {
        // Assign each value, will not be vectorized by nvcc
        for (size_t j = 0; j < COLUMN_COUNT; j++)
        {
            atRef(idx, j) = value[j];
        }
    }

    /// Return a single row of the matrix.
    CUDA_BOTH_INLINE
    auto getRow(size_t idx) const -> TRow
    {
        TRow res;
        for (size_t j = 0; j < COLUMN_COUNT; j++)
        {
            res[j] = atInternal(idx, j);
        }
        return res;
    }

    /// Return a modifiable reference to an element.
    /// Only elements that are stored can be obtained like this.
    CUDA_BOTH_INLINE
    auto atRef(size_t const row, size_t const col) -> T&
    {
        // this call will bounds check row & col
        // if it returns, it's in bounds, so no additional bounds test needed
        size_t const idx = TStorage::indexFromRowCol(row, col);

#if DW_RUNTIME_CHECKS()
        if (!TStorage::isElementStored(row, col))
        {
#if !defined(__CUDACC__)
            throwMatrixIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>(
                "BaseMatrix::atRef(): Tried to get reference of element outside of valid storage");
#endif
        }
#endif
        // TODO(dwplc): FP - the index never outside of matrix
        // coverity[cert_arr30_c_violation]
        // coverity[cert_str31_c_violation]
        // coverity[cert_str50_cpp_violation]
        // coverity[overrun-call]
        return m_data[idx];
    }

    CUDA_BOTH_INLINE
    auto operator()(size_t const row, size_t const col) const -> T
    {
        return atInternal(row, col);
    }

    /// Return a single element of the matrix.
    CUDA_BOTH_INLINE
    auto at(size_t const idx) -> T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= ELEMENT_COUNT)
        {
#if !defined(__CUDACC__)
            throwMatrixIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("BaseMatrix::at(): Index out of bounds");
#endif
        }
#endif
        return m_data[idx];
    }

    CUDA_BOTH_INLINE
    auto at(size_t const idx) const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        if (idx >= ELEMENT_COUNT)
        {
#if !defined(__CUDACC__)
            throwMatrixIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>("BaseMatrix::at(): Index out of bounds");
#endif
        }
#endif
        return m_data[idx];
    }

    CUDA_BOTH_INLINE
    auto operator()(size_t const row) const -> const T&
    {
        return at(row);
    }

    CUDA_BOTH_INLINE
    auto operator()(size_t const row) -> T&
    {
        return at(row);
    }

    CUDA_BOTH_INLINE
    auto operator[](size_t const idx) const -> const T&
    {
#if DW_RUNTIME_CHECKS()
        return at(idx);
#else
        return m_data[idx];
#endif
    }

    CUDA_BOTH_INLINE
    auto operator[](size_t const idx) -> T&
    {
#if DW_RUNTIME_CHECKS()
        return at(idx);
#else
        return m_data[idx];
#endif
    }

    /// Copy data from another matrix. Same as the assignment operator.
    template <class To, class TSo>
    CUDA_BOTH_INLINE void copyFrom(const BaseMatrix<To, ROW_COUNT, COLUMN_COUNT, TSo>& other)
    {
        this->operator=(other);
    }

    CUDA_BOTH_INLINE
    void copyFrom(const T* buffer)
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            m_data[i] = buffer[i];
        }
    }

    /// Copy the matrix data to a buffer
    CUDA_BOTH_INLINE
    void copyTo(T* buffer)
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            buffer[i] = m_data[i];
        }
    }

    CUDA_BOTH_INLINE auto data() -> T* { return m_data.data(); }
    CUDA_BOTH_INLINE auto data() const -> const T* { return m_data.data(); }

    CUDA_BOTH_INLINE constexpr size_t size() const { return ELEMENT_COUNT; }

    CUDA_BOTH_INLINE auto begin() -> T* { return m_data.begin(); }
    CUDA_BOTH_INLINE auto begin() const -> const T* { return m_data.begin(); }

    CUDA_BOTH_INLINE auto end() -> T* { return m_data.end(); }
    CUDA_BOTH_INLINE auto end() const -> const T* { return m_data.end(); }

    template <size_t SubRows, size_t SubCols = COLUMN_COUNT, size_t StartRow = 0, size_t StartColumn = 0>
    CUDA_BOTH_INLINE auto
    getBlock() const -> Matrix<T, SubRows, SubCols>
    {
        static_assert(SubCols + StartColumn <= COLUMN_COUNT, "Column count incorrect");
        static_assert(SubRows + StartRow <= ROW_COUNT, "Row count incorrect");

        Matrix<T, SubRows, SubCols> res;
        for (size_t i = 0; i < SubCols; i++)
        {
            Vector<T, SubRows> ci;
            Vector<T, ROW_COUNT> const& ai = getColumn(StartColumn + i);
            for (size_t j = 0; j < SubRows; j++)
            {
                ci[j] = ai[StartRow + j];
            }
            res.setColumn(i, ci);
        }
        return res;
    }

    /// Set a block [rows,cols] inside the matrix [N,M], with rows <= N and cols <= M.
    template <size_t SubRows, size_t SubCols = COLUMN_COUNT, size_t StartRow = 0, size_t StartColumn = 0>
    CUDA_BOTH_INLINE void setBlock(const Matrix<T, SubRows, SubCols>& b)
    {
        static_assert(SubCols + StartColumn <= COLUMN_COUNT, "Column count incorrect");
        static_assert(SubRows + StartRow <= ROW_COUNT, "Row count incorrect");

        // TODO(danielh): this may not get vectorized because a full column is not assigned
        for (size_t i = 0; i < SubCols; i++)
        {
            auto bi = b.getColumn(i);
            for (size_t j = 0; j < SubRows; j++)
            {
                atRef(StartRow + j, StartColumn + i) = bi[j];
            }
        }
    }

    /// Return the first N rows of the matrix.
    template <size_t N>
    CUDA_BOTH_INLINE auto getTopRows() const -> Matrix<T, N, COLUMN_COUNT>
    {
        return getBlock<N, COLUMN_COUNT>();
    }

    /// Set the first N rows of the matrix.
    template <size_t N>
    CUDA_BOTH_INLINE void setTopRows(const Matrix<T, N, COLUMN_COUNT>& value)
    {
        setBlock<N, COLUMN_COUNT>(value);
    }

    /// Return the last N rows of the matrix.
    template <size_t N>
    CUDA_BOTH_INLINE auto getBottomRows() const -> Matrix<T, N, COLUMN_COUNT>
    {
        return getBlock<N, COLUMN_COUNT, ROW_COUNT - N>();
    }

    /// Set the last N rows of the matrix.
    template <size_t N>
    CUDA_BOTH_INLINE void setBottomRows(const Matrix<T, N, COLUMN_COUNT>& value)
    {
        setBlock<N, COLUMN_COUNT, ROW_COUNT - N>(value);
    }

    /// Return the first M cols of the matrix, from left to right.
    template <size_t M>
    CUDA_BOTH_INLINE auto getLeftColumns() const -> Matrix<T, ROW_COUNT, M>
    {
        return getBlock<ROW_COUNT, M>();
    }

    /// Set the first M cols of the matrix, from left to right.
    template <size_t M>
    CUDA_BOTH_INLINE void setLeftColumns(const Matrix<T, ROW_COUNT, M>& value)
    {
        setBlock<ROW_COUNT, M>(value);
    }

    /// Return the last M cols of the matrix, from left to right.
    template <size_t M>
    CUDA_BOTH_INLINE auto getRightColumns() const -> Matrix<T, ROW_COUNT, M>
    {
        return getBlock<ROW_COUNT, M, 0, COLUMN_COUNT - M>();
    }

    /// Set the last M cols of the matrix, from left to right.
    template <size_t M>
    CUDA_BOTH_INLINE void setRightColumns(const Matrix<T, ROW_COUNT, M>& value)
    {
        setBlock<ROW_COUNT, M, 0, COLUMN_COUNT - M>(value);
    }

    /// Return the top-left [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE auto getTopLeftCorner() const -> Matrix<T, N, M>
    {
        return getBlock<N, M>();
    }

    /// Set the top-left [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE void setTopLeftCorner(const Matrix<T, N, M>& value)
    {
        setBlock<N, M>(value);
    }

    /// Return the top-right [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE auto getTopRightCorner() const -> Matrix<T, N, M>
    {
        return getBlock<N, M, 0, COLUMN_COUNT - M>();
    }

    /// Set the top-right [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE void setTopRightCorner(const Matrix<T, N, M>& value)
    {
        setBlock<N, M, 0, COLUMN_COUNT - M>(value);
    }

    /// Return the bottom-right [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE auto getBottomRightCorner() const -> Matrix<T, N, M>
    {
        return getBlock<N, M, ROW_COUNT - N, COLUMN_COUNT - M>();
    }

    /// Set the bottom-right [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE void setBottomRightCorner(const Matrix<T, N, M>& value)
    {
        setBlock<N, M, ROW_COUNT - N, COLUMN_COUNT - M>(value);
    }

    /// Return the bottom-left [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE auto getBottomLeftCorner() const -> Matrix<T, N, M>
    {
        return getBlock<N, M, ROW_COUNT - N, 0>();
    }

    /// Set the bottom-left [N, M] matrix block.
    template <size_t N, size_t M>
    CUDA_BOTH_INLINE void setBottomLeftCorner(const Matrix<T, N, M>& value)
    {
        setBlock<N, M, ROW_COUNT - N, 0>(value);
    }

    /// Return the sum of squares of each element of the matrix.
    CUDA_BOTH_INLINE
    auto squaredNorm() const -> T
    {
        T res{};
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            res += m_data[i] * m_data[i];
        }
        return res;
    }

    /// Return the square root of the sum of squares of each element of the matrix.
    CUDA_BOTH_INLINE
    auto norm() const -> T
    {
#if defined(__CUDA_ARCH__)
        // device: uses sqrt() intrinsic
        return sqrt(squaredNorm());
#else
        // host: uses safeSqrt() and may raise an exception
        // the result of squaredNorm must be non-negative
        using std::sqrt;
        // TODO(dwplc): FP - this function is called on non floating point data. Masking off for this to be addressed separately."
        // coverity[autosar_cpp14_a0_4_4_violation]
        // coverity[cert_flp32_c_violation]
        return sqrt(squaredNorm());
#endif
    }

    /// Return the diagonal vector of the matrix.
    static constexpr size_t DIAGONAL_SIZE = util::min(ROW_COUNT, COLUMN_COUNT);
    CUDA_BOTH_INLINE
    Vector<T, DIAGONAL_SIZE> diagonal() const
    {
        Vector<T, DIAGONAL_SIZE> res;
        for (size_t i = 0; i < DIAGONAL_SIZE; i++)
        {
            res[i] = atInternal(i, i);
        }
        return res;
    }

    /// Return the sum of diagonal elements of the matrix.
    CUDA_BOTH_INLINE
    auto trace() const -> T
    {
        static_assert(ROW_COUNT == COLUMN_COUNT, "Trace is defined only for square matrices");

        Vector<T, DIAGONAL_SIZE> diag = diagonal();
        T tr = 0;
        for (size_t i = 0; i < DIAGONAL_SIZE; i++)
        {
            tr += diag[i];
        }

        return tr;
    }

    /// Return true is the matrix is Zero, otherwise false.
    CUDA_BOTH_INLINE
    bool isZero(T const tolerance = util::numeric_limits<T>::epsilon()) const
    {
        for (const auto value : m_data)
        {
            if (cuda::isnan(value) || std::abs(value) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    /// Return true is the matrix is an Identity matrix, otherwise false.
    CUDA_BOTH_INLINE
    bool isIdentity(T tolerance = util::numeric_limits<T>::epsilon()) const
    {
        for (size_t i = 0; i < COLUMN_COUNT; i++)
        {
            for (size_t j = 0; j < ROW_COUNT; j++)
            {
                T expected = (i == j) ? T(1) : T(0);
                if (std::abs(atInternal(j, i) - expected) > tolerance)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// Return true if the passed Matrix is equal to this matrix inside the passed tolerance.
    CUDA_BOTH_INLINE
    bool isApprox(const BaseMatrix& other, T const tolerance = util::numeric_limits<T>::epsilon()) const
    {
        const T* it1 = this->begin();
        const T* it2 = other.begin();
        while (it1 != this->end())
        {
            if (std::abs(*it1 - *it2) > tolerance)
            {
                return false;
            }
            it1++;
            it2++;
        }
        return true;
    }

    /// Return the maximum element of the Matrix.
    CUDA_BOTH_INLINE
    auto max() const -> T
    {
        T res = at(0);
        for (size_t i = 1; i < ELEMENT_COUNT; i++)
        {
            T value = at(i);
            if (res < value)
            {
                res = value;
            }
        }
        return res;
    }

    /// Return the minimum element of the Matrix.
    CUDA_BOTH_INLINE
    auto min() const -> T
    {
        T res = at(0);
        for (size_t i = 1; i < ELEMENT_COUNT; i++)
        {
            T value = at(i);
            if (res > value)
            {
                res = value;
            }
        }
        return res;
    }

    /// Return the maximum element in absolute value of the Matrix.
    CUDA_BOTH_INLINE
    auto maxAbs() const -> T
    {
        T res = std::abs(at(0));
        for (size_t i = 1; i < ELEMENT_COUNT; i++)
        {
            T value = std::abs(at(i));
            if (res < value)
            {
                res = value;
            }
        }
        return res;
    }

    /// Return the minimum element in absolute value of the Matrix.
    CUDA_BOTH_INLINE
    auto minAbs() const -> T
    {
        T res = std::abs(at(0));
        for (size_t i = 1; i < ELEMENT_COUNT; i++)
        {
            T value = std::abs(at(i));
            if (res > value)
            {
                res = value;
            }
        }
        return res;
    }

    /// Set all elements of the matrix with their absolute value.
    CUDA_BOTH_INLINE
    auto abs() const -> TDerived
    {
        TDerived res;
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            res.at(i) = std::abs(at(i));
        }
        return res;
    }

    /// Return the sum of all elements in the matrix.
    CUDA_BOTH_INLINE
    auto sum() const -> T
    {
        T res = at(0);
        for (size_t i = 1; i < ELEMENT_COUNT; i++)
        {
            res += at(i);
        }
        return res;
    }

    /// Return the sum of the absoute value of all elements in the matrix.
    CUDA_BOTH_INLINE
    auto l1Norm() const -> T
    {
        return abs().sum();
    }

    /// Normalize the matrix in place.
    CUDA_BOTH_INLINE
    void normalize()
    {
        T n = norm();
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            at(i) /= n;
        }
    }

    /// Return the normalized matrix, without changing the current matrix.
    CUDA_BOTH_INLINE
    auto normalized() const -> TDerived
    {
        T const n = norm();
        TDerived res;
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            res.at(i) = at(i) / n;
        }
        return res;
    }

    /// Return true if at least one element is Not a Number.
    CUDA_BOTH_INLINE
    bool hasNaN() const
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            if (cuda::isnan(at(i)))
            {
                return true;
            }
        }
        return false;
    }

    /// Return true if at least one element is infinite.
    CUDA_BOTH_INLINE
    bool hasInf() const
    {
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
        {
            if (cuda::isinf(at(i)))
            {
                return true;
            }
        }
        return false;
    }

    /// Return a copy of this matrix.
    CUDA_BOTH_INLINE auto matrix() const -> TDense
    {
        TDense m;
        for (size_t j = 0; j < COLUMN_COUNT; ++j)
        {
            for (size_t i = 0; i < ROW_COUNT; ++i)
            {
                m(i, j) = atInternal(i, j);
            }
        }
        return m;
    }

    template <class T2, typename = typename std::enable_if<!std::is_same<T, T2>::value>::type>
    CUDA_BOTH_INLINE auto cast() const -> typename detail::MatrixFromStorage<TStorage, T2>::type
    {
        using TMatrix2 = typename detail::MatrixFromStorage<TStorage, T2>::type;

        TMatrix2 res;
        for (size_t i = 0; i < TStorage::ELEMENT_COUNT; ++i)
        {
            res[i] = static_cast<T2>(at(i));
        }
        return res;
    }

    /// Specialization to avoid cast when not needed.
    template <class T2, typename = typename std::enable_if<std::is_same<T, T2>::value>::type>
    CUDA_BOTH_INLINE auto cast() const -> TDerived const&
    {
        return static_cast<const TDerived&>(*this);
    }

protected:
    BaseMatrix()                  = default;
    BaseMatrix(const BaseMatrix&) = default;
    BaseMatrix(BaseMatrix&&)      = default;
    BaseMatrix& operator=(const BaseMatrix&) = default;
    BaseMatrix& operator=(BaseMatrix&&) = default;

    ~BaseMatrix() = default;

    CUDA_BOTH_INLINE
    explicit BaseMatrix(TStorage&& storage)
        : m_data(std::move(storage))
    {
    }

    CUDA_BOTH_INLINE
    explicit BaseMatrix(span<const T, 1> const dataIn)
        : m_data()
    {
#if DW_RUNTIME_CHECKS()
        if (dataIn.size() != ELEMENT_COUNT)
        {
            assertException<OutOfBoundsException>(
                "BaseMatrix(): Wrong number of matrix elements provided in span");
        }
#endif
        for (size_t i = 0; i < ELEMENT_COUNT; ++i)
        {
            m_data[i] = dataIn[i];
        }
    }

    CUDA_BOTH_INLINE
    explicit BaseMatrix(StaticSpan<const T, ELEMENT_COUNT> const dataIn)
        : m_data()
    {
        for (size_t i = 0; i < ELEMENT_COUNT; ++i)
        {
            m_data[i] = dataIn[i];
        }
    }

    /// Forces matrix m into the current matrix,
    /// copying only the coefficients that are stored.
    template <class TStorage2>
    CUDA_BOTH_INLINE void coerceInternal(const BaseMatrix<T, ROW_COUNT, COLUMN_COUNT, TStorage2>& other)
    {
        for (size_t col = 0; col < COLUMN_COUNT; ++col)
        {
            for (size_t row = col; row < ROW_COUNT; ++row)
            {
                if (TStorage::isElementStored(row, col))
                {
                    atRef(row, col) = other(row, col);
                }
            }
        }
    }

    /// Return the value of the matrix at the given (row, col)
    /// If the element is not stored, T(0) is returned
    /// (e.g., non-diagonal elements in a diagonal matrix).
    /// It only returns by-value because non-stored elements
    /// do not have a reference to return.
    ///
    /// To modify values use atRef() or the derived class at().
    CUDA_BOTH_INLINE
    auto atInternal(size_t const row, size_t const col) const -> T
    {
#if DW_RUNTIME_CHECKS()
        if ((row >= ROW_COUNT) || (col >= COLUMN_COUNT))
        {
#if !defined(__CUDACC__)
            throwMatrixIndexOutOfBounds();
#else
            assertException<OutOfBoundsException>(
                "BaseMatrix::atInternal(): Tried to access element outside of matrix");
#endif
        }
#endif
        if (!TStorage::isElementStored(row, col))
        {
            return T{};
        }
        else
        {
            size_t const idx = TStorage::indexFromRowCol(row, col);
            // TODO(dwplc): FP - the index never outside of matrix
            // coverity[cert_arr30_c_violation]
            // coverity[cert_str31_c_violation]
            // coverity[cert_str50_cpp_violation]
            // coverity[overrun-call]
            return m_data[idx];
        }
    }

private:
    TStorage m_data;
};

// See forward define for documentation
template <class T, size_t ROW_COUNT, size_t COLUMN_COUNT>
class Matrix : public BaseMatrix<T, ROW_COUNT, COLUMN_COUNT, detail::DenseMatrixStorage<T, ROW_COUNT, COLUMN_COUNT>>
{
public:
    using TStorage = detail::DenseMatrixStorage<T, ROW_COUNT, COLUMN_COUNT>;
    using Base     = BaseMatrix<T, ROW_COUNT, COLUMN_COUNT, TStorage>;
    using Base::at;
    using Base::getBlock;
    using Base::IS_VECTOR;
    using Base::operator();

    Matrix()  = default;
    ~Matrix() = default;

    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;

    /// Move constructor.
    CUDA_BOTH_INLINE
    Matrix(Matrix&& other)
        : Base(std::move(other))
    {
    }

    /// Copy-constructor.
    CUDA_BOTH_INLINE
    Matrix(const Matrix& other)
        : Base(other)
    {
    }

    /// Construct from span.
    CUDA_BOTH_INLINE
    explicit Matrix(dw::core::span<const T, 1> data_)
        : Base(data_)
    {
    }

    /// Construct from StaticSpan.
    CUDA_BOTH_INLINE
    explicit Matrix(dw::core::StaticSpan<const T, Base::ELEMENT_COUNT> data_)
        : Base(data_)
    {
    }

    /// Construct vector (single element).
    /// Note: requires a template so the static_assert is only evaluated when used.
    template <typename = void>
    CUDA_BOTH_INLINE explicit Matrix(T xIn)
    {
        static_assert(Base::ELEMENT_COUNT == 1, "Cannot use constructor with less elements than matrix");
        at(0) = xIn;
    }

    /// Construct vector (2 elements).
    /// Note: requires a template so the static_assert is only evaluated when used.
    ///
    /// We keep separate types for the inputs (Tx, Ty) instead of a single type to allow
    /// implicit conversions. The compiler will automatically flag potentially dangerous conversions.
    /// For example:
    ///     Vector2ui a = {uint8_t(0),  uint32_t(1)}; // Valid implicit type upgrade
    ///     Vector2ui b = {uint64_t(0), uint32_t(1)}; // Error: implicit conversion might overflow
    ///     Vector2f  c = {1.0, 1.0f}; // Error: first argument is double and precision might be reduced
    ///
    template <typename Tx, typename Ty>
    CUDA_BOTH_INLINE
    Matrix(Tx const xIn, Ty const yIn)
    {
        static_assert(Base::ELEMENT_COUNT == 2, "Cannot use constructor with less elements than matrix");
        static_assert(Base::IS_VECTOR, "Use initializer list constructor for matrices");
        at(0) = xIn;
        at(1) = yIn;
    }

    /// Construct vector (3 elements)
    /// See Matrix::Matrix(Tx, Ty) for details on why the types are separate.
    template <typename Tx, typename Ty, typename Tz>
    CUDA_BOTH_INLINE
    Matrix(Tx const xIn, Ty const yIn, Tz const zIn)
    {
        static_assert(Base::ELEMENT_COUNT == 3, "Cannot use constructor with less elements than matrix");
        static_assert(Base::IS_VECTOR, "Use initializer list constructor for matrices");
        at(0) = xIn;
        at(1) = yIn;
        at(2) = zIn;
    }

    /// Construct vector (4 elements)
    /// See Matrix::Matrix(Tx, Ty) for details on why the types are separate.
    template <typename Tx, typename Ty, typename Tz, typename Tw>
    CUDA_BOTH_INLINE
    Matrix(Tx const xIn, Ty const yIn, Tz const zIn, Tw const wIn)
    {
        static_assert(Base::ELEMENT_COUNT == 4, "Cannot use constructor with less elements than matrix");
        static_assert(IS_VECTOR, "Use initializer list constructor for matrices");
        at(0) = xIn;
        at(1) = yIn;
        at(2) = zIn;
        at(3) = wIn;
    }

    /// Construct vector (5 elements)
    /// See Matrix::Matrix(Tx, Ty) for details on why the types are separate.
    template <typename Tx, typename Ty, typename Tz, typename Tw, typename Tv>
    CUDA_BOTH_INLINE
    Matrix(Tx xIn, Ty yIn, Tz zIn, Tw wIn, Tv vIn)
    {
        static_assert(Base::ELEMENT_COUNT == 5, "Cannot use constructor with less elements than matrix");
        static_assert(IS_VECTOR, "Use initializer list constructor for matrices");
        at(0) = xIn;
        at(1) = yIn;
        at(2) = zIn;
        at(3) = wIn;
        at(4) = vIn;
    }

    /// Matrix Constructor from array.
    /// The array is expected in column-major because that matches the internal
    /// representation of the matrix
    /// E.g. T arr[] = {r00, r10, r20,  //First column
    ///                 r01, r11, r21,  //Second column
    ///                 r02, r12, r22}; //Third column
    ///      Matrix3f r = arr;
    template <std::size_t LN>
    CUDA_BOTH_INLINE explicit Matrix(const T (&list)[LN])
    {
        static_assert(LN == Base::ELEMENT_COUNT, "Array must be the same size as the matrix");

        size_t idx = 0;
        for (size_t i = 0; i < COLUMN_COUNT; i++)
        {
            for (size_t j = 0; j < ROW_COUNT; j++)
            {
                at(j, i) = list[idx++];
            }
        }
    }

    /// Matrix Constructor from initializer list.
    /// The array is expected in row-major because that is how literals are written
    /// E.g. Matrix3f r = {{r00, r01, r02,   // First row
    ///                     r10, r11, r12,   // Second row
    ///                     r20, r21, r22}}; // Third row
    ///
    /// This constructor takes care of things like Vector3f{}, Vector3f{1.f}, and Vector3f{1.f,2.f,3.f}
    ///
    /// If the initializer list is smaller than the number of elements, the rest are initialized to zero.
    CUDA_BOTH_INLINE
    Matrix(std::initializer_list<T> const list)
    {
        if (list.size() > Base::ELEMENT_COUNT)
        {
            dw::core::assertException<dw::core::InvalidArgumentException>("Matrix: too many items in initializer");
        }

        std::size_t i = 0;
        std::size_t j = 0;
        for (T const value : list)
        {
            at(j, i) = value;
            if (COLUMN_COUNT == 1)
            {
                // TODO(dwplc): FP - j is 0-based so it isn't wrapped even in extreme case (list.size = SIZE_MAX)
                // coverity[cert_int30_c_violation]
                j++;
            }
            else
            {
                i++;
                if (i >= COLUMN_COUNT)
                {
                    i = 0;
                    j++;
                }
            }
        }
        if (list.size() == Base::ELEMENT_COUNT)
        {
            return;
        }

        // The rest of the values are initialized to zero
        for (; j < ROW_COUNT; ++j)
        {
            for (; i < COLUMN_COUNT; ++i)
            {
                at(j, i) = T{};
            }
            i = 0;
        }
    }

    static CUDA_BOTH_INLINE auto
    Zero() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res{};
        res.setZero();
        return res;
    }

    static CUDA_BOTH_INLINE auto
    Ones() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res{};
        res.setOnes();
        return res;
    }

    static CUDA_BOTH_INLINE constexpr auto
    Identity() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res{};
        res.setIdentity();
        return res;
    }

    static CUDA_BOTH_INLINE auto
    UnitX() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res;
        res.setZero();
        res.x() = T(1);
        return res;
    }

    static CUDA_BOTH_INLINE auto
    UnitY() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res;
        res.setZero();
        res.y() = T(1);
        return res;
    }

    static CUDA_BOTH_INLINE auto
    UnitZ() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res{};
        res.setZero();
        res.z() = T(1);
        return res;
    }

    static CUDA_BOTH_INLINE auto
    UnitW() -> Matrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        Matrix res;
        res.setZero();
        res.w() = T(1);
        return res;
    }

    CUDA_BOTH_INLINE auto toMatrix() const -> const Matrix&
    {
        return *this;
    }
    /// If Matrix is a vector, return the first element.
    template <class = void>
    CUDA_BOTH_INLINE auto x() const -> const T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 1, "not enough elements for accessor");
        return at(0);
    }

    template <class = void>
    CUDA_BOTH_INLINE auto x() -> T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 1, "not enough elements for accessor");
        return at(0);
    }

    /// If Matrix is a vector, return the second element.
    template <class = void>
    CUDA_BOTH_INLINE auto y() const -> const T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 2, "not enough elements for accessor");
        return at(1);
    }

    template <class = void>
    CUDA_BOTH_INLINE auto y() -> T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 2, "not enough elements for accessor");
        return at(1);
    }

    /// If Matrix is a vector, returns the third element.
    template <class = void>
    CUDA_BOTH_INLINE auto z() const -> const T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 3, "not enough elements for accessor");
        return at(2);
    }

    template <class = void>
    CUDA_BOTH_INLINE auto z() -> T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 3, "not enough elements for accessor");
        return at(2);
    }

    /// If Matrix is a vector, returns the fourth element.
    template <class = void>
    CUDA_BOTH_INLINE auto w() const -> const T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 4, "not enough elements for accessor");
        return at(3);
    }

    template <class = void>
    CUDA_BOTH_INLINE auto w() -> T&
    {
        static_assert(ROW_COUNT == 1 || COLUMN_COUNT == 1, "xyzw accessor only valid for vectors");
        static_assert(Base::ELEMENT_COUNT >= 4, "not enough elements for accessor");
        return at(3);
    }

    /// Return the transpose of this matrix.
    CUDA_BOTH_INLINE
    auto transpose() const -> dw::core::Matrix<T, COLUMN_COUNT, ROW_COUNT>
    {
        return matrixTranspose(*this);
    }

    /// Return a homogenous version of the matrix/vector
    /// For column vectors, it appens a one at the end.
    /// For matrices, it appens [0 0 ... 0 1] as a new row
    /// Row vectors are treated as matrices.
    CUDA_BOTH_INLINE
    auto homogeneous() const -> dw::core::Matrix<T, ROW_COUNT + 1, COLUMN_COUNT>
    {
        dw::core::Matrix<T, ROW_COUNT + 1, COLUMN_COUNT> res;
        res.template setBlock<ROW_COUNT, COLUMN_COUNT>(*this);
        for (size_t i = 0; i < COLUMN_COUNT; i++)
        {
            res(ROW_COUNT, i) = (i == COLUMN_COUNT - 1) ? T(1) : T(0);
        }
        return res;
    }

    CUDA_BOTH_INLINE
    auto hnormalized() const -> dw::core::Matrix<T, ROW_COUNT - 1, COLUMN_COUNT>
    {
        T factor = at(ROW_COUNT - 1, COLUMN_COUNT - 1);
        return Base::template getBlock<ROW_COUNT - 1, COLUMN_COUNT>() / factor;
    }

    template <class To>
    CUDA_BOTH_INLINE auto dot(const Matrix<To, ROW_COUNT, COLUMN_COUNT>& other) const -> T
    {
        return dw::core::dot(*this, other);
    }

    CUDA_BOTH_INLINE
    auto at(std::size_t row, std::size_t col) const -> T
    {
        return Base::atInternal(row, col);
    }

    CUDA_BOTH_INLINE
    auto at(std::size_t const row, std::size_t const col) -> T&
    {
        return Base::atRef(row, col);
    }

    CUDA_BOTH_INLINE
    auto operator()(std::size_t const row, std::size_t const col) -> T&
    {
        return at(row, col);
    }

    /// Set the dimension of the matrix while keeping the internal representation constant.
    /// This is useful to reshape a 3x3 matrix into a linear 9x1 vector or vice-versa.
    template <std::size_t R2, std::size_t C2>
    auto reshape() const -> dw::core::Matrix<T, R2, C2>
    {
        static_assert(R2 * C2 == Base::ELEMENT_COUNT, "Cannot change the element count when reshaping");
        dw::core::Matrix<T, R2, C2> res{};
        for (std::size_t i = 0; i < Base::ELEMENT_COUNT; i++)
        {
            res.at(i) = this->at(i);
        }
        return res;
    }

    /// Reshape the matrix into a linear vector.
    dw::core::Vector<T, ROW_COUNT * COLUMN_COUNT> reshapeToVector() const
    {
        return this->reshape<ROW_COUNT * COLUMN_COUNT, 1>();
    }

private:
    template <std::size_t R2, std::size_t C2>
    explicit Matrix(const detail::DenseMatrixStorage<T, R2, C2>& other)
        : Base(TStorage{other})
    {
    }
};
} // namespace core
} // namespace dw

#undef MATRIX_MAKE_TYPEDEFS_ALL_SIZES

#endif // DW_CORE_MATRIX_BASEMATRIX_HPP_
