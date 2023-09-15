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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_MATRIX_SYMMETRICBANDEDMATRIX_HPP_
#define DW_CORE_MATRIX_SYMMETRICBANDEDMATRIX_HPP_

#include "../BaseMatrix.hpp"
#include <dwshared/dwfoundation/dw/core/container/Array.hpp>

namespace dw
{

namespace core
{

/// Forward def
template <class T, size_t DIM, size_t UB>
class SymmetricBandedMatrix;

// Aliases
using SymmetricBandedMatrix43f = SymmetricBandedMatrix<float32_t, 4, 3>;
using SymmetricBandedMatrix42f = SymmetricBandedMatrix<float32_t, 4, 2>;

namespace detail
{

////////////////////////////////////////////////////////////////////////
// SymmetricBandedMatrixStorage
// Storage class for class to represent a symmetric banded matrix, storing only the
// banded portion.
//
// For example, a matrix of dimension 5 and upper-bandwidth 3 such as
// 10  2   3   0   0
// 2   20  4   5   0
// 3   4   30  6   7
// 0   5   6   40  8
// 0   0   7   8   50
//
// would be represented as
// 10  2  3
// 20  4  5
// 30  6  7
// 40  8  0
// 50  0  0
//
// storing only 5 x 3 entries in column-major fashion, i.e. with the following memory layout
// 0   5   10  x   x
// 5   1   6   11  x
// 10  6   2   7   12
// x   11  7   3   8  (13)
// x   x   12  8   4  (9) (14)
//
// Note: the indices in parentheses (9), (13) and (14) store padded zeroes that don't correspond to anything from the
// original representation.
// The reason for padded zeroes is simplification of indexing. This storage design is not wasteful compared to the
// one used for symmetric matrices as long as UB is less than N/2.
// TODO(victzhang) Potentially revise the storage design to not include any padded zeroes https://jirasw.nvidia.com/browse/AVPC-3793
////////////////////////////////////////////////////////////////////////
template <class T, size_t DIM, size_t UB>
struct SymmetricBandedMatrixStorage
{
    using Scalar                          = T;
    static constexpr size_t ELEMENT_COUNT = DIM * UB; //
    static constexpr size_t ALIGNMENT     = detail::AlignmentHelper<T, ELEMENT_COUNT>::VALUE;

    /// Indicates if the element is stored
    CUDA_BOTH_INLINE
    static constexpr bool isElementStored(size_t const row, size_t const col)
    {
        if ((row < DIM) && (col < DIM))
        {
            return (col < row + UB) && (row < col + UB);
        }
        else
        {
            return false;
        }
    }

    /// Converts the 2d index into a linear index inside dataArray
    CUDA_BOTH_INLINE
    static constexpr size_t indexFromRowCol(size_t const row, size_t const col)
    {
        return ((row >= col) ? indexFromRowColNormalized(row, col) : indexFromRowColNormalized(col, row));
    }

    /// Converts a linear index of the underlying data into a 2d {row,col} index of a normal matrix representation.
    /// Note:
    ///   {row,col} may be out of bounds due to the padded zeroes in the banded representation.
    ///   In the example given above, indices (9), (13) and (14) will result in NULLOPT.
    CUDA_BOTH_INLINE
    static Optional<dw::Array<size_t, 2>> rowColFromIndex(size_t const index)
    {
        size_t const row = (index / DIM) + (index % DIM);
        if (row < DIM)
        {
            return Optional<dw::Array<size_t, 2>>({row, index % DIM});
        }
        return NULLOPT;
    }

    CUDA_BOTH_INLINE
    auto operator[](std::size_t const idx) -> T&
    {
        return m_dataArray[idx];
    }

    CUDA_BOTH_INLINE
    auto operator[](std::size_t const idx) const -> const T&
    {
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
    CUDA_BOTH_INLINE
    static constexpr size_t indexFromRowColNormalized(size_t const row, size_t const col)
    {
        size_t idx{0};
        if ((col < DIM) && (row < DIM))
        {
            idx = (row - col) * DIM + col;
        }
        else
        {
            assertException<OutOfBoundsException>(
                "SymmetricBandedMatrix::indexFromRowColNormalized(): Tried to access element outside of matrix");
        }
        return idx;
    }
};

} // namespace detail

////////////////////////////////////////////////////////////////////////
// SymmetricBandedMatrix
// Matrix class to represent a symmetric banded matrix, storing only the
// banded portion.
// NOTE: See documentation of SymmetricBandedMatrixStorage for how the
// memory is arranged.
////////////////////////////////////////////////////////////////////////
template <class T, size_t DIM_, size_t UB_>
class SymmetricBandedMatrix : public BaseMatrix<T, DIM_, DIM_, detail::SymmetricBandedMatrixStorage<T, DIM_, UB_>>
{
public:
    static constexpr std::size_t DIM = DIM_;
    static constexpr std::size_t UB  = UB_;
    static_assert(DIM > 0, "Cannot define a matrix with zero size");
    static_assert(UB <= DIM, "Cannot define a symmetric banded matrix whose upper bandwidth is larger than the dimension");

    using TStorage = detail::SymmetricBandedMatrixStorage<T, DIM, UB>;
    using Base     = BaseMatrix<T, DIM, DIM, TStorage>;
    using TDense   = typename Base::TDense;

    using Base::operator();

    SymmetricBandedMatrix()                             = default;
    SymmetricBandedMatrix(const SymmetricBandedMatrix&) = default;
    SymmetricBandedMatrix(SymmetricBandedMatrix&&)      = default;
    SymmetricBandedMatrix& operator=(const SymmetricBandedMatrix&) = default;
    SymmetricBandedMatrix& operator=(SymmetricBandedMatrix&&) = default;
    ~SymmetricBandedMatrix()                                  = default;

    /// Prevent implicit conversion to construction from span to happen with matrices other than symmetric
    template <typename OtherStorage>
    SymmetricBandedMatrix(const BaseMatrix<T, DIM, DIM, OtherStorage>&) = delete;

    /// Construct from span while enforcing that padded zeroes are correctly placed
    CUDA_BOTH_INLINE
    explicit SymmetricBandedMatrix(dw::core::span<const T, 1> data_)
    {
#if DW_RUNTIME_CHECKS()
        if (data_.size() != Base::ELEMENT_COUNT)
        {
            dw::core::assertException<dw::core::OutOfBoundsException>(
                "SymmetricBandedMatrix(): Wrong number of matrix elements provided in span");
        }
#endif
        for (std::size_t i = 0U; i < data_.size(); ++i)
        {
            if ((i / DIM) + (i % DIM) >= DIM) // if out of band, forcibly set value to zero
            {
                Base::at(i) = static_cast<T>(0.0);
            }
            else // if within band, store value
            {
                Base::at(i) = data_[i];
            }
        }
    }

    /// Forces a normal matrix to become a symmetric banded matrix
    /// It takes only the lower-triangular part and ignores the upper-triangle
    template <class TStorage2>
    CUDA_BOTH_INLINE static auto coerce(const BaseMatrix<T, DIM, DIM, TStorage2>& other) -> SymmetricBandedMatrix
    {
        SymmetricBandedMatrix res;
        res.setZero();
        for (uint32_t col{0U}; col < DIM; ++col)
        {
            for (uint32_t row{col}; row < DIM; ++row)
            {
                if (TStorage::isElementStored(row, col)) // only store elements that are within the band
                {
                    res.atRef(row, col) = other(row, col);
                }
            }
        }
        return res;
    }

    CUDA_BOTH_INLINE auto toMatrix() const -> TDense
    {
        return Base::matrix();
    }

    static CUDA_BOTH_INLINE auto Zero() -> SymmetricBandedMatrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        SymmetricBandedMatrix m;
        m.setZero();
        return m;
    }

    static CUDA_BOTH_INLINE auto Identity() -> SymmetricBandedMatrix // clang-tidy NOLINT(readability-identifier-naming)
    {
        SymmetricBandedMatrix m;
        m.setIdentity();
        return m;
    }
};

} // namespace core
} // namespace dw

#endif // DW_CORE_MATRIX_SYMMETRICBANDEDMATRIX_HPP_
