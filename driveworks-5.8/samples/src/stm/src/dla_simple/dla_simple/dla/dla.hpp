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

#ifndef _DLA_H_
#define _DLA_H_

#if VIBRANTE_PDK_DECIMAL < 6000400
#include <memory>
#include <vector>
#include <array>

#include "nvmedia_dla.h"
#include "nvmedia_core.h"

#include "tensor.hpp"

//! Dla class
//! Dla class abstract NvMediaDla APIs and provide functions to load loadable and
//! execute loadable with provided input data.

class Dla final
{
public:
    static std::unique_ptr<Dla> Create();

    ~Dla();

    NvMediaStatus Init(uint32_t dlaId, uint32_t numTasks);

    //! One Dla class can hold only one loadable.
    NvMediaStatus AddLoadable(uint32_t& loadableIndex);

    NvMediaStatus GetDesc(
        uint32_t loadableIndex,
        std::vector<NvMediaDlaTensorDescriptor>& vInputTensorDesc,
        std::vector<NvMediaDlaTensorDescriptor>& vOutputTensorDesc);

    NvMediaStatus DataRegister(
        uint32_t loadableIndex,
        Tensor* tensor);

    NvMediaStatus DataUnregister(
        uint32_t loadableIndex,
        Tensor* tensor);

    NvMediaStatus RemoveLoadable(uint32_t loadableIndex);

    NvMediaStatus Submit(
        uint32_t loadableIndex,
        std::vector<Tensor*>& vpInputTensor,
        std::vector<Tensor*>& vpOutputTensor);

    NvMediaDla* GetPtr() { return m_pDla; }

protected:
    NvMediaStatus PrintTensorDesc(NvMediaDlaTensorDescriptor* tensorDesc);

private:
    Dla(NvMediaDla* m_pDla);

    NvMediaDla* m_pDla;

    std::vector<NvMediaDlaLoadable*> m_vLoadables;

    static const std::size_t MAX_NUM_OF_DLA_DATA = 10;

    std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aInputDlaData;

    std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aOutputDlaData;
};

#endif
#endif // END OF _DLA_H_
