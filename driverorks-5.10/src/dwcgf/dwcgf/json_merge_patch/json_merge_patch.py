#########################################################################################
# This code contains NVIDIA Confidential Information and is disclosed
# under the Mutual Non-Disclosure Agreement.
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
# NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
# THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
# NVIDIA Corporation assumes no responsibility for the consequences of use of such
# information or for any infringement of patents or other rights of third parties that may
# result from its use. No license is granted by implication or otherwise under any patent
# or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied.
# NVIDIA Corporation products are not authorized for use as critical
# components in life support devices or systems without express written approval of
# NVIDIA Corporation.
#
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# rights in and to this software and related documentation and any modifications thereto.
# Any use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation is
# strictly prohibited.
#
#########################################################################################
"""JSON merge patch implementation."""
import copy


def merge(target, *patches):
    """JSON merge patch.

    @param target target JSON object
    @param patches patches apply to the target

    Patches are deep copied
    """
    for patch in patches:
        target = _merge_single(target, patch)
    return target


def _merge_single(target, patch):
    """Json merge patch implementation.

    Only a single patch is accepted.
    """
    patch = copy.deepcopy(patch)
    if not isinstance(patch, dict):
        return patch
    if not isinstance(target, dict):
        return _merge_single({}, patch)
    for key, value in patch.items():
        if key in target:
            if value is None:
                del target[key]
            else:
                target[key] = _merge_single(target[key], patch[key])
        else:
            if value is not None:
                target[key] = _merge_single({}, value)
    return target
