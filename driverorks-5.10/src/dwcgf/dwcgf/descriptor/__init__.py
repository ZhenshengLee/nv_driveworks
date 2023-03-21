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
"""Package descriptor."""
# following imports not only exposes all names to descriptor
# also register Descriptor implementations
from dwcgf.descriptor.descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.descriptor_factory import *  # noqa: F401, F403
from dwcgf.descriptor.app_config_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.application_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.extra_info_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.node_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.graphlet_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.required_sensors_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.schedule_definition import *  # noqa: F401, F403
from dwcgf.descriptor.transformation_descriptor import *  # noqa: F401, F403
from dwcgf.descriptor.descriptor_loader import *  # noqa: F401, F403
