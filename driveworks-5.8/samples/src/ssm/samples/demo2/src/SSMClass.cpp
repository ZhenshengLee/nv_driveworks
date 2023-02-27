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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <SSMClass.hpp>

static int a = 0;

namespace SystemStateManager
{
namespace SM2
{

void SSMClass::printChildStates()
{

    if (get_SM1State(this) == SystemStateManager::SM2::SM1States::A1)
    cout << "     Sharat######## SM1: A1"<<  endl;
    cout << "     ############## SM2: " << getCurrentState("SM2") << endl;
    cout << "     ############## SSM: " << getCurrentState() << endl;
}

void SSMClass::Standby()
{
    printChildStates();
    Notification n;
    if (getNextNotification(n))
    {
        cout << "=================== " << n.receivedFrom << " : " << n.arg << endl;
        changeStateWithPayload("NormalOperation", "SM2", 99);
    }
    else
    {
        cout << "SSM in Standby..." << endl;
    }
    a++;
}

void SSMClass::NormalOperation()
{
    printChildStates();
    SystemStateManager::StateUpdate su{};
    bool status = getCommandFromSSMChannel(su);
    if (su.command == INIT_RESET)
    {
        std::cout << "###### INIT RESET ######" << std::endl;
        resetSSM();
        a = 0;
    }

    if (a == 12)
    {
        cout << "===================" << endl;
        changeState(SSMStates::Degrade);
    }
    else
    {
        cout << "SSM in NormalOperation..." << endl;
    }
    a++;
}

void SSMClass::Degrade()
{
    printChildStates();
    if (a == 16)
    {
        cout << "===================" << endl;
        changeState(SSMStates::UrgentOperation);
    }
    else
    {
        cout << "SSM in Degraded..." << endl;
    }
    a++;
}

void SSMClass::UrgentOperation()
{
    printChildStates();
    cout << "SSM in Urgent..." << endl;
}
}
}