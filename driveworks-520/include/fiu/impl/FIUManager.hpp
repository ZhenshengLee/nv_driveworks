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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// FI Tool Version:    0.2.0
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef FIU_MANAGER_H
#define FIU_MANAGER_H
#include <fiu/impl/FIUCommon.hpp>
#include <fiu/impl/FIUClass.hpp>
#include <fiu/impl/ObjectManager.hpp>
#include <fiu/impl/JSONParser.hpp>
#include <fiu/impl/FIUIncludes.hpp>

namespace fiu2
{

class FIUManager
{
public:
    static FIUManager& getInstance()
    {
        static FIUManager instance;
        static bool isClientStarted{false};

        if (!isClientStarted)
        {
            isClientStarted = true;

            // Start the FIClient
            instance.startClient();
        }

        return instance;
    }

    /** FIUClass uses this API to add its "this" pointer to the FIUManager
     *
     *  @param[in] object Pointer to the FIUClass object
     *  @return true if the fault has been added to FIUManager's internal cache
     */
    bool addFIUObject(FIUClass* fiuObject);

    /** Returns if FIManager is active or not
     *
     *  @return true if FIManager object is enabled
     */
    bool isFIActive();

    /** Returns the total number of usable FIUCommand objects that can be used for new FIU commands
     *
     *  @return total number of FIUCommand objects free
     */
    uint32_t getFreeFIUCommandCount();

    /** Returns the total number of FIUCommands that can be stashed
     *
     *  @return total number faults that can be enabled whenever the Fault is available
     */
    uint32_t getEnabledFIUCommandStashCount();

    /** Returns Objet Manager object
     *
     *  @return Pointer to the object manager
     */
    ObjectManager* getObjectManager();

    /** API that FIClass uses to pass a used FIUCommand to the FIUManager
     *  so that it can be garbage collected.
     *
     * @param[in] object FIUCommand object that needs to be garbage collected
     * @return true if the object has been successfully garbage collected or not
     */
    bool reuseFIUCommandObject(FIUCommand* fiuCommandObj);

    /** Instructs FIUManager that it should verify the stashed cmd queue and
     *  enable the relevant commands. Also inputs the fault and instance within
     *  which an event was triggered (Added)
     *
     *  @param[in] faultName Name of the fault
     *  @param[in] instanceName Name of the instance
     *  @return true if the fault instances was enabled right away or not
     */
    bool enableStashedFIUCommands(const char* faultName, const char* instanceName);

    /**
     * API that attempts to enable the given fault immediately
     * Stashes the FIUCommand if it can't do so for different reasons
     *    - FIUClass has not been defined yet during init phase before main() executes
     *    - The instance has not been defined yet
     *    - Chained FIUs awaiting the above two conditions (FUTURE USE)
     *
     * @param[in] FIUCommand object
     * @return true if FIUManager was able to enable the fault immediately; false otherwise
     */
    bool enableFault(FIUCommand* fiuCommand);

    /**
     * Parses the JSON command, creates an FIU Command object and uses it to enable the FIU
     *
     * @param[in] command FIU Command in JSON string format
     * @return true if the JSON command is valid and has been accepted
     */
    bool parseJSONCommands(const nlohmann::json& command);

    /**
     * Parses the file that contains JSON commands
     *
     * @param[in] filename Path to the file that contains JSON commands
     * @return true if the file is valid and has been read successfully
     */
    bool parseFIUFile(const char* filename);

    /**
     * The following APIs permanently alter the FIUManager functionality
     * These APIs should be unsed for test purposes only.
     *
     * Note: These APIs is not exposed to the production code during the build process
     */
    void gtests_DeactivateFaultLibrary();
    void gtests_ClearFaultLibrary();
    void gtests_SetFaultLibraryActive();
    uint32_t gtests_getFaultCount();
    bool gtests_isFIUPresent(const char* name);
    FIUCommand* gtests_getReusableFIUCommand();
    void gtests_resetEnabledStashedFIUCommands();

private:
    /**
     * Destructor
     */
    ~FIUManager()
    {
        m_shutdownFlag = true;
        for (uint32_t index = 0; index < m_totalFIUCommandsSupported; index++)
        {
            delete m_fiuCommandArray[index];
        }
        delete[] m_fiuCommandArray;
        delete[] m_enabledFIUCommandsStash;
    }

    /**
     * Constructor
     */
    FIUManager()
    {
        m_fiuInstancePtr  = &FIUInstance::getInstance();
        m_fiuObjectMapPtr = FIClassMapPtr(new FIClassMap());
        m_objectManager.allocateMemoryForParams();
        m_jsonParserPtr = JSONParserPtr(new JSONParser);
        m_jsonParserPtr->setupFreeFIUCommandsList(m_totalFIUCommandsSupported);

        const char* filename = std::getenv(FI_FAULTS_LIST_FILE);
        parseFIUFile(filename);

        // Allocate FIUCommands
        m_fiuCommandArray         = new FIUCommand*[m_totalFIUCommandsSupported];
        m_enabledFIUCommandsStash = new FIUCommand*[m_totalFIUCommandsSupported];
        for (uint32_t index = 0; index < m_totalFIUCommandsSupported; index++)
        {
            m_fiuCommandArray[index] = new FIUCommand(&m_objectManager);
            m_fiuCommandArray[index]->setReuseFlag();
            m_jsonParserPtr->stashFreeFIUCommand(m_fiuCommandArray[index]);
            m_enabledFIUCommandsStash[index] = nullptr;
        }
    }

    /**
     * Add the fiuCommand object to the stash
     *
     * @param[in] *fiuCommand Pointer to the fiuCommand object
     * @return True if the fiu is stashed; else false
     */
    bool stashEnabledFIUCommand(FIUCommand* fiuCommand);

    /**
     * Getting function to get hold of the required FIUClass
     *
     * @param[in] *fiuName Pointer to the name of the fiu class
     * @param[in] *instanceName Pointer to the name of the fiu instance
     * @return FIUClass object pointer
     */
    FIUClass* getFIUClassObject(const char* faultName, const char* instanceName);

    /**
     * Starts the FI Client in a parallel thread
     */
    void startClient();

    /**
     * Getting function to get hold of the required FIUClass
     *
     * @param[in] *fiuName Pointer to the name of the fiu class
     * @return FIUClass object pointer
     */
    FIUClass* getFIUClassObject(const char* fiuName);

    /**
     * Disable the constructor
     *
     * @param[in] &FIUManager Reference to the FIUManager object
     */
    FIUManager(const FIUManager&) = delete;

    /**
     * Disable copy assignment operator
     *
     * @param[in] &FIUManager Reference to the FIUManager object
     */
    FIUManager& operator=(const FIUManager&) = delete;

    /** This pointer array points to all the FIUCommands it manages
     *  This array represents all the FIUCommands that are usable across
     *  the system.
     *
     * NOTE: One of several objects will be using these objects at any point of time
     *       Do not use or free these objects. JSONParser shall be stashing all the
     *       free FIUCommand objects.
     */
    FIUCommand** m_fiuCommandArray{};

    /** This pointer array points to all the FIUCommands that are filled, but
     *  can't be assigned to the FIUClasses due to various reasons:
     *    - FIUClass has not been defined yet during init phase before main() executes
     *    - The instance has not been defined yet
     *    - Chained FIUs awaiting the above two conditions (FUTURE USE)
     */
    FIUCommand** m_enabledFIUCommandsStash{};

    /// Main object Manager object that manages all the memory blocks for the FI tool
    ObjectManager m_objectManager;

    /// Pointer to the JSONParser object that has the functionality to parse JSON strings
    JSONParserPtr m_jsonParserPtr{};

    /// Flag that indicates to the parallel client threads the the FIManager is about to shutdown
    std::atomic<bool> m_shutdownFlag{false};

    /// Flag holds if the FI Manager is activated or not
    bool m_isFIDeactivated{false};

    /// Pointer to the FIUClass object
    FIClassMapPtr m_fiuObjectMapPtr{};

    /// Pointer to the fiuInstance object
    FIUInstance* m_fiuInstancePtr{};

    /// Integer that holds the total FIU Command objects supported in parallel
    uint32_t m_totalFIUCommandsSupported{TOTAL_FAULT_COMMAND_SUPPORTED};
};

/** Helper function that returns the fiu manager singleton object
 *
 * @return reference to FIUManager singleton object
 */
fiu2::FIUManager& getFIUManagerSingleton();
};

#endif
