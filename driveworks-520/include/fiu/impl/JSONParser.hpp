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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <modern-json/json.hpp>
#include <fiu/impl/FIUCommand.hpp>

namespace fiu2
{

/**
 * @class JSONParser
 * @brief This class provides functionalities to parse JSON commands into FIUCommand objects.
 */
class JSONParser
{
public:
    /**
    * @brief Distructor clears the allocated memory
    */
    ~JSONParser()
    {
        delete[] m_freeFIUCommandArray;
        delete[] m_reusableArray;
    };

    /**
    * @brief Constructor allocates memory to m_reusableArray
    */
    JSONParser()
    {
        m_reusableArray = new unsigned char[totalArraySize]();
    };

    /**
    * @brief Resets the memory of the m_reusableArray buffer
    */
    void resetReusableArray()
    {
        memset(m_reusableArray, 0, totalArraySize);
    }

    /**
    * @brief Sets the total FIUCommands supported and creates an array to hold them
    *
    * @param[in] totalFIUCommands allowed for this execution
    * @return true if buffer allocated; false if the buffer has already been allocated
    */
    bool setupFreeFIUCommandsList(uint32_t max)
    {
        if (m_freeFIUCommandArray)
        {
            return false;
        }
        m_totalFIUCommandsSupported = max;
        m_freeFIUCommandArray       = new FIUCommand*[m_totalFIUCommandsSupported];
        for (uint32_t index = 0; index < m_totalFIUCommandsSupported; index++)
        {
            m_freeFIUCommandArray[index] = nullptr;
        }
        return true;
    };

    /**
    * @brief Stashes the given fiu command into a array for future use
    *
    * @param[in] fiuCommand Pointer to the fiu command
    * @return true if command has been stashed correctly
    */
    bool stashFreeFIUCommand(FIUCommand* fiuCommand);

    /**
    * @brief Returns a FIUCommand object from its m_freeFIUCommandArray
    *
    * @return FIUCommand object if available; nulptr if otherwise
    */
    FIUCommand* getReusableFIUCommand();

    /**
    * @brief Returns the count of free FIUCommand objects
    *
    * @return count of free FIUCommand objects
    */
    uint32_t getFreeFIUCommandCount();

    /**
    * @brief Parses the JSON commands from a json instance.
    *
    * @param[in] json The json instance containing the commands.
    * @return Vector of parsed commands in ParsedCommand structure format.
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON.
    */
    bool parseJSONCommands(const nlohmann::json& json);

    /**
    * @brief Processes the log from a JSON command and sets it on the FIUCommand object.
    *
    * @param[out] command Reference to the FIUCommand object.
    * @param[in] log Refernece to log data in JSON command.
    */
    bool processLog(FIUCommand& command, const std::string& log);

    /**
    * @brief Processes the data from a JSON command and sets it on the FIUCommand object.
    *
    * @param[out] command Reference to the FIUCommand object.
    * @param[in] data The data in JSON command.
    */
    bool processData(FIUCommand& command, const nlohmann::json& data);

    /**
    * @brief Parses the JSON commands from a file.
    *
    * @param[in] filename The name of the file containing the JSON commands.
    * @return true if the file has been parsed correctly or not.
    * @throw std::runtime_error if the file cannot be opened.
    */
    bool parseFIUFile(const char* filename);

private:
    /**
    * @brief Processes the frequency from a JSON command and sets it on the FIUCommand object.
    *
    * @param[out] fiuCommand Reference to the FIUCommand object.
    * @param[in] jsonString The frequency data in JSON command.
    * @return true if the frequency is valid
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool processFrequency(FIUCommand& fiuCommand, const nlohmann::json& jsonString);

    /**
    * @brief Processes the enable command
    *
    * @param[out] fiuCommand Reference to the FIUCommand object.
    * @param[in] jsonString Reference to the json string.
    * @return true if the command is valid
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool processCommandEnable(FIUCommand& fiuCommand, const nlohmann::json& jsonString);

    /**
    * @brief Processes a fiu command given the input json string
    *
    * @param[out] fiuCommand Pointer to the fiuCommand obect.
    * @param[in] jsonString The json instance containing the command.
    * @return true if the command is valid
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool processCommand(FIUCommand& fiuCommand, const nlohmann::json& jsonString);

    /**
    * @brief Copes json vector into an array
    *
    * @param[out] ptr Pointer to the buffer structure.
    * @param[in] param Reference to the json parameter
    * @param[in] size Reference to the size of the buffer
    * @tparam b Variable of the given type
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    template <typename T>
    void copyToVector(void* ptr, const nlohmann::json& param, size_t& size, T b)
    {
        size        = 0;
        T* typedPtr = reinterpret_cast<T*>(ptr);
        for (auto& elem : param["value"])
        {
            typedPtr[size] = elem.get<T>();
            size++;
        }
        size *= sizeof(T);
    }

    /**
    * @brief Adds the content of thejson vector into the FIUCommand array
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @param ptype Reference to the ParamType structure that contains the details of the parameter
    * @tparam a Variable of the given type
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    template <typename T>
    bool handleVectorType(FIUCommand& command, const char* key, const nlohmann::json& param, ParamType& ptype, T a)
    {
        T* b{};
        if (!strcmp(ptype.demangledName, getDemangledTypeName(b)))
        {
            copyToVector<T>(ptype.arrayPtr, param, ptype.arraySize, 1);
            command.addVectorParam(key, ptype.arrayPtr, ptype.arraySize, getDemangledTypeName(b));
            return true;
        }
        return false;
    }

    /**
    * @brief Processes the given JSON parameter
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool processParamType(FIUCommand& command, const char* key, const nlohmann::json& param);

    /**
    * @brief Processes the boolean type from JSON info
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @param ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleBoolType(FIUCommand& command, const char* key, const nlohmann::json& param, ParamType ptype);

    /**
    * @brief Processes the char type from JSON info
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @param ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleCharType(FIUCommand& command, const char* key, const nlohmann::json& param, ParamType ptype);

    /**
    * @brief Processes the given JSON parameter
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool processParam(FIUCommand& command, const char* key, const nlohmann::json& param);

    /**
    * @brief Processes scalar array type from JSON info
    *
    * @param[out] command Reference to the FIUCommand object
    * @param key Pointer to the key
    * @param param Reference to the json parameter
    * @param ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesArray(FIUCommand& command,
                                const char* key,
                                const nlohmann::json& param,
                                ParamType& ptype);

    /**
    * @brief Processes scalar array type from JSON info
    *
    * @param[in] param Reference to the json parameter
    * @param[in] arrayCount Reference to the size of the array
    * @param[in] invalidArray Reference to the flag that indicates if the array is valid or not
    * @param[in] outOfRangeElement Reference to the flag that indicates if the array is valid or not
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesProcessValueValidation(const nlohmann::json& param,
                                                 int arrayCount,
                                                 bool invalidArray,
                                                 bool outOfRangeElement);

    /**
    * @brief Validates scalar related JSON info
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesProcessValue(FIUCommand& command,
                                       const char* key,
                                       const nlohmann::json& param,
                                       ParamType& ptype);

    /**
    * @brief Processes char types
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesChar(FIUCommand& command,
                               const char* key,
                               const nlohmann::json& param,
                               ParamType& ptype);

    /**
    * @brief Processes vector integer types
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesVectorTypeValidationIntTypes(FIUCommand& command,
                                                       const char* key,
                                                       const nlohmann::json& param,
                                                       ParamType& ptype);

    /**
    * @brief Processes vector float/double/char types
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypesVectorTypeValidationFloatTypes(FIUCommand& command,
                                                         const char* key,
                                                         const nlohmann::json& param,
                                                         ParamType& ptype);

    /**
    * @brief Processes scalar types
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleScalarTypes(FIUCommand& command,
                           const char* key,
                           const nlohmann::json& param,
                           ParamType& ptype);

    /**
    * @brief Processes signed scalar types
    *
    * @param[out] command Reference to the FIUCommand object
    * @param[in] key Pointer to the key
    * @param[in] param Reference to the json parameter
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool handleSignedScalarTypes(FIUCommand& command,
                                 const char* key,
                                 const nlohmann::json& param,
                                 ParamType ptype);

    /**
    * @brief Validate Range signed scalar types
    *
    * @param[in] ptype Reference to the ParamType structure that contains the details of the parameter
    * @param[in] param Reference to the json parameter
    * @return true if the command is valid and processed successfully
    * @throw std::runtime_error if mandatory fields are missing or contain invalid data in JSON command.
    */
    bool checkRange(ParamType ptype, const nlohmann::json& param);

    /// Array of FIUCommands
    FIUCommand** m_freeFIUCommandArray{};

    /// Holds the total fiu commands supported
    uint32_t m_totalFIUCommandsSupported{TOTAL_FAULT_COMMAND_SUPPORTED};

    /// Pointer to a reusable char buffer
    unsigned char* m_reusableArray{};

    /// Size of the reusable char buffer
    size_t totalArraySize{MAX_ALLOWED_JSON_ARRAY_SIZE * sizeof(double)};
};

/// Shared pointer typedef of JSONParser
using JSONParserPtr = std::shared_ptr<JSONParser>;

/**
* @brief processes a JSON FIU command
*
* @param[in] command Reference to the fiu command in json format
* @return true if the command is valid and processed successfully
*/
extern bool globalParseJSONCommands(const nlohmann::json& command);

} // namespace fiu2

#endif // JSON_PARSER_HPP
