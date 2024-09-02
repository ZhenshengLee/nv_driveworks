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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "SamplesDataPath.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#ifdef __QNX__
#include <fstream>
#endif
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace dw_samples
{
namespace common
{

std::vector<std::string> SamplesDataPath::getSearchPaths()
{
    // Collect list of search-paths
    auto searchPaths = std::vector<std::string>{};

    // Prefer env variable "SDK_TESTS_DATA_PATH" over other search paths
    auto const envDataPath = std::getenv("SDK_TESTS_DATA_PATH");
    if (envDataPath)
    {
        searchPaths.emplace_back(envDataPath);
    }

    // Check for data location in current working directory
    auto res = std::array<char, PATH_MAX>{};
    if (nullptr != ::getcwd(res.data(), res.max_size()))
    {
        std::string cwdDataPath{res.data()};
        cwdDataPath += "/data";
        searchPaths.emplace_back(cwdDataPath);
    }

    // Check for .runfiles location in built dazel environment
    auto execFolderPath = Executable::getFolderPath();
    if (execFolderPath.size() > 0U)
    {
        searchPaths.emplace_back(execFolderPath + ".runfiles/av/data");
    }

    // Check for .runfiles location with absolute exec path as prefix
    auto execPath = Executable::getPath();
    if (execPath.size() > 0U)
    {
        searchPaths.emplace_back(execPath + ".runfiles/av/data");
    }

    auto runfilesDir = std::getenv("RUNFILES_DIR");
    if (runfilesDir)
    {
        searchPaths.emplace_back(std::string(runfilesDir) + "/av/data");
    }
    else
    {
        // Tokenize the current executables path, and search upwards for a 'data' folder (living in root of install tree / bazel sandbox).
        auto path = Executable::getFolderPath();
        if (path.size() > 0U)
        {
            while (path.size() > 1u)
            {
                searchPaths.emplace_back(path + "/data");      // corresponds to install folder / bazel sandbox name
                path = path.substr(0, path.find_last_of("/")); // peel of one folder at a time from full path
            }
        }
        else
        {
            // Fallback to use simple well-known relative paths if the executable's path can't be determined
            auto relativePath            = std::string{"data"};
            constexpr uint32_t MAX_DEPTH = 5U;
            for (auto depth = 0U; depth < MAX_DEPTH; ++depth)
            {
                relativePath.insert(0, "../");
                searchPaths.push_back(relativePath);
            }
        }
    }

    //The null character '\0' in searchPaths needs to be removed because of the use of c_str()
    for (auto& searchPath : searchPaths)
    {
        size_t nullPos = searchPath.find('\0');
        while (nullPos != std::string::npos)
        {
            searchPath.erase(nullPos, 1);
            nullPos = searchPath.find('\0');
        }
    }

    return searchPaths;
}

std::string SamplesDataPath::getBasePathFor(std::string const& fileOrDirectory,
                                            std::vector<std::string> const& searchPaths)
{
    for (auto const& searchPath : searchPaths)
    {
        auto const fullPath = searchPath + "/" + fileOrDirectory;
        if (Filesystem::fileExists(fullPath.c_str()) || Filesystem::directoryExists(fullPath.c_str()))
        {
            return searchPath;
        }
    }

    // File was not found in search paths - print warning and return empty string
    {
        auto warningMessage = std::string{"Warning: SamplesDataPath::getBasePathFor(): '"} + fileOrDirectory + "' not found in search paths [";
        auto searchPathsString =
            std::accumulate(std::begin(searchPaths), std::end(searchPaths), std::string{},
                            [](std::string const& ss, std::string const& s) {
                                return ss.empty() ? s : ss + " | " + s;
                            });
        std::cerr << warningMessage + searchPathsString + "]" << std::endl;
        return "";
    }
}

std::string SamplesDataPath::getPathFor(std::string const& fileOrDirectory,
                                        std::vector<std::string> const& searchPaths)
{
    return getBasePathFor(fileOrDirectory, searchPaths) + "/" + fileOrDirectory;
}

std::string SamplesDataPath::getBasePath(std::vector<std::string> const& searchPaths)
{
    return getBasePathFor("DATA_ROOT", searchPaths);
}

std::string SamplesDataPath::get()
{
    return getBasePath();
}

char const* SamplesDataPath::get_cstr()
{
    static auto BASE_PATH = std::string{};
    BASE_PATH             = getBasePath();
    return BASE_PATH.c_str();
}

std::string Executable::getPath()
{
#ifndef __QNX__

    auto res = std::array<char, PATH_MAX>{};
    if (!::readlink("/proc/self/exe", res.data(), PATH_MAX))
    {
        throw std::runtime_error("Executable::getPath() - no absolute executable path found");
    }
    return {res.data()};

#else

    auto file = std::ifstream("/proc/self/exefile");
    if (!file.good())
    {
        throw std::runtime_error("Executable::getPath() - no absolute executable path found");
    }
    auto path = std::string{};
    std::getline(file, path);

    //in QNX, /proc/self/exefile may contain "./" and "../", which needs to be converted to a canonical path
    char* realPath = realpath(path.c_str(), NULL);
    if (!realPath)
    {
        throw std::runtime_error(std::string("[ERROR] Cannot resolve real path of ") + path);
    }

    std::string res = std::string(realPath);
    free(realPath);
    return res;

#endif
}

std::string Executable::getFolderPath()
{
    auto const exePath = getPath();
    return exePath.substr(0, exePath.find_last_of("/"));
}

std::string Executable::getName()
{
    auto const exePath = getPath();
    return exePath.substr(exePath.find_last_of("/") + 1);
}

bool Filesystem::fileExists(char const* fileName)
{
    struct stat file
    {
    };
    return (::stat(fileName, &file) == 0 && S_ISREG(file.st_mode));
}

bool Filesystem::directoryExists(char const* directoryName)
{
    struct stat sb
    {
    };
    return (::stat(directoryName, &sb) == 0 && S_ISDIR(sb.st_mode));
}

} // namespace common
} // namespace dw_samples
