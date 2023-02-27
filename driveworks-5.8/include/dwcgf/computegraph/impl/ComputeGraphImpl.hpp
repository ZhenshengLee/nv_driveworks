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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_COMPUTEGRAPH_IMPL_HPP_
#define DW_FRAMEWORK_COMPUTEGRAPH_IMPL_HPP_

#include <dw/core/base/Types.h>

#include <dwcgf/Types.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/computegraph/Connection.hpp>

#include <dw/core/container/Span.hpp>
#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/container/HashContainer.hpp>

#include <stack>

#include <vector>
#include <algorithm>
#include <iterator>

namespace dw
{
namespace core
{
template <typename Type, size_t CapacityAtCompileTime_ = 0>
class StackFixed : public VectorFixed<Type, CapacityAtCompileTime_>
{
public:
    StackFixed()
        : VectorFixed<Type, CapacityAtCompileTime_>()
    {
    }

    explicit StackFixed(size_t size_)
        : VectorFixed<Type, CapacityAtCompileTime_>(size_)
    {
    }

    bool push(const Type& value)
    {
        return this->push_back(value);
    }

    bool pop()
    {
        if (this->empty())
        {
            return false;
        }
        this->pop_back();
        return true;
    }

    const Type& top()
    {
        if (this->empty())
        {
            return this->at(0);
        }
        return this->back();
    }

    bool contains(const Type& value) const
    {
        for (const auto& each : *this)
        {
            if (each == value)
            {
                return true;
            }
        }
        return false;
    }
};

template <typename Type, size_t CapacityAtCompileTime_ = 0>
class QueueFixed : public VectorFixed<Type, CapacityAtCompileTime_>
{
public:
    QueueFixed()
        : VectorFixed<Type, CapacityAtCompileTime_>()
    {
    }

    explicit QueueFixed(size_t size_)
        : VectorFixed<Type, CapacityAtCompileTime_>(size_)
    {
    }

    bool push(const Type& value)
    {
        return this->push_back(value);
    }

    bool empty()
    {
        return this->size() == 0;
    }

    bool pop()
    {
        if (this->empty())
        {
            return false;
        }
        m_index++;
        return true;
    }

    const Type& top()
    {
        if (this->empty())
        {
            return this->at(0);
        }
        return this->at(m_index);
    }

    size_t size()
    {
        return VectorFixed<Type, CapacityAtCompileTime_>::size() - m_index;
    }

    void clear()
    {
        VectorFixed<Type, CapacityAtCompileTime_>::clear();
        m_index = 0;
    }

    bool contains(const Type& value) const
    {
        for (const auto& each : *this)
        {
            if (each == value)
            {
                return true;
            }
        }
        return false;
    }

private:
    size_t m_index = 0;
};
}
}

// TODO(eklein): clang-tidy complains about this "using" here. They suggest instead we use "using-declarations".
// See https://en.cppreference.com/w/cpp/language/using_declaration.
//
// That's a fair bit of work, so for now I'm leading this as a TODO, but we ought to at least look into it.
//
// NOLINTNEXTLINE(google-build-using-namespace)
//using namespace dw::core;

namespace dw
{
namespace framework
{

/***
 * @brief A collection of different execution types
 */
enum class ComputeGraphTraversalOrder
{
    TRAVERSAL_ORDER_BREADTH_FIRST = 0,
    TRAVERSAL_ORDER_DEPTH_FIRST
};

class ComputeGraphImpl
{
public:
    static constexpr char LOG_TAG[] = "ComputeGraphImpl";

    explicit ComputeGraphImpl();

    virtual ~ComputeGraphImpl() = default;

    void setChannels(span<ChannelObject*> channels);

    void setNodes(span<Node*> nodes);

    void setConnections(span<const Connection> connections);

    void resetConnections();

    void build();

    void getPass(Pass*& pass, const char* key);

    void getPasses(VectorFixed<Pass*>& passList, uint64_t nodeIndex);

    void getPasses(VectorFixed<Pass*>& passList, ComputeGraphTraversalOrder order);

    void getPasses(VectorFixed<Pass*>& passList, dwProcessorType processorType, dwProcessType processType, ComputeGraphTraversalOrder order);

    void run(ComputeGraphTraversalOrder order);

    void printAdjacencyMatrix();

private:
    static constexpr uint64_t MAX_ADJACENCY_MATRIX_DIM      = 1000;
    static constexpr uint64_t MAX_ADJACENCY_MATRIX_CELL_DIM = 100;

    void connectNode(const Connection& connection);

    bool hasChannels() const;
    bool hasNodes() const;
    bool hasConnections() const;
    bool hasValidInputs() const;
    bool connectNodeInputsAreValid(const Connection& connection) const;

    using CellEntry_t = struct CellEntry_t
    {
        uint64_t inputNodeId;
        uint64_t channelId;
    };

    using AdjacencyMatrixCell_t = VectorFixed<CellEntry_t, MAX_ADJACENCY_MATRIX_CELL_DIM>;
    using AdjacencyMatrixRow_t  = VectorFixed<AdjacencyMatrixCell_t, MAX_ADJACENCY_MATRIX_DIM>;
    using AdjacenyMatrix_t      = VectorFixed<AdjacencyMatrixRow_t, MAX_ADJACENCY_MATRIX_DIM>;
    using Queue_t               = QueueFixed<uint64_t, MAX_ADJACENCY_MATRIX_DIM>;
    using Stack_t               = StackFixed<uint64_t, MAX_ADJACENCY_MATRIX_DIM>;
    using Vector_t              = VectorFixed<uint64_t, MAX_ADJACENCY_MATRIX_DIM>;
    using VectorB_t             = VectorFixed<bool, MAX_ADJACENCY_MATRIX_DIM>;

    static void labelDisconnectedInputs(AdjacenyMatrix_t& adjacencyMatrix);

    static bool graphHasImproperChannelUse(Vector_t& usedChannels,
                                           const AdjacenyMatrix_t& adjacencyMatrix,
                                           uint64_t numChannels);
    static bool graphIsCyclic(uint64_t rowIndex,
                              VectorB_t& visited,
                              VectorB_t& nodesToExplore,
                              const AdjacenyMatrix_t& adjacencyMatrix);
    static bool graphIsIncorrect(Stack_t& stack,
                                 VectorB_t& visited,
                                 VectorB_t& nodesToExplore,
                                 const Queue_t& rootNodes,
                                 const AdjacenyMatrix_t& adjacencyMatrix);

    template <typename ContainterType>
    static void findRootNodes(ContainterType& rootNodes,
                              const AdjacenyMatrix_t& adjacencyMatrix);

    template <typename ContainerType, typename VisitFunction>
    static void traverse(ContainerType& nodesToVisit,
                         const Queue_t& rootNodes,
                         const AdjacenyMatrix_t& adjacencyMatrix,
                         VisitFunction visit);

    static void prettyPrint(const AdjacenyMatrix_t& adjacencyMatrix);

private:
    span<ChannelObject*> m_channels;
    span<Node*> m_nodes;
    span<const Connection> m_connections;

    bool m_validated;

    //////////////////////////////////////////////
    /// \brief m_adjacencyMatrix
    /// I = Input
    /// O = Output
    /// 0-9 = Number of nodes that
    /// represent graph vertices.
    /// Each cell contains a list of channels to
    /// indicate edges between vertices.
    ///  I 0 1 2 3 4 5 6 7 8 9 O
    /// I
    /// 0
    /// 1
    /// 2
    /// 3
    /// 4
    /// 5
    /// 6
    /// 7
    /// 8
    /// 9
    /// O

    AdjacenyMatrix_t m_adjacencyMatrix;

    Vector_t m_usedChannels;
    VectorB_t m_nodesToExplore;
    VectorB_t m_visitedNodes;
    Queue_t m_rootNodes;
    Queue_t m_queue;
    Stack_t m_stack;

    VectorFixed<Pass*> m_passList;
    using PassTableString = FixedString<256>;
    HashMap<PassTableString, Pass*> m_passTable;
};
}
}

#endif // DW_FRAMEWORK_COMPUTEGRAPH_IMPL_HPP_
