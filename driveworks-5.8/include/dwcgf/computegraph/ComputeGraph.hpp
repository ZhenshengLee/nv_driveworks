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

#ifndef DW_FRAMEWORK_COMPUTEGRAPH_HPP_
#define DW_FRAMEWORK_COMPUTEGRAPH_HPP_

#include <dw/core/base/Types.h>

#include <dwcgf/Types.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/pass/Pass.hpp>

#include <dwcgf/computegraph/Connection.hpp>
#include <dwcgf/computegraph/impl/ComputeGraphImpl.hpp>
#include <dw/core/container/Span.hpp>
#include <dw/core/container/VectorFixed.hpp>

using dw::core::span;

namespace dw
{
namespace framework
{
class ComputeGraph
{
public:
    explicit ComputeGraph();

    virtual ~ComputeGraph() = default;

    /***
     * @brief Set the channel array for this graph. These are
     * the channels this graph will use for inputs, outputs,
     * and connections between nodes.
     * @param[in] channels A contiguous array holding the all the channel pointers for the graph.
     * @return DW_SUCCESS - All channels are accepted as input.
     *         DW_INVALID_ARGUMENT - @ref channels is not a valid pointer.
     *         DW_INVALID_ARGUMENT - size of @ref channels is 0.
     *         DW_FAILURE - The channels cannot be set.
     */
    dwStatus setChannels(span<ChannelObject*> channels);

    /***
     * @brief Set the node array for this graph. These are
     * the nodes this graph will use as its vertices, while
     * channels are the edges.
     * @param[in] nodes A contiguous array holding the all the node pointers for the graph.
     * @return DW_SUCCESS - All nodes are accepted as input.
     *         DW_INVALID_ARGUMENT - @ref nodes is not a valid pointer.
     *         DW_INVALID_ARGUMENT - size of @ref nodes is 0.
     *         DW_FAILURE - The nodes cannot be set.
     */
    dwStatus setNodes(span<Node*> nodes);

    /***
     * @brief Set the connection array for this graph.
     * @param[in] nodes A contiguous array holding the all the connections for the graph.
     * @return DW_SUCCESS - All nodes are accepted as input.
     *         DW_INVALID_ARGUMENT - @ref connections is not a valid pointer.
     *         DW_INVALID_ARGUMENT - size of @ref connections is 0.
     *         DW_FAILURE - The nodes cannot be set.
     */
    dwStatus setConnections(span<const Connection> connections);

    /***
     * @brief Resets all the connections between nodes.
     * @return DW_SUCCESS - The connections are reset.
     *         DW_CALL_NOT_ALLOWED - @ref setNodes(span<Node*>) has not been called with valid data.
     *         DW_CALL_NOT_ALLOWED - @ref setChannels(span<ChannelObject*>) has not been called with valid data.
     */
    dwStatus resetConnections();

    /***
     * @brief Build and validate the graph.
     * @return DW_SUCCESS - The graph is ready for execution.
     *         DW_CALL_NOT_ALLOWED - @ref setNodes(span<Node*>) has not been called with valid data.
     *         DW_CALL_NOT_ALLOWED - @ref setChannels(span<ChannelObject*>) has not been called with valid data.
     *         DW_CALL_NOT_ALLOWED - @ref setConnections(span<const Connection>) has not been called with valid data.
     *         DW_INTERNAL_ERROR - Channels from @ref setChannels(span<ChannelObject*>) do not connect to one
     *                      input and one output exactly.
     *         DW_INTERNAL_ERROR - There is at least one cycle detected in the graph.
     *         DW_INTERNAL_ERROR - Not all nodes are used.
     *         DW_INTERNAL_ERROR - All node inputs do not have input channels.
     *         DW_INTERNAL_ERROR - All node outputs do not have output channels.
     */
    dwStatus build();

    /***
     * @brief Get a pass by name.
     * @param[out] pass The pass to populate.
     * @param[in] key String key is node_name.pass# (i.e., "detector.0").
     * @return DW_SUCCESS - The pass is populated.
     *         DW_CALL_NOT_ALLOWED - @ref build has not been called.
     *         DW_INVALID_ARGUMENT - @ref key is not valid.
     */
    dwStatus getPass(Pass*& pass, const char* key);

    /***
     * @brief Get the pass list for a specific node.
     * @param[out] passList The list to populate with passes.
     * @param[in] nodeIndex The index of the node.
     * @return DW_SUCCESS - The pass list is populated.
     *         DW_CALL_NOT_ALLOWED - @ref build has not been called.
     *         DW_INVALID_ARGUMENT - @ref nodeIndex is not valid.
     */
    dwStatus getPasses(VectorFixed<Pass*>& passList, uint64_t nodeIndex);

    /***
     * @brief Get the pass list for all nodes in a specific order.
     * @param[out] passList The list to populate with passes.
     * @param[in] order The traversal order of a the graph.
     * @return DW_SUCCESS - The pass list is populated.
     *         DW_CALL_NOT_ALLOWED - @ref build has not been called.
     *         DW_INVALID_ARGUMENT - @ref order is not valid.
     */
    dwStatus getPasses(VectorFixed<Pass*>& passList, ComputeGraphTraversalOrder order);

    /***
     * @brief Get the pass list for all nodes filtered by a specific processor type and process type in a specific order.
     * @param[out] passList The list to populate with passes.
     * @param[in] processorType The processor type.
     * @param[in] processType The process type.
     * @param[in] order The traversal order of a the graph.
     * @return DW_SUCCESS - The pass list is populated.
     *         DW_CALL_NOT_ALLOWED - @ref build has not been called.
     *         DW_INVALID_ARGUMENT - @ref order is not valid.
     */
    dwStatus getPasses(VectorFixed<Pass*>& passList, dwProcessorType processorType, dwProcessType processType, ComputeGraphTraversalOrder order);

    /***
     * @brief Execute the compute graph.
     * @param[in] order The traversal order of the graph to run.
     * @return DW_SUCCESS - The graph is executed.
     *         DW_CALL_NOT_ALLOWED - @ref build has not been called.
     *         DW_INVALID_ARGUMENT - @ref type is not a valid @ref ComputeGraphTraversalOrder
     *         DW_FAILURE
     */
    dwStatus run(ComputeGraphTraversalOrder order);

    /***
     * @brief Prints the adjacency matrix.
     * @return DW_SUCCESS - The matrix is output.
     *         DW_CALL_NOT_ALLOWED - @ref setNodes(span<Node*>) has not been called with valid data.
     *         DW_CALL_NOT_ALLOWED - @ref setChannels(span<ChannelObject*>) has not been called with valid data.
     *         DW_CALL_NOT_ALLOWED - @ref setConnections(span<const Connection>) has not been called with valid data.
     */
    dwStatus printAdjacencyMatrix();

protected:
    std::unique_ptr<ComputeGraphImpl> m_impl;
};
}
}
#endif // DW_FRAMEWORK_COMPUTEGRAPH_HPP_
