// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/** @brief Calls GatherSinking forward group of transformations to propogate
 *  Gather layer from the start (GNAConolution/GNAMaxPool nodes) to the end of the network.
 *  All called transformations use register_new_node() method to add new created Gather node
 *  into the MatcherPass queue. That way allows to avoid calling same transformations in
 *  infinite loop to propogate one Gather layer through the graph.
 *  To avoid the same Gather node move from GNAConvolution to the Result of the graph and
 *  return on the initial position while backward propagation we setup NoGatherSinkingAttr.
 *  Setup NoGatherSinkingAttr if the next layer after Gather is not supported by GatherSinking
 *  transformations.
*/
class GatherSinkingGeneralForward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GatherSinkingGeneralForward", "0");
    GatherSinkingGeneralForward();
};

/** @brief Calls GatherSinking backward group of transformations to propogate
 *  Gather layer from the end (GNAConolution/GNAMaxPool nodes) to the start of the network.
 *  All called transformations use register_new_node() method to add new created Gather node
 *  into the MatcherPass queue. That way allows to avoid calling same transformations in
 *  infinite loop to propogate one Gather layer through the graph.
*/
class GatherSinkingGeneralBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GatherSinkingGeneralBackward", "0");
    GatherSinkingGeneralBackward();
};

/** @brief Does forward and backward Gather layers propagation.
*/
class GatherSinkingGeneral : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GatherSinkingGeneral", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
