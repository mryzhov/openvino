// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Splits cell state in the unrolled LSTM Cell into high and low parts to quantize them separately in int16
 * and support int32 precision of the whole state
 */
class SplitCellState : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitCellState();
};

} // namespace GNAPluginNS