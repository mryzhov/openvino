// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Converts Multiply (int16 inputs) - Add pattern to Diagonal layer compatible with
 * Gna2OperationTypeElementWiseAffine primitive
 */
class ConvertMulAddToDiagonal : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMulAddToDiagonal();
};

/**
 * @brief Converts Multiply (int16 and int32 inputs) layer to Diagonal layer compatible with
 * Gna2OperationTypeElementWiseAffine primitive
 */
class ConvertMulToDiagonal : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMulToDiagonal();
};

class ConvertToDiagonal : public ngraph::pass::GraphRewrite {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertToDiagonal() {
    add_matcher<ConvertMulAddToDiagonal>();
    add_matcher<ConvertMulToDiagonal>();
  }
};


} // namespace GNAPluginNS