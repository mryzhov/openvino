// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Substitutes ngraph::Convolution (NCHW) by GNA-specific GNAConvolution (NHWC)
 * from
 *         Any #1
 *           |
 *   ngraph::Convolution (NCHW)
 *           |
 *        Any #2
 * to
 *         Any #1
 *           |
 *  Transpose NCHW -> NHWC
 *           |
 *    GNAConvolution NHWC
 *           |
 *  Transpose NHWC -> NCHW
 *           |
 *        Any #2
 */
class SubstituteGNAConvolution : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAConvolution();
};

/**
 * @brief Substitutes ngraph::MaxPool (NCHW) by GNA-specific GNAMaxPool (NHWC)
 * from
 *         Any #1
 *           |
 *   ngraph::MaxPool (NCHW)
 *           |
 *        Any #2
 * to
 *         Any #1
 *           |
 *  Transpose NCHW -> NHWC
 *           |
 *    GNAMaxPool NHWC
 *           |
 *  Transpose NHWC -> NCHW
 *           |
 *        Any #2
 */
class SubstituteGNAMaxPool : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGNAMaxPool();
};

/** @brief Calls SubstituteGNAConvolution and SubstituteGNAMaxPool transformations
 *
 */
class TransposeNCHW : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
