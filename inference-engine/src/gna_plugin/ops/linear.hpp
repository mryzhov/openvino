// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace GNAPluginNS {

class Linear : public ov::op::util::UnaryElementwiseArithmetic {
public:
    NGRAPH_RTTI_DECLARATION;

    Linear() = default;
    /// \brief Constructs an SoftSign operation.
    ///
    /// \param data Input tensor
    Linear(const ngraph::Output<ngraph::Node>& arg);
    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool evaluate(ov::runtime::TensorVector& output_values,
                  const ov::runtime::TensorVector& input_values,
                  const ov::EvaluationContext & evaluation_context) const override;
    bool has_evaluate() const override;
};
}  // namespace GNAPluginNS
