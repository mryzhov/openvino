// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "ngraph/node.hpp"

namespace GNAPluginNS {
/// \brief GNA Diagonal layer
///
class Diagonal : public ov::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Diagonal() = default;
    /// \brief Constructs a Diagonal operation.
    /// Performs multiplication of the input tensor with the diagonal weights matrix and
    /// addition of the result with the diagonal bias matrix.
    /// y = input x weights + biases
    /// \param input Node that produces the first input tensor.
    /// \param weights Node that produces the diagonal of the weights matrix.
    /// \param biases Node that produces the diagonal of the biases matrix.
    Diagonal(const ngraph::Output<ngraph::Node>& input,
             const ngraph::Output<ngraph::Node>& weights,
             const ngraph::Output<ngraph::Node>& biases);
    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
};
}  // namespace GNAPluginNS
