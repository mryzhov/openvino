// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "diagonal.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

NGRAPH_RTTI_DEFINITION(GNAPluginNS::Diagonal, "Diagonal", 0);

namespace GNAPluginNS {

Diagonal::Diagonal(const ngraph::Output<ngraph::Node>& input,
                   const ngraph::Output<ngraph::Node>& weights,
                   const ngraph::Output<ngraph::Node>& biases)
        : ov::op::Op(ngraph::OutputVector{input, weights, biases}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> Diagonal::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Diagonal>(new_args.at(0), new_args.at(1), new_args.at(2));
}

bool Diagonal::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

void Diagonal::validate_and_infer_types() {
    ngraph::element::Type element_type = get_input_element_type(0);
    ngraph::PartialShape pshape = get_input_partial_shape(0);

    for (size_t i = 1; i < get_input_size(); ++i) {
        NODE_VALIDATION_CHECK(this,
                              ngraph::element::Type::merge(element_type, element_type, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
    }

    NODE_VALIDATION_CHECK(this,
                          element_type.is_dynamic() || element_type != ngraph::element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          element_type,
                          ").");

    set_output_type(0, element_type, pshape);
}

} // namespace GNAPluginNS