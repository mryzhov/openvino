// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/split_cell_state.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

#include "backend/gna_limitations.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SplitCellState, "SplitCellState", 0);

SplitCellState::SplitCellState() {
    MATCHER_SCOPE(SplitCellState);

    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto read_value = ngraph::pattern::wrap_type<ngraph::opset8::ReadValue>({constant});
    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({ngraph::pattern::any_input(), read_value});
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({multiply, ngraph::pattern::any_input()});
    auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset8::Unsqueeze>({add, ngraph::pattern::any_input()});
    auto squeeze = ngraph::pattern::wrap_type<ngraph::opset8::Squeeze>({unsqueeze, ngraph::pattern::any_input()});
    auto assign_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{squeeze, add});
    auto assign = ngraph::pattern::wrap_type<ngraph::opset8::Assign>({assign_input});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto read_value_node = pattern_map.at(read_value).get_node_shared_ptr();
        auto assign_node = pattern_map.at(assign).get_node_shared_ptr();
        std::cout << "Cell state recognized: " << assign_node->get_friendly_name() << "\n";

        ngraph::NodeVector new_ops;
        auto shape = pattern_map.at(constant).get_node_shared_ptr()->get_output_shape(0);
        size_t size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());

        auto constant_div = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, 1 / (GNALimitations::cellStateDivider)));
        auto mul1 = std::make_shared<ngraph::opset8::Multiply>(read_value_node, constant_div);
        new_ops.push_back(mul1);

        auto constant_neg = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, -GNALimitations::cellStateDivider));
        auto mul2 = std::make_shared<ngraph::opset8::Multiply>(mul1, constant_neg);
        new_ops.push_back(mul2);
        auto add2 = std::make_shared<ngraph::opset8::Add>(mul2, read_value_node);
        new_ops.push_back(add2);

        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto mul_high = multiply_node->clone_with_new_inputs(ngraph::OutputVector{mul1, multiply_node->input_value(0)});
        new_ops.push_back(mul_high);
        auto mul_low = multiply_node->clone_with_new_inputs(ngraph::OutputVector{add2, multiply_node->input_value(0)});
        new_ops.push_back(mul_low);
        auto add_low = add_node->clone_with_new_inputs(ngraph::OutputVector{mul_low, add_node->input_value(1)});
        new_ops.push_back(add_low);

        auto constant_pos = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, GNALimitations::cellStateDivider));
        auto mul3 = std::make_shared<ngraph::opset8::Multiply>(mul_high, constant_pos);
        auto add3 = std::make_shared<ngraph::opset8::Add>(mul3, add_low);
        new_ops.push_back(add3);

        copy_runtime_info(assign_node, new_ops);
        replace_node(assign_node->input_value(0).get_node_shared_ptr(), add3);

        // Insert Add layer to prevent fusing of the last diagonal layer with Tanh,
        // otherwise state buffer will be between the fused layers
        auto constant_zero = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, 0.0));
        auto add4 = std::make_shared<ngraph::opset8::Add>(add3, constant_zero);
        for (const auto &input : add3->get_output_target_inputs(0)) {
            if (auto tanh_node = std::dynamic_pointer_cast<ngraph::opset8::Tanh>(input.get_node()->shared_from_this())) {
                tanh_node->input(0).replace_source_output(add4->output(0));
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(assign, matcher_name);
    this->register_matcher(m, callback);
}