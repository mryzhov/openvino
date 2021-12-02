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

        auto constant_mul = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, GNALimitations::cellStateDivider));
        auto floor_input = std::make_shared<ngraph::opset8::Multiply>(read_value_node, constant_mul);
        new_ops.push_back(floor_input);

        auto floor = std::make_shared<ngraph::opset8::Floor>(floor_input);
        new_ops.push_back(floor);

        auto constant_div = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, 1 / GNALimitations::cellStateDivider));
        auto int_part = std::make_shared<ngraph::opset8::Multiply>(floor, constant_div);
        new_ops.push_back(int_part);

        auto fp_part = std::make_shared<ngraph::opset8::Subtract>(read_value_node, int_part);
        new_ops.push_back(fp_part);

        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto mul_int = multiply_node->clone_with_new_inputs(ngraph::OutputVector{int_part, multiply_node->input_value(0)});
        new_ops.push_back(mul_int);
        auto mul_fp = multiply_node->clone_with_new_inputs(ngraph::OutputVector{fp_part, multiply_node->input_value(0)});
        new_ops.push_back(fp_part);
        auto add_fp = add_node->clone_with_new_inputs(ngraph::OutputVector{mul_fp, add_node->input_value(1)});
        new_ops.push_back(add_fp);

        auto full_val = std::make_shared<ngraph::opset8::Add>(mul_int, add_fp);
        new_ops.push_back(full_val);

        copy_runtime_info(assign_node, new_ops);
        replace_node(assign_node->input_value(0).get_node_shared_ptr(), full_val);

        // Insert Add layer to prevent fusing of the last diagonal layer with Tanh,
        // otherwise state buffer will be between the fused layers
        auto constant_zero = ngraph::opset8::Constant::create(read_value_node->get_element_type(),
            shape, std::vector<float>(size, 0.0));
        auto new_tanh_input = std::make_shared<ngraph::opset8::Add>(full_val, constant_zero);
        for (const auto &input : full_val->get_output_target_inputs(0)) {
            if (auto tanh_node = std::dynamic_pointer_cast<ngraph::opset8::Tanh>(input.get_node()->shared_from_this())) {
                tanh_node->input(0).replace_source_output(new_tanh_input->output(0));
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(assign, matcher_name);
    this->register_matcher(m, callback);
}