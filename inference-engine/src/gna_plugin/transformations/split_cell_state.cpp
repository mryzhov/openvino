// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/split_cell_state.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

#include "backend/gna_limitations.hpp"
#include "frontend/scale_factor_calc.hpp"
#include "ops/diagonal.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SplitCellState, "SplitCellState", 0);

SplitCellState::SplitCellState() {
    MATCHER_SCOPE(SplitCellState);

    auto read_value = ngraph::pattern::wrap_type<ngraph::opset8::ReadValue>();

    auto mul_fq1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>();
    auto mul_fq2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({read_value, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({mul_fq1, mul_fq2});

    auto add_fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({multiply, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_fq, ngraph::pattern::any_input()});

    auto assign_fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({add, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset8::Unsqueeze>({assign_fq, ngraph::pattern::any_input()});
    auto squeeze = ngraph::pattern::wrap_type<ngraph::opset8::Squeeze>({unsqueeze, ngraph::pattern::any_input()});
    auto assign_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{squeeze, assign_fq});
    auto assign = ngraph::pattern::wrap_type<ngraph::opset8::Assign>({assign_input});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto read_value_node = pattern_map.at(read_value).get_node_shared_ptr();

        auto mul_fq1_node = pattern_map.at(mul_fq1).get_node_shared_ptr();
        auto mul_fq2_node = pattern_map.at(mul_fq2).get_node_shared_ptr();
        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto assign_fq_node = pattern_map.at(assign_fq).get_node_shared_ptr();
        auto assign_node = pattern_map.at(assign).get_node_shared_ptr();

        ngraph::NodeVector new_ops;
        auto shape = read_value_node->get_output_shape(0);
        size_t size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
        auto el_type = read_value_node->get_element_type();

        // Change states precision to 32-bits
        size_t levels_32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
        size_t levels_16 = static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
        bool clamp_values = true;
        const float clamped_val = 3.0;

        // Calculate int16 sf for state range
        auto get_const_val = [](std::shared_ptr<ngraph::Node> node) {
            auto node_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
            IE_ASSERT(node_const != nullptr);
            return node_const->get_vector<float>().front();
        };
        float state_min = clamp_values ? -clamped_val : get_const_val(mul_fq2_node->input_value(1).get_node_shared_ptr());
        float state_max = clamp_values ? clamped_val : get_const_val(mul_fq2_node->input_value(2).get_node_shared_ptr());
        float sf = frontend::CalculateScaleFactorFromStats(levels_16, state_min, state_max);

        auto create_fq = [el_type](std::shared_ptr<ngraph::Node> input, float min_val, float max_val, size_t levels) {
            return std::make_shared<ngraph::opset8::FakeQuantize>(input,
                ngraph::opset8::Constant::create(el_type, ngraph::Shape{}, std::vector<float>(1, min_val)),
                ngraph::opset8::Constant::create(el_type, ngraph::Shape{}, std::vector<float>(1, max_val)),
                ngraph::opset8::Constant::create(el_type, ngraph::Shape{}, std::vector<float>(1, min_val)),
                ngraph::opset8::Constant::create(el_type, ngraph::Shape{}, std::vector<float>(1, max_val)),
                levels);
        };

        std::shared_ptr<ngraph::opset8::FakeQuantize> first_fq;
        if (clamp_values) {
            first_fq = create_fq(read_value_node, -clamped_val, clamped_val, levels_32);
            new_ops.push_back(first_fq);
        } else {
            first_fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(mul_fq2_node);
            first_fq->set_levels(levels_32);
        }
        // Create Add (with bias = state) and FQ which will actualy performs Floor operation and
        // gives the high parts of values
        auto constant_zero1 = ngraph::opset8::Constant::create(el_type, shape, std::vector<float>(size, 0.0));
        auto constant_zero2 = ngraph::opset8::Constant::create(el_type, shape, std::vector<float>(size, 0.0));
        //auto add_zero = std::make_shared<ngraph::opset8::Add>(mul_fq2_node, constant_zero);
        auto add_zero = std::make_shared<Diagonal>(constant_zero1, constant_zero2, first_fq);
        add_zero->set_friendly_name(read_value_node->get_friendly_name() + "/high_part");
        new_ops.push_back(add_zero);
        auto high_part = create_fq(add_zero, state_min, state_max, levels_16);
        new_ops.push_back(high_part);

        // Create Substract (with bias = state) and FQ which gives the low part of values
        //auto sub = std::make_shared<ngraph::opset8::Subtract>(read_value_node, high_part);
        auto constant_neg = ngraph::opset8::Constant::create(el_type, shape, std::vector<float>(size, -1.0));
        auto sub = std::make_shared<Diagonal>(constant_neg, high_part, first_fq);
        sub->set_friendly_name(read_value_node->get_friendly_name() + "/low_part");
        new_ops.push_back(sub);
        float low_part_min = -2 / sf;
        float low_part_max = 2 / sf;
        auto low_part = create_fq(sub, low_part_min, low_part_max, levels_16);
        new_ops.push_back(low_part);

        // Run original operations (multiply and add) separately for high and low parts:
        // (h + l) * a + b = h * a + (l * a + b)
        //auto mul_high = multiply_node->clone_with_new_inputs(ngraph::OutputVector{high_part, multiply_node->input_value(0)});
        //new_ops.push_back(mul_high);
        auto mul_low = multiply_node->clone_with_new_inputs(ngraph::OutputVector{low_part, multiply_node->input_value(0)});
        new_ops.push_back(mul_low);
        float sigm_out_min = get_const_val(mul_fq1_node->input_value(1).get_node_shared_ptr());
        float sigm_out_max = get_const_val(mul_fq1_node->input_value(2).get_node_shared_ptr());
        auto add_low_fq = create_fq(mul_low, -sigm_out_min * low_part_min, sigm_out_max * low_part_max, levels_16);
        new_ops.push_back(add_low_fq);
        auto add_low_input = add_node->input_value(1).get_node_shared_ptr();
        auto add_low = std::make_shared<ngraph::opset8::Add>(add_low_fq, add_low_input);
        new_ops.push_back(add_low);
        float mul_out_min = get_const_val(add_low_input->input_value(1).get_node_shared_ptr());
        float mul_out_max = get_const_val(add_low_input->input_value(2).get_node_shared_ptr());
        auto full_val_fq = create_fq(add_low, mul_out_min + low_part_min, mul_out_max + low_part_max, levels_32);
        new_ops.push_back(full_val_fq);

        // Combine high and low parts into the full value
        //auto full_val_fq = create_fq(mul_high, -sigm_out_min * min_val_after_floor, sigm_out_max * max_val_after_floor, levels_16);
        //new_ops.push_back(full_val_fq);
        //auto full_val = std::make_shared<ngraph::opset8::Add>(full_val_fq, add_low);
        auto full_val = std::make_shared<Diagonal>(high_part, multiply_node->input_value(0), full_val_fq);
        full_val->set_friendly_name(read_value_node->get_friendly_name() + "/full_val");
        new_ops.push_back(full_val);

        std::shared_ptr<ngraph::opset8::FakeQuantize> last_fq;
        if (clamp_values) {
            last_fq = create_fq(full_val, -clamped_val, clamped_val, levels_32);
            last_fq->set_friendly_name(assign_fq_node->get_friendly_name());
            new_ops.push_back(last_fq);
            replace_node(assign_fq_node, last_fq);
        } else {
            last_fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(assign_fq_node);
            last_fq->set_levels(levels_32);
            replace_node(assign_fq_node->input_value(0).get_node_shared_ptr(), full_val);
        }

        copy_runtime_info(assign_node, new_ops);

        // Insert Add layer to prevent fusing of the last diagonal layer with Tanh,
        // otherwise state buffer will be between the fused layers
        auto constant1 = ngraph::opset8::Constant::create(el_type, shape, std::vector<float>(size, 0.0));
        auto constant2 = ngraph::opset8::Constant::create(el_type, shape, std::vector<float>(size, 0.0));
        auto diag_tanh = std::make_shared<Diagonal>(constant1, constant2, last_fq);
        for (const auto &input : last_fq->get_output_target_inputs(0)) {
            if (auto tanh_node = std::dynamic_pointer_cast<ngraph::opset8::Tanh>(input.get_node()->shared_from_this())) {
                tanh_node->input(0).replace_source_output(diag_tanh->output(0));
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(assign, matcher_name);
    this->register_matcher(m, callback);
}