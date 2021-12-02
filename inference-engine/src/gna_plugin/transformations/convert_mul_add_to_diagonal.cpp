// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/convert_mul_add_to_diagonal.hpp"
#include "ops/diagonal.hpp"
#include "ops/linear.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

#include <ie/ie_common.h>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ConvertMulAddToDiagonal, "ConvertMulAddToDiagonal", 0);
NGRAPH_RTTI_DEFINITION(ConvertMulToDiagonal, "ConvertMulToDiagonal", 0);
NGRAPH_RTTI_DEFINITION(ConvertToDiagonal, "ConvertToDiagonal", 0);

ConvertMulAddToDiagonal::ConvertMulAddToDiagonal() {
    MATCHER_SCOPE(ConvertMulAddToDiagonal);
    auto is_16bit_input = [](const ngraph::Output<ngraph::Node>& node) {
        auto fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(node.get_node_shared_ptr());
        IE_ASSERT(fq != nullptr);
        return fq->get_levels() == std::numeric_limits<uint16_t>::max() ||
               fq->get_levels() == std::numeric_limits<uint16_t>::max() + 1;
    };
    auto fq_mul_in1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>(is_16bit_input);
    auto fq_mul_in2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>(is_16bit_input);
    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto in1 = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_mul_in1, constant});
    auto in2 = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_mul_in2, constant});
    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({in1, in2});
    auto eltwise_multiply = ngraph::pattern::wrap_type<ngraph::op::Eltwise>({in1, in2},
        [](const ngraph::Output<ngraph::Node>& node) {
            auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node.get_node_shared_ptr());
            IE_ASSERT(eltwise != nullptr);
            return eltwise->eltwise_type == ELTWISE_TYPE::Prod;
        });
    auto mul_root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{multiply, eltwise_multiply});
    auto fq_add = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({mul_root, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({fq_add, ngraph::pattern::any_input()});
    auto eltwise_add = ngraph::pattern::wrap_type<ngraph::op::Eltwise>({fq_add, ngraph::pattern::any_input()},
        [](const ngraph::Output<ngraph::Node>& node) {
            auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node.get_node_shared_ptr());
            IE_ASSERT(eltwise != nullptr);
            return eltwise->eltwise_type == ELTWISE_TYPE::Sum;
        });
    auto add_root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, eltwise_add});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto multiply_it = pattern_map.find(multiply);
        if (multiply_it == std::end(pattern_map)) {
            multiply_it = pattern_map.find(eltwise_multiply);
        }
        auto multiply_node = multiply_it->second.get_node_shared_ptr();

        auto add_it = pattern_map.find(add);
        if (add_it == std::end(pattern_map)) {
            add_it = pattern_map.find(eltwise_add);
        }
        auto add_node = add_it->second.get_node_shared_ptr();

        std::cout << "Diagonal layer is recognized: mul=" << multiply_node->get_friendly_name()
                  << ", add=" << add_node->get_friendly_name() << "\n";

        auto fq_add_node = pattern_map.at(fq_add).get_node_shared_ptr();
        size_t third_input_ix = add_node->input_value(0).get_node_shared_ptr() == fq_add_node ? 1 : 0;
        auto diagonal_node = std::make_shared<Diagonal>(multiply_node->input_value(0), multiply_node->input_value(1),
            add_node->input_value(third_input_ix));

        copy_runtime_info(add_node, diagonal_node);
        replace_node(add_node, diagonal_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add_root, matcher_name);
    this->register_matcher(m, callback);
}

ConvertMulToDiagonal::ConvertMulToDiagonal() {
    MATCHER_SCOPE(ConvertMulToDiagonal);
    auto fq_mul_in1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>([](const ngraph::Output<ngraph::Node>& node) {
        auto fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(node.get_node_shared_ptr());
        IE_ASSERT(fq != nullptr);
        return fq->get_levels() == std::numeric_limits<uint32_t>::max() + 1ul;
    });
    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fq_mul_in2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({fq_mul_in1, fq_mul_in2});
    auto eltwise_multiply = ngraph::pattern::wrap_type<ngraph::op::Eltwise>({fq_mul_in1, fq_mul_in2},
        [](const ngraph::Output<ngraph::Node>& node) {
            auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node.get_node_shared_ptr());
            IE_ASSERT(eltwise != nullptr);
            return eltwise->eltwise_type == ELTWISE_TYPE::Prod;
        });
    auto mul_root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{multiply, eltwise_multiply});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto multiply_it = pattern_map.find(multiply);
        if (multiply_it == std::end(pattern_map)) {
            multiply_it = pattern_map.find(eltwise_multiply);
        }
        auto multiply_node = multiply_it->second.get_node_shared_ptr();

        std::cout << "Diagonal layer is recognized: mul=" << multiply_node->get_friendly_name() << "\n";

        auto shape = multiply_node->get_output_shape(0);
        size_t size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
        auto constant_zero1 = ngraph::opset8::Constant::create(multiply_node->get_element_type(),
            shape, std::vector<float>(size, 0.0f));
        auto constant_zero2 = ngraph::opset8::Constant::create(multiply_node->get_element_type(),
            shape, std::vector<float>(size, 0.0f));
        auto diagonal_node = std::make_shared<Diagonal>(constant_zero1, constant_zero2, pattern_map.at(fq_mul_in1));
        auto linear_act = std::make_shared<Linear>(diagonal_node);

        copy_runtime_info(multiply_node, diagonal_node);
        copy_runtime_info(multiply_node, linear_act);

        replace_node(multiply_node, linear_act);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_root, matcher_name);
    this->register_matcher(m, callback);
}