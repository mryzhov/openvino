// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/convert_floor_to_add.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

#include "backend/gna_limitations.hpp"
#include "ops/diagonal.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ConvertFloorToAdd, "ConvertFloorToAdd", 0);

ConvertFloorToAdd::ConvertFloorToAdd() {
    MATCHER_SCOPE(ConvertFloorToAdd);

    auto read_value = ngraph::pattern::wrap_type<ngraph::opset8::ReadValue>();
    auto const1 = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fq_const1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({const1, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto fq_rv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({read_value, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto multiply1 = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({fq_rv, fq_const1});

    auto floor = ngraph::pattern::wrap_type<ngraph::opset8::Floor>({multiply1});
    auto fq_floor = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({floor, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto const2 = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fq_const2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({const2, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto multiply2 = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({fq_floor, fq_const2});
    auto fq_mul2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({multiply2, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto mul3_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto multiply3 = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({multiply2, mul3_const});
    auto add1 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({ngraph::pattern::any_input(), multiply3});
    auto fq_add1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({add1, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto multiply4 = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({fq_add1, ngraph::pattern::any_input()});
    auto fq_mul4 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({multiply4, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto add2 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({fq_mul4, ngraph::pattern::any_input()});
    auto fq_add2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({add2, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto multiply5 = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({fq_mul2, ngraph::pattern::any_input()});
    auto fq_mul5 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({multiply5, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto add3 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({fq_mul5, fq_add2});
    auto fq_add3 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({add3, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto assign = ngraph::pattern::wrap_type<ngraph::opset8::Assign>({fq_add3});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto floor_node = pattern_map.at(floor).get_node_shared_ptr();
        auto first_fq_node = pattern_map.at(fq_rv).get_node_shared_ptr();
        auto last_fq_node = pattern_map.at(fq_add3).get_node_shared_ptr();
        auto fq_add1_node = pattern_map.at(fq_add1).get_node_shared_ptr();
        auto mu2_node = pattern_map.at(multiply2).get_node_shared_ptr();
        auto add1_node = pattern_map.at(add1).get_node_shared_ptr();
        auto add3_node = pattern_map.at(add3).get_node_shared_ptr();
        auto mul5_node = pattern_map.at(multiply5).get_node_shared_ptr();

        std::cout << "Floor recognized: " << floor_node->get_friendly_name() << "\n";

        auto first_fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(first_fq_node);
        first_fq->set_levels(std::numeric_limits<uint32_t>::max());
        auto last_fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(last_fq_node);
        last_fq->set_levels(std::numeric_limits<uint32_t>::max());

        auto shape = first_fq->get_output_shape(0);
        size_t size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());

        auto constant_zero1 = ngraph::opset8::Constant::create(first_fq->get_element_type(),
            shape, std::vector<float>(size, 0.0));
        auto constant_zero2 = ngraph::opset8::Constant::create(first_fq->get_element_type(),
            shape, std::vector<float>(size, 0.0));
        auto diagonal1 = std::make_shared<Diagonal>(constant_zero1, constant_zero2, first_fq);
        auto diagonal1_fq = fq_add1_node->clone_with_new_inputs(ngraph::OutputVector{diagonal1, fq_add1_node->input_value(1),
            fq_add1_node->input_value(2), fq_add1_node->input_value(3), fq_add1_node->input_value(4)});

        copy_runtime_info(mu2_node, ngraph::NodeVector{diagonal1, diagonal1, diagonal1_fq});
        replace_node(mu2_node, diagonal1_fq);

        auto constant_neg = ngraph::opset8::Constant::create(first_fq->get_element_type(),
            shape, std::vector<float>(size, -1.0));
        auto diagonal3 = std::make_shared<Diagonal>(diagonal1_fq, constant_neg, first_fq->input_value(0));

        /*auto last_fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(last_fq_node);
        auto floor_const_node = pattern_map.at(const1).get_node_shared_ptr();
        auto floor_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(floor_const_node);
        IE_ASSERT(last_fq != nullptr && floor_const != nullptr);

        auto floor_const_vector = floor_const->get_vector<float>();
        float floor_fq_max = (last_fq->get_levels() - 1) / (2 * floor_const_vector.front());
        float floor_fq_min = -floor_fq_max;
        auto floor_fq_max_node = ngraph::opset8::Constant::create(first_fq->get_element_type(), {}, &floor_fq_max);
        auto floor_fq_min_node = ngraph::opset8::Constant::create(first_fq->get_element_type(), {}, &floor_fq_min);
        auto fq_floor = last_fq_node->clone_with_new_inputs(ngraph::OutputVector{identity, last_fq_node->input_value(1),
            last_fq_node->input_value(2), last_fq_node->input_value(3), last_fq_node->input_value(4)});*/

        copy_runtime_info(add1_node, ngraph::NodeVector{diagonal3});
        replace_node(add1_node, diagonal3);

        auto diagonal4 = std::make_shared<Diagonal>(mul5_node->input_value(0), mul5_node->input_value(1), add3_node->input_value(1));
        copy_runtime_info(add3_node, ngraph::NodeVector{diagonal4});
        replace_node(add3_node, diagonal4);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(assign, matcher_name);
    this->register_matcher(m, callback);
}