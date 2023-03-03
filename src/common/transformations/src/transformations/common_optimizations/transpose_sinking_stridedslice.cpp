// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_stridedslice.hpp"

#include <openvino/pass/pattern/op/or.hpp>

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov::pass::pattern;
using namespace ov;
using namespace ov::opset10;
using namespace transpose_sinking;

ov::pass::TransposeSinkingStridedSliceForward::TransposeSinkingStridedSliceForward() {
    MATCHER_SCOPE(TransposeSinkingStridedSliceForward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), any_input()}, consumers_count(1));
    auto stridedslice_label = wrap_type<StridedSlice>({transpose_label, any_input(), any_input(), any_input()}, consumers_count(1));

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label).get_node_shared_ptr());
        auto stridedslice = as_type_ptr<StridedSlice>(pattern_to_output.at(stridedslice_label).get_node_shared_ptr());

        // remove Transpose on data input
        {
            auto transpose_parent = stridedslice->input_value(0).get_node()->input_value(0);
            stridedslice->input(0).replace_source_output(transpose_parent);
        }

        // TODO: fix StridedSlice input constants
        // TODO: fix new_axis_mask, shrink_axis_mask, ellipsis_mask

        TransposeInputsInfo transpose_input_info = {transpose, transpose_const, 0};
        for (auto& new_node : sink_forward::InsertOutputTransposes(stridedslice, transpose_input_info)) {
            register_new_node(new_node);
            transpose_sinking::UpdateForwardSinkingAbility(new_node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(stridedslice_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeSinkingStridedSliceBackward::TransposeSinkingStridedSliceBackward() {
    MATCHER_SCOPE(TransposeSinkingStridedSliceBackward);

    auto stridedslice_label = wrap_type<StridedSlice>({any_input(), any_input(), any_input(), any_input()}, [](const Output<Node>& output) -> bool {
            return has_static_rank()(output) && is_sinking_node(output) && consumers_count(1)(output);
        });
    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({stridedslice_label, any_input()}, consumers_count(1));

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label).get_node_shared_ptr());
        auto stridedslice = as_type_ptr<StridedSlice>(pattern_to_output.at(stridedslice_label).get_node_shared_ptr());

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(stridedslice,
                                                                       transpose_const,
                                                                       /* input_indexes= */ {0})) {
            register_new_node(new_node);
        }

        // TODO: fix StridedSlice constants
        // TODO: fix new_axis_mask, shrink_axis_mask, ellipsis_mask

        // remove output transposes
        RemoveSingleOutputConsumers(stridedslice);

        return true;
    };

    auto m = std::make_shared<Matcher>(stridedslice_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
