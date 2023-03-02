#include "transformations/common_optimizations/transpose_sinking_unary.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov;
using namespace ov::opset9;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace transpose_sinking;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

/**
 * @brief SwapNodes allows to perform swapping nodes even if there are more than one consumers but has less performance
 *
 * @param first_node first node pointer
 * @param second_node first node pointer
 * @return NodePair pair of nodes in new order that allows to register them in MatcherPass
 */
NodePair SwapNodes(NodePtr first_node, NodePtr second_node) {
    auto second_node_inputs = second_node->input_values();
    second_node_inputs[0] = first_node->input_value(0);

    auto new_first_node = second_node->clone_with_new_inputs(second_node_inputs);

    auto first_node_inputs = first_node->input_values();
    first_node_inputs[0] = new_first_node;
    auto new_second_node = first_node->clone_with_new_inputs(first_node_inputs);

    new_second_node->set_friendly_name(second_node->get_friendly_name());
    ov::copy_runtime_info({first_node, second_node}, {new_first_node, new_second_node});

    ov::replace_node(second_node, new_second_node);

    return std::make_pair(new_first_node, new_second_node);
}

/**
 * @brief SwapOutputs has much better performance than SwapNodes and covers the most of the real situations
 *        but cannot work when the consumers count greater than one
 * @param first_node first node pointer
 * @param second_node second node pointer
 * @return NodePair pair of nodes in new order that allows to register them in MatcherPass
 */
NodePair SwapOutputs(NodePtr first_node, NodePtr second_node) {
    const auto first_node_output_names = first_node->output(0).get_names();
    const auto second_node_output_names = second_node->output(0).get_names();

    auto swap_names = [&]() {
        const std::string first_name = first_node->get_friendly_name();
        first_node->set_friendly_name(second_node->get_friendly_name());
        second_node->set_friendly_name(first_name);

        first_node->output(0).set_names(second_node_output_names);
        second_node->output(0).set_names(first_node_output_names);
    };

    auto out_1 = first_node->input_value(0);
    second_node->input(0).replace_source_output(out_1);

    auto out_2 = second_node->output(0);
    second_node->output(0).replace(first_node->output(0));

    first_node->input(0).replace_source_output(out_2);

    swap_names();

    return std::make_pair(second_node, first_node);
}

NodePair Swap(NodePtr first_node, NodePtr second_node) {
    NodePair new_nodes;

    if (first_node->output(0).get_target_inputs().size() > 1 || second_node->output(0).get_target_inputs().size() > 1)
        new_nodes = SwapNodes(first_node, second_node);
    else
        new_nodes = SwapOutputs(first_node, second_node);

    return new_nodes;
}

NodePtr GetPatternNode(const PatternValueMap& pattern_map, const NodeVector& node_labels) {
    for (const auto& node_label : node_labels) {
        if (pattern_map.count(node_label))
            return pattern_map.at(node_label).get_node_shared_ptr();
    }
    return {};
}

}  // namespace

ov::pass::TransposeSinkingUnaryForward::TransposeSinkingUnaryForward() {
    MATCHER_SCOPE(TransposeSinkingUnaryForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), any_input()});
    auto fq_label = wrap_type<FakeQuantize>({transpose_label, any_input(), any_input(), any_input(), any_input()});
    auto unary_op_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({transpose_label});
    auto unary_label = std::make_shared<pattern::op::Or>(OutputVector{fq_label, unary_op_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = GetPatternNode(pattern_to_output, NodeVector{unary_op_label, fq_label});

        const NodePair new_nodes = Swap(transpose, unary);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        UpdateForwardSinkingAbility(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(unary_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

namespace {
bool IfSinkingEnabled(const Output<Node>& output) {
    return is_sinking_node(output.get_node_shared_ptr());
}
}  // namespace

ov::pass::TransposeSinkingUnaryBackwardSingleConsumer::TransposeSinkingUnaryBackwardSingleConsumer() {
    MATCHER_SCOPE(TransposeSinkingUnaryBackwardSingleConsumer);

    auto fq_label =
        wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()}, consumers_count(1));
    auto unary_op_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({any_input()},
                                                                                         consumers_count(1));
    auto unary_label = std::make_shared<pattern::op::Or>(OutputVector{fq_label, unary_op_label});
    auto transpose_label = wrap_type<Transpose>({unary_label, any_input()}, IfSinkingEnabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = GetPatternNode(pattern_to_output, NodeVector{unary_op_label, fq_label});

        const NodePair new_nodes = Swap(unary, transpose);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

namespace {
std::function<bool(Output<Node>)> consumers_more_than(size_t n) {
    return [=](Output<Node> output) -> bool {
        return output.get_target_inputs().size() > n;
    };
}
}  // namespace

ov::pass::TransposeSinkingUnaryBackwardMultiConsumers::TransposeSinkingUnaryBackwardMultiConsumers() {
    MATCHER_SCOPE(TransposeSinkingUnaryBackwardMultiConsumers);

    auto unary_restrictions = [](const Output<Node>& output) -> bool {
        return consumers_more_than(1)(output) && HasSameOutputTransposeNodes(output);
    };

    auto fq_label =
        wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()}, unary_restrictions);
    auto unary_op_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({any_input()},
                                                                                         unary_restrictions);
    auto unary_label = std::make_shared<pattern::op::Or>(OutputVector{fq_label, unary_op_label});

    auto transpose_const_label = wrap_type<Constant>();

    auto transpose_label = wrap_type<Transpose>({unary_label, transpose_const_label}, IfSinkingEnabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = GetPatternNode(pattern_to_output, NodeVector{unary_op_label, fq_label});

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(unary,
                                                                       transpose_const,
                                                                       /* input_indexes */ {0})) {
            register_new_node(new_node);
        }

        // remove output transposes
        RemoveSingleOutputConsumers(unary);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeSinkingUnaryBackward::TransposeSinkingUnaryBackward() {
    MATCHER_SCOPE(TransposeSinkingUnaryBackward);
    add_matcher<ov::pass::TransposeSinkingUnaryBackwardSingleConsumer>();
    add_matcher<ov::pass::TransposeSinkingUnaryBackwardMultiConsumers>();
}
