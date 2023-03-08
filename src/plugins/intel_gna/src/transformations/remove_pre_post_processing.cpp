// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/remove_pre_post_processing.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "openvino/pass/pass.hpp"

using namespace ov::opset9;
using namespace ov::intel_gna::pass;

namespace {

ov::Shape SqueezeShape(const ov::Shape& shape) {
    ov::Shape squeezed_shape;
    std::copy_if(shape.begin(), shape.end(), std::back_inserter(squeezed_shape), [](size_t x) {
        return x != 1;
    });
    return squeezed_shape;
}

bool IsPreprocessingLayerSuppported(std::shared_ptr<ngraph::Node>& layer) {
    // Gather layers are not supported by GNA and have to be executed on CPU
    if (std::dynamic_pointer_cast<ov::opset1::Gather>(layer)) {
        return true;
    }

    // 2-d Transposes layers can be executed on GNA
    if (std::dynamic_pointer_cast<ov::opset1::Transpose>(layer)) {
        const ov::Shape squeezed_shape = SqueezeShape(layer->get_shape());
        const size_t min_input_dim = std::min(squeezed_shape[0], squeezed_shape[1]);
        const size_t max_input_dim = std::max(squeezed_shape[0], squeezed_shape[1]);

        if (squeezed_shape.size() > 2) {
            return true;
        } else if (min_input_dim > 8) {
            return true;
        } else if (ALIGN(max_input_dim, limitations::noOfInputsDivisor) != max_input_dim) {
            // TODO: need to test gna_config.gnaFlags.input_low_precision
            return true;
        } else {
            log::trace() << "Input shape with rank: " << squeezed_shape.size() << " is not required to be transposed"
                         << std::endl;
        }
    }

    return false;
}

/*
  works only if we have one date input and one output
 */
void RemoveSingleInputNodeFromFunction(std::shared_ptr<ov::Node> node) {
    const ov::Shape input_node_shape = node->get_input_shape(0);
    const ov::Shape output_node_shape = node->get_output_shape(0);

    std::shared_ptr<ov::Node> node_parent = node->get_input_node_shared_ptr(0);
    if (!std::equal(input_node_shape.begin(), input_node_shape.end(), output_node_shape.begin())) {
        auto reshape_const_node =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{output_node_shape.size()}, output_node_shape);
        node_parent = std::make_shared<Reshape>(node_parent, reshape_const_node, false);
    }

    ov::replace_output_update_name(node->output(0), node_parent->output(0));
}

/*
  Support only one data node as 0 input
 */
std::shared_ptr<ov::Model> CopySingleInputNodeFromFunction(std::shared_ptr<ov::Node> node) {
    const ov::element::Type& input_type = node->get_input_element_type(0);
    const ov::Shape& input_shape = node->get_input_shape(0);

    auto param = std::make_shared<Parameter>(input_type, input_shape);
    ov::OutputVector input_nodes = node->input_values();
    input_nodes[0] = param;
    auto node_copy = node->clone_with_new_inputs(input_nodes);
    auto result = std::make_shared<Result>(node_copy);
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    // ov::pass::Manager manager;
    // manager.register_pass<ov::pass::Serialize>("pre_model.xml", "pre_model.bin");
    // manager.run_passes(model);

    return model;
}

}  // namespace

bool RemoveInputsProcessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(RemoveInputsProcessing);
    bool result = false;
    for (const auto& param_node : model->inputs()) {
        for (auto& param_target : param_node.get_target_inputs()) {
            auto target_node = param_target.get_node()->shared_from_this();
            // Parameter -> Transpose, Parameter -> Gather
            if (IsPreprocessingLayerSuppported(target_node)) {
                if (m_subgraph_cpu_map) {
                    m_subgraph_cpu_map->emplace(param_node.get_node_shared_ptr()->get_friendly_name(),
                                                CopySingleInputNodeFromFunction(target_node));
                }
                RemoveSingleInputNodeFromFunction(target_node);
                result = true;
            }
        }
    }
    return result;
}

bool RemoveOutputsProcessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(RemoveOutputsProcessing);
    bool result = false;
    for (std::shared_ptr<ov::Node> r_node : model->get_results()) {
        for (auto& r_input : r_node->input_values()) {
            auto r_input_node = r_input.get_node_shared_ptr();
            // Transpose -> Result, Gather -> Result
            if (IsPreprocessingLayerSuppported(r_input_node)) {
                if (m_subgraph_cpu_map) {
                    m_subgraph_cpu_map->emplace(r_input_node->get_friendly_name(),
                                                CopySingleInputNodeFromFunction(r_input_node));
                }
                RemoveSingleInputNodeFromFunction(r_input_node);
                result = true;
            }
        }
    }
    return result;
}