// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {

void validate_ports(const ExpressionPtr& expr) {
    auto validate_descriptor = [](const PortDescriptorPtr& desc) {
        const auto& shape = desc->get_shape();
        const auto& layout = desc->get_layout();
        const auto max_dim = *std::max_element(layout.begin(), layout.end());
        OPENVINO_ASSERT(max_dim < shape.size(), "Max layout index can't be larger than the shape size");
        OPENVINO_ASSERT(shape.size() == layout.size(), "Shape and layout must have the same length");
    };
    const auto& in_descs = expr->get_input_port_descriptors();
    const auto& out_descs = expr->get_output_port_descriptors();
    std::for_each(in_descs.cbegin(), in_descs.cend(), validate_descriptor);
    std::for_each(out_descs.cbegin(), out_descs.cend(), validate_descriptor);
}

void validate_parameter(const ExpressionPtr& expr, const LinearIR& linear_ir) {
    OPENVINO_ASSERT(ov::is_type<ov::op::v0::Parameter>(expr->get_node()),
                    "Parameter validation expects Parameter op");
    const auto& shape_infer_seq = utils::get_first_child_shape_infer_expr_seq(expr);
    const auto& expr_val = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
    auto consumer_inputs = expr_val->get_output_port_connector(0)->get_consumers();
    std::set<std::vector<size_t>> layouts;
    for (const auto& consumer_input : consumer_inputs) {
        const auto& node = consumer_input.get_expr()->get_node();
        if (const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(node)) {
            OPENVINO_ASSERT(ma->is_memory_access_input_port(consumer_input.get_index()),
                            "Parameter expects MemoryAccess on output");
            layouts.insert(consumer_input.get_descriptor_ptr()->get_layout());
        } else {
            OPENVINO_ASSERT(ov::is_type<op::LoopEnd>(node), "Parameter must be connected to MemoryAccess op or LoopEnd");
        }
    }
    OPENVINO_ASSERT(layouts.size() == 1, "All consumers of Parameter must have the same layout");
}

void validate_result(const ExpressionPtr& expr, const LinearIR& linear_ir) {
    OPENVINO_ASSERT(ov::is_type<ov::op::v0::Result>(expr->get_node()),
                    "Result validation expects Result op");
    const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(expr);
    const auto& expr_val = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
    const auto source = expr_val->get_input_port_connector(0)->get_source();
    const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(source.get_expr()->get_node());
    OPENVINO_ASSERT(ma && ma->is_memory_access_output_port(source.get_index()),
                    "Result expects MemoryAccess parent");
}

void validate_buffer(const ExpressionPtr& expr, const LinearIR& linear_ir) {
    OPENVINO_ASSERT(ov::is_type<op::Buffer>(expr->get_node()),
                    "Buffer validation expects Buffer op");
    for (const auto& input : expr->get_input_port_connectors()) {
        const auto& source = input->get_source();
        const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(source.get_expr()->get_node());
        OPENVINO_ASSERT(ma && ma->is_memory_access_input_port(source.get_index()),
                    "Buffer expects MemoryAccess parent");
        const auto buffer_siblings = input->get_consumers();
        for (const auto& buffer_sibling : buffer_siblings) {
            const auto& buffer_sibling_expr = buffer_sibling.get_expr();
            OPENVINO_ASSERT(buffer_sibling_expr == expr || ov::is_type<op::LoopEnd>(buffer_sibling_expr->get_node()),
                            "Buffer can have only LoopEnd siblings!");
        }
    }

    const auto& shape_infer_seq = utils::get_first_child_shape_infer_expr_seq(expr);
    const auto& expr_val = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
    const auto& out = expr_val->get_output_port_connector(0);
    const auto consumers = out->get_consumers();
    for (const auto& consumer_input : consumers) {
        const auto& node = consumer_input.get_expr()->get_node();
        if (const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(node)) {
            OPENVINO_ASSERT(ma->is_memory_access_input_port(consumer_input.get_index()),
                            "Buffer expects MemoryAccess and LoopEnd on output");
        } else {
            OPENVINO_ASSERT(ov::is_type<op::LoopEnd>(node), "Buffer expects MemoryAccess and LoopEnd on output");
        }
    }
}

void validate_loop_end(const ExpressionPtr& expr, const LinearIR& linear_ir) {
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
    OPENVINO_ASSERT(loop_end, "LoopEnd validation expects LoopEnd op");
    OPENVINO_ASSERT(loop_end->get_loop_begin() != nullptr,
                    "LoopEnd must be connected to the LoopBegin");

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_end->get_id());
    OPENVINO_ASSERT(loop_info->get_work_amount() == loop_end->get_work_amount() &&
                    loop_info->get_increment() == loop_end->get_increment(),
                    "Incompatible LoopEnd and the corresponding LoopInfo");

    const auto input_port_infos = loop_info->get_input_ports_info();
    const auto output_port_infos = loop_info->get_output_ports_info();
    OPENVINO_ASSERT(input_port_infos.size() == loop_end->get_input_num() &&
                    output_port_infos.size() == loop_end->get_output_num(),
                    "Incompatible LoopEnd and the corresponding LoopInfo");

    const auto& is_incremented = loop_end->get_is_incremented();
    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& final_offsets = loop_end->get_finalization_offsets();
    auto validate_loop_ports = [&](const std::vector<UnifiedLoopInfo::LoopPortInfo>& loop_port_infos, size_t shift = 0) {
        for (size_t i = 0; i < loop_port_infos.size(); ++i) {
            OPENVINO_ASSERT(is_incremented[i + shift] == loop_port_infos[i].port.is_incremented &&
                            ptr_increments[i + shift] == loop_port_infos[i].desc.ptr_increment &&
                            final_offsets[i + shift] == loop_port_infos[i].desc.finalization_offset,
                            "Incompatible data ptr shifts in LoopEnd and the corresponding LoopInfo");
        }
    };
    validate_loop_ports(input_port_infos);
    validate_loop_ports(output_port_infos, loop_end->get_input_num());
}

// TODO [143395] : Extract this validation checks to the separate `ValidateBuffers` pass
void validate_buffer_expressions(const LinearIR::container& buffer_expressions) {
    std::set<size_t> cluster_ids;
    std::map<size_t, std::set<lowered::ExpressionPtr>> dynamic_buffer_clusters, static_buffer_clusters;

    for (const auto& buffer_expr : buffer_expressions) {
        const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Expected Buffer ops in Buffer expressions of LinearIR");

        // TODO [143395] : MemoryManager should provide exact containers with needed buffers (static or dynamic) without any `is_defined()`
        auto& clusters = buffer->is_defined() ? static_buffer_clusters : dynamic_buffer_clusters;
        clusters[buffer->get_cluster_id()].insert(buffer_expr);
        cluster_ids.insert(buffer->get_cluster_id());
    }

    OPENVINO_ASSERT(cluster_ids.size() == dynamic_buffer_clusters.size() + static_buffer_clusters.size(), "Incorrect count of Buffer clusters");
    OPENVINO_ASSERT(cluster_ids.empty() || (*cluster_ids.cbegin() == 0 && *cluster_ids.crbegin() == (cluster_ids.size() - 1)),
                    "Incorrect indetifiers of Buffer clusters");

    for (const auto& p : static_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;
        OPENVINO_ASSERT(dynamic_buffer_clusters.count(cluster_id) == 0, "Buffers from the same cluster must be only static or dynamic");

        OPENVINO_ASSERT(cluster.size() > 0, "Incorrect size of buffer cluster");
        size_t cluster_offset = ov::as_type_ptr<op::Buffer>((*cluster.cbegin())->get_node())->get_offset();
        for (const auto& buffer_expr : cluster) {
            OPENVINO_ASSERT(cluster_offset == ov::as_type_ptr<op::Buffer>(buffer_expr->get_node())->get_offset(),
                            "Static Buffers from the same cluster must have the same offset!");
        }
    }
}
} // namespace

Validate::Validate() {
    m_validation_map = {
        {ov::op::v0::Parameter::get_type_info_static(), validate_parameter},
        {ov::op::v0::Result::get_type_info_static(), validate_result},
        {ov::snippets::op::Buffer::get_type_info_static(), validate_buffer},
        {ov::snippets::op::LoopEnd::get_type_info_static(), validate_loop_end},
    };
}

bool Validate::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Validate")

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto expr = *expr_it;
        const auto node = expr->get_node();
        const auto found = m_validation_map.find(node->get_type_info());
        if (found != m_validation_map.cend()) {
            (found->second)(expr, linear_ir);
        }
        expr->validate();
        // Loop expr doesn't have shapes and layouts
        if (!ov::is_type<op::LoopBase>(node))
            validate_ports(expr);
    }

    validate_buffer_expressions(linear_ir.get_buffers());

    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
