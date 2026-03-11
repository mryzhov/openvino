// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pass_config.hpp"

ov::pass::PassConfig::PassConfig() {
    m_callback = [](const std::shared_ptr<const ::ov::Node>&) {
        return false;
    };
}

ov::pass::param_callback ov::pass::PassConfig::get_callback(const DiscreteTypeInfo& type_info) const {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    const auto& it = m_callback_map.find(type_info);
    if (it != m_callback_map.end()) {
        return it->second;
    } else {
        return m_callback;
    }
}

void ov::pass::PassConfig::enable(const ov::DiscreteTypeInfo& type_info) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_disabled.erase(type_info);
    m_enabled.insert(type_info);
}

void ov::pass::PassConfig::disable(const ov::DiscreteTypeInfo& type_info) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_enabled.erase(type_info);
    m_disabled.insert(type_info);
}

void ov::pass::PassConfig::add_disabled_passes(const PassConfig& rhs) {
    if (this == &rhs) {
        return;
    }

    std::scoped_lock lock(m_config_mutex, rhs.m_config_mutex);
    for (const auto& pass : rhs.m_disabled) {
        if (m_enabled.count(pass))
            continue;
        m_disabled.insert(pass);
    }
}

ov::pass::PassConfig::PassConfig(const PassConfig& other) {
    std::lock_guard<std::mutex> lock(other.m_config_mutex);
    m_callback = other.m_callback;
    m_callback_map = other.m_callback_map;
    m_disabled = other.m_disabled;
    m_enabled = other.m_enabled;
    // m_config_mutex is default-initialized as a fresh mutex
}

ov::pass::PassConfig::PassConfig(PassConfig&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.m_config_mutex);
    m_callback = std::move(other.m_callback);
    m_callback_map = std::move(other.m_callback_map);
    m_disabled = std::move(other.m_disabled);
    m_enabled = std::move(other.m_enabled);
    // m_config_mutex is default-initialized as a fresh mutex
}

ov::pass::PassConfig& ov::pass::PassConfig::operator=(const PassConfig& other) {
    if (this != &other) {
        std::scoped_lock lock(m_config_mutex, other.m_config_mutex);
        m_callback = other.m_callback;
        m_callback_map = other.m_callback_map;
        m_disabled = other.m_disabled;
        m_enabled = other.m_enabled;
    }
    return *this;
}

ov::pass::PassConfig& ov::pass::PassConfig::operator=(PassConfig&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_config_mutex, other.m_config_mutex);
        m_callback = std::move(other.m_callback);
        m_callback_map = std::move(other.m_callback_map);
        m_disabled = std::move(other.m_disabled);
        m_enabled = std::move(other.m_enabled);
    }
    return *this;
}
