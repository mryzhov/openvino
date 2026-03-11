// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <typeinfo>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

namespace ov {
namespace pass {
/**
 * @brief Manager class allows to manage transformation passes
 * @ingroup ov_pass_cpp_api
 *
 * ## Thread Safety
 * 
 * The Manager class is thread-safe for multi-threaded usage, but execution on a single
 * Manager instance is serialized.
 *
 * This is achieved through:
 *
 * 1. **Registration Synchronization**: register_pass(), register_pass_instance(), and
 *    set_per_pass_validation() use a mutex to protect manager state.
 *
 * 2. **Serialized Execution**: run_passes() is protected by the same mutex because registered
 *    pass instances are mutable and not safe for concurrent execution on the same Manager.
 *
 * 3. **Thread-Safe Profiling**: Static counters used for pass profiling and visualization use
 *    std::atomic to prevent data races in multi-threaded scenarios.
 *
 * ## Usage Example
 *
 *     // Safe to use from multiple threads (execution is serialized per Manager):
 *     pass::Manager manager;
 *     manager.register_pass<SomePass>();
 *     std::thread t1([&]() { manager.run_passes(model1); });
 *     std::thread t2([&]() { manager.run_passes(model2); });
 *     t1.join();
 *     t2.join();
 */
class OPENVINO_API Manager {
public:
    Manager();
    virtual ~Manager();

    Manager(const Manager& other);
    Manager& operator=(const Manager& other);
    Manager(Manager&& other) noexcept;
    Manager& operator=(Manager&& other) noexcept;

    //// \brief Construct Manager with a provided name.
    explicit Manager(std::string name);

    //// \brief Construct Manager with shared PassConfig instance
    explicit Manager(std::shared_ptr<PassConfig> pass_config, std::string name = "UnnamedManager");

    //// \brief Construct Manager with a copied PassConfig instance; it will not share PassConfig as in the constructor
    /// above
    explicit Manager(const PassConfig& pass_config, std::string name = "UnnamedManager");

    /// \brief Register given transformation class type to execution list
    /// Example below show the basic usage of pass::Manager
    ///
    ///     pass::Manager manager;
    ///     manager.register_pass<MyTransformation>(/* transformation constructor args */);
    ///     manager.run_passes(f);
    ///
    /// For some purposes transformation can be registered and disabled by default.
    ///
    ///     manager.register_pass<MyTransformation, false>();
    ///
    /// \return shared_ptr to the transformation instance
    /// \note This method is thread-safe. Calls can be concurrent with run_passes(),
    ///       but run_passes() executes under manager-level serialization.
    template <typename T, bool Enable = true, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        std::lock_guard<std::mutex> lock(m_registration_mutex);
        auto rc = push_pass<T>(std::forward<Args>(args)...);
        rc->set_pass_config(m_pass_config);
        if (m_per_pass_validation && T::get_type_info_static() != Validate::get_type_info_static()) {
            push_validate_pass();
        }
        if (!Enable && !m_pass_config->is_enabled<T>()) {
            m_pass_config->disable<T>();
        }
        return rc;
    }

    /// \brief Register a transformation pass instance to execution list
    /// \param pass Shared pointer to PassBase instance
    /// \return shared_ptr to the registration instance
    /// \note This method is thread-safe. Calls can be concurrent with run_passes(),
    ///       but run_passes() executes under manager-level serialization.
    std::shared_ptr<PassBase> register_pass_instance(std::shared_ptr<PassBase> pass) {
        std::lock_guard<std::mutex> lock(m_registration_mutex);
        pass->set_pass_config(m_pass_config);
        m_pass_list.push_back(pass);
        if (m_per_pass_validation && !ov::as_type_ptr<Validate>(pass)) {
            push_validate_pass();
        }
        return pass;
    }

    /// \brief      Runs registered transformations on a given model
    ///
    /// \param      model Input model
    ///
    /// \return     Returns true if the model was changed by transformations,
    ///             false otherwise.
    bool run_passes(const std::shared_ptr<Model>& model);

    /// \brief Set flag to enable/disable running Validate pass after executing
    /// each registered pass
    /// \param new_state Value "true" enables Validate pass run; "false", otherwise
    /// \note This method is thread-safe. However, changes take effect only for subsequent run_passes calls.
    void set_per_pass_validation(bool new_state);

    /// \return PassConfig shared object. This object is used for transformations pipeline
    /// configuration.
    /// This object allows to disable/enable transformations execution, set callback to
    /// particular
    /// transformation. For more details see PassConfig class.
    std::shared_ptr<PassConfig> get_pass_config() {
        return m_pass_config;
    }

protected:
    template <typename T, class... Args>
    std::shared_ptr<T> push_pass(Args&&... args) {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        m_pass_list.push_back(pass);
        return pass;
    }

    virtual void push_validate_pass() {
        push_pass<Validate>();
    }

    std::shared_ptr<PassConfig> m_pass_config;
    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    bool m_per_pass_validation = true;
    std::string m_name = "UnnamedManager";

private:
    /// \brief Thread-safety mutex protecting m_pass_list mutations and m_per_pass_validation.
    /// Acquired during register_pass and run_passes to ensure safe concurrent access.
    mutable std::mutex m_registration_mutex;

    bool run_pass(const std::shared_ptr<PassBase>& pass, const std::shared_ptr<Model>& model);
};
}  // namespace pass
}  // namespace ov
