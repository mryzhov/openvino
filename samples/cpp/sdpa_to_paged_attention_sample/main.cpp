// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/serialize.hpp"

#include "samples/slog.hpp"
// clang-format on

namespace {

std::vector<std::string> split_devices(const std::string& devices) {
    std::vector<std::string> result;
    std::stringstream ss(devices);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

std::string sanitize_for_filename(const std::string& value) {
    std::string sanitized = value;
    for (char& ch : sanitized) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return sanitized;
}

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable << " <path_to_model> [device|device,device]" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: The transformed model contains custom extension operations (PagedCausalConv1D," << std::endl;
    std::cout << "      PagedGatedDeltaNet) that require specialized plugin support." << std::endl;
    std::cout << "      The transformed IR is saved as 'sdpa_to_paged_attention_sample.{xml,bin}'." << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;

        if (argc < 2 || argc > 3) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        const std::string model_path = argv[1];
        const std::string devices_arg = argc == 3 ? argv[2] : "CPU";
        const std::vector<std::string> devices = split_devices(devices_arg);
        OPENVINO_ASSERT(!devices.empty(), "No devices were provided.");

        ov::Core core;

        slog::info << "Reading model: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        slog::info << "Applying SDPAToPagedAttention transformation" << slog::endl;
        ov::pass::Manager manager("sdpa_to_paged_attention_sample");
        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        manager.register_pass<ov::pass::Serialize>("sdpa_to_paged_attention_sample.xml",
                                                   "sdpa_to_paged_attention_sample.bin");
        manager.run_passes(model);
        model->validate_nodes_and_infer_types();

        slog::info << "Note: Transformed model saved as sdpa_to_paged_attention_sample.{xml,bin}" << slog::endl;

        slog::info << "Attempting to compile transformed model..." << slog::endl;
        slog::info << "Note: Compilation may fail if the device does not support custom extension operations."
                   << slog::endl;

        for (const auto& device : devices) {
            slog::info << "Compiling for device: " << device << slog::endl;
            ov::CompiledModel compiled_model = core.compile_model(model, device);
            slog::info << "Compilation succeeded for device: " << device << slog::endl;

            const std::shared_ptr<const ov::Model> runtime_model = compiled_model.get_runtime_model();
            const std::string device_suffix = sanitize_for_filename(device);
            const std::string runtime_xml = "sdpa_to_paged_attention_runtime_" + device_suffix + ".xml";
            const std::string runtime_bin = "sdpa_to_paged_attention_runtime_" + device_suffix + ".bin";

            ov::serialize(runtime_model, runtime_xml, runtime_bin);
            slog::info << "Post-compile runtime model saved as " << runtime_xml << " and " << runtime_bin << slog::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}