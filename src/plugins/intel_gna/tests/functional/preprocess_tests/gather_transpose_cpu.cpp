// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include <vector>
#include <map>
#include <numeric>

#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace InferenceEngine;
using namespace ov::opset9;
using namespace ov::test;

namespace LayerTestsDefinitions {

namespace {

std::vector<size_t> MakeIndexes(size_t size) {
    std::vector<size_t> indexes(size);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::reverse(indexes.begin(), indexes.end());
    return indexes;
}

}  // namespace

typedef std::tuple<std::vector<size_t>,                // Input shape
                   ov::element::Type,                  // Net precision
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Configuration
                   >
    preprocessTestParamsSet;

class PreprocessBaseTest : public testing::WithParamInterface<preprocessTestParamsSet>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessTestParamsSet>& obj) {
        std::vector<size_t> input_shape;
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf;

        std::tie(input_shape, net_type, target_device, conf) = obj.param;

        std::ostringstream result;
        result << "Shape=" << CommonTestUtils::vec2str(input_shape) << "_";
        result << "netPRC=" << net_type << "_";
        result << "trgDev=" << target_device;
        for (auto const& conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }
};

class RemoveGatherInput : public PreprocessBaseTest {
protected:
    void SetUp() override {
        ov::element::Type net_type;
        std::vector<size_t> input_shape;
        std::map<std::string, std::string> conf;
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();

        std::tie(input_shape, net_type, targetDevice, conf) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({input_shape, input_shape});
        configuration.insert(conf.begin(), conf.end());

        init_input_shapes(input_shapes);

        auto params = ngraph::builder::makeParams(net_type, {input_shape});
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        const std::vector<size_t> indexes = MakeIndexes(input_shape_size);
        auto gather_indexes = Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = 1;
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(params[0], gather_indexes, gather_axis_const);

        auto mul_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(gather_node, mul_input_const);

        auto add_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        ov::ResultVector results{std::make_shared<Result>(add_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveGatherOutput : public PreprocessBaseTest {
protected:
    void SetUp() override {
        ov::element::Type net_type;
        std::vector<size_t> input_shape;
        std::map<std::string, std::string> conf;
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();

        std::tie(input_shape, net_type, targetDevice, conf) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({input_shape, input_shape});
        configuration.insert(conf.begin(), conf.end());

        init_input_shapes(input_shapes);

        auto params = ngraph::builder::makeParams(net_type, {input_shape});

        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto mul_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(params[0], mul_input_const);

        auto add_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        const std::vector<size_t> indexes = MakeIndexes(input_shape_size);
        auto gather_indexes = Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = 1;
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(add_node, gather_indexes, gather_axis_const);

        ov::ResultVector results{std::make_shared<Result>(gather_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveTransposeInput : public PreprocessBaseTest {
protected:
    void SetUp() override {
        ov::element::Type net_type;
        std::vector<size_t> input_shape;
        std::map<std::string, std::string> conf;
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();

        std::tie(input_shape, net_type, targetDevice, conf) = this->GetParam();
        std::vector<size_t> transpose_order;
        switch (input_shape.size())
        {
        case 2:
            transpose_order = {1, 0};
            break;
        case 3:
            transpose_order = {0, 2, 1};
            break;
        case 4:
            transpose_order = {0, 2, 3, 1};
            break;
        default:
            break;
        }

        std::vector<size_t> transpose_shape;
        for (size_t i : transpose_order) {
            transpose_shape.push_back(input_shape[i]);
        }

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({input_shape, input_shape});
        configuration.insert(conf.begin(), conf.end());

        init_input_shapes(input_shapes);

        auto params = ngraph::builder::makeParams(net_type, {input_shape});
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto transpose_const = std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(params[0], transpose_const);

        auto mul_input_const = Constant::create(net_type, transpose_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(transpose_node, mul_input_const);

        auto add_input_const = Constant::create(net_type, transpose_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        ov::ResultVector results{std::make_shared<Result>(add_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveTransposeOutput : public PreprocessBaseTest {
protected:
    void SetUp() override {
        ov::element::Type net_type;
        std::vector<size_t> input_shape;
        std::map<std::string, std::string> conf;
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();

        std::tie(input_shape, net_type, targetDevice, conf) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({input_shape, input_shape});
        configuration.insert(conf.begin(), conf.end());
        init_input_shapes(input_shapes);

        std::vector<size_t> transpose_order;
        switch (input_shape.size())
        {
        case 2:
            transpose_order = {1, 0};
            break;
        case 3:
            transpose_order = {0, 2, 1};
            break;
        case 4:
            transpose_order = {0, 2, 3, 1};
            break;
        default:
            break;
        }

        std::vector<size_t> transpose_shape;
        for (size_t i : transpose_order) {
            transpose_shape.push_back(input_shape[i]);
        }

        auto params = ngraph::builder::makeParams(net_type, {input_shape});

        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto mul_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(params[0], mul_input_const);

        auto add_input_const = Constant::create(net_type, input_shape, CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        // std::vector<size_t> transpose_order = MakeIndexes(input_shape.size());
        auto transpose_const = std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(add_node, transpose_const);

        ov::ResultVector results{std::make_shared<Result>(transpose_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

TEST_P(RemoveTransposeInput, CompareWithRefs) {
    run();
}

TEST_P(RemoveTransposeOutput, CompareWithRefs) {
    run();
}

TEST_P(RemoveGatherInput, CompareWithRefs) {
    run();
}

TEST_P(RemoveGatherOutput, CompareWithRefs) {
    run();
}

std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}
};

const std::vector<std::vector<size_t>> input_shapes = {
    {2, 64}, {1, 2, 64}, {1, 2, 4, 16}
};

const ov::element::TypeVector input_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveGatherInput,
                         ::testing::Combine(::testing::Values( std::vector<size_t>{1, 128}),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveGatherInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveGatherOutput,
                         ::testing::Combine(::testing::Values(std::vector<size_t>{1, 128}),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveGatherInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveTransposeInput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveTransposeInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveTransposeOutput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveTransposeOutput::getTestCaseName);

}  // namespace LayerTestsDefinitions
