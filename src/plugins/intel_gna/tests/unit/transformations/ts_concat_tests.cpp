// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

#include <ngraph/function.hpp>
#include <openvino/opsets/opset10.hpp>
#include <ops/gna_convolution.hpp>
#include <ops/gna_max_pool.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "transformations/ts_concat.hpp"

using namespace ov;
using namespace ov::opset10;

void ShiftLeft(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.begin(), vec.begin() + k, buffer.begin());

    for (int i = k; i < vec.size(); ++i) {
        vec[i - k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.end() - k);
}

void ShiftRight(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.end() - k, vec.end(), buffer.begin());

    for (int i = vec.size() - 1 - k; i >= 0; --i) {
        vec[i + k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.begin());
}

std::vector<size_t> GatherForward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    ShiftLeft(vec, 2);
    return vec;
}

std::vector<size_t> GatherBackward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value); // Not the same as in binary tests
    ShiftRight(vec, 2);
    return vec;
}

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;
using Output = ov::Output<ov::Node>;

template <typename CreateIndicesF>
std::shared_ptr<Gather> MakeGather(NodePtr input_node, CreateIndicesF create_indices_func, size_t axis) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    const std::vector<size_t> indexes = create_indices_func(input_shape[axis], 0);

    auto gather_indexes_node = Constant::create(ngraph::element::i64, ov::Shape{indexes.size()}, indexes);

    auto gather_axis_node = Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});

    return std::make_shared<Gather>(input_node->output(0), gather_indexes_node, gather_axis_node);
}

std::vector<size_t> TSConcat_Forward_indexes(size_t size, size_t initial_value) {
    return std::vector<size_t>{0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
}

TEST(TSConcat, Forward) {
    std::shared_ptr<Model> function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});

        auto transpose_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose = std::make_shared<Transpose>(input_params1, transpose_const);

        auto concat = std::make_shared<Concat>(NodeVector{transpose, input_params2}, 0);

        const auto result = std::make_shared<Result>(concat);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::TSConcatForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});

        auto reshape_const1 = Constant::create(ngraph::element::i64, ov::Shape{2}, ov::Shape{1,16});
        auto reshape1 = std::make_shared<Reshape>(input_params1, reshape_const1, false);

        auto reshape_const2 = Constant::create(ngraph::element::i64, ov::Shape{2}, ov::Shape{1,16});
        auto reshape2 = std::make_shared<Reshape>(input_params2, reshape_const2, false);

        auto concat = std::make_shared<Concat>(NodeVector{reshape1, reshape2}, 1);

        auto gather = MakeGather(concat, TSConcat_Forward_indexes, /* axis */ 1);

        auto reshape_const3 = Constant::create(ngraph::element::i64, ov::Shape{4}, ov::Shape{4,2,2,2});
        auto reshape3 = std::make_shared<Reshape>(gather, reshape_const3, false);

        const auto result = std::make_shared<Result>(reshape3);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
