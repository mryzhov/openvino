// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <gtest/gtest.h>
#include <legacy/ie_layers.h>
#include <legacy/graph_tools.hpp>
#include <legacy/details/ie_cnn_network_tools.h>
#include "ngraph_functions/builders.hpp"
#include "memory/gna_memory.hpp"
#include "gna_plugin.hpp"
#include "gna_fused_iterator.hpp"
#include "gna_data_types.hpp"


using namespace InferenceEngine;
using namespace GNAPluginNS::memory;

class GNAMemoryCompactTest : public ::testing::Test {
 protected:
    GNAMemory<std::allocator<uint8_t>> mem;
    bool isCompact = true;

    void SetUp() override  {
    }
};

TEST_F(GNAMemoryCompactTest, canOptimizeReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    mem.reserve_ptr(layer1, pFuture1, 3 * sizeof(float));
    mem.reserve_ptr(layer2, pFuture2, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 3 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 3 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizePushValue) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    mem.push_value(layer1, pFuture1, 1.f, 2);
    mem.push_value(layer2, pFuture2, 2.f, 3);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 5 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 5 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizePushValueAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.push_value(layer1, pFuture1, 3.f, 2);
    mem.bind_ptr(layer2, pFuture2, pFuture1, 0, 2);
    mem.reserve_ptr(layer3, pFuture3, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 2 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 2 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizeTwoPushValueAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    CNNLayerPtr layer4 = std::make_shared<CNNLayer>(LayerParams("layer4", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    layer4->userValue.v_int = 4;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.push_value(layer1, pFuture1, 1.f, 2);
    mem.push_value(layer2, pFuture2, 2.f, 3);
    mem.reserve_ptr(layer3, pFuture3, 5 * sizeof(float));
    mem.bind_ptr(layer2, pFuture2, pFuture1, 0, 2);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 5 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 5 * sizeof(float));
}


TEST_F(GNAMemoryCompactTest, canOptimizePushPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float input[]  = {1, 2, 3};
    size_t input_size = sizeof(input);

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.push_ptr(layer1, pFuture1, input, input_size);
    mem.reserve_ptr(layer2, pFuture2, input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), input_size);
    ASSERT_EQ(mem.getTotalBytes(), input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizePushLocalPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        input_size = input.size() * sizeof(float);
        mem.push_local_ptr(layer1, pFuture1, &*input.begin(), input_size);
    }

    mem.reserve_ptr(layer2, pFuture2, input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), input_size);
    ASSERT_EQ(mem.getTotalBytes(), input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizePushInitilizerPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    size_t input_size;
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        input_size = input.size() * sizeof(float);
        mem.push_initializer(layer1, pFuture1, input_size, [=](void* data, size_t size){
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    mem.reserve_ptr(layer2, pFuture2, 2 * input_size);
    mem.bind_ptr(layer3, pFuture3, pFuture2, 0, input_size);

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 2 * input_size);
    ASSERT_EQ(mem.getTotalBytes(), 2 * input_size);
}

TEST_F(GNAMemoryCompactTest, canOptimizeBindInitilizerPtrAndReservePtr) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    CNNLayerPtr layer4 = std::make_shared<CNNLayer>(LayerParams("layer4", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    layer4->userValue.v_int = 4;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);
    float* pFuture4 = reinterpret_cast<float*>(&pFuture4);

    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        mem.bind_initializer(layer2, pFuture1, [=](void* data, size_t size){
            ie_memcpy(data, size, &input[0], input.size());
        });
    }

    mem.reserve_ptr(layer1, pFuture1, 4 * sizeof(float));
    mem.reserve_ptr(layer3, pFuture3, 2 * sizeof(float));
    mem.bind_ptr(layer4, pFuture4, pFuture3, 0, 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 4 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 4 * sizeof(float));
}

TEST_F(GNAMemoryCompactTest, canOptimizeReservePtrWithOffset) {
    IE_SUPPRESS_DEPRECATED_START
    CNNLayerPtr layer1 = std::make_shared<CNNLayer>(LayerParams("layer1", "test", Precision::FP32));
    CNNLayerPtr layer2 = std::make_shared<CNNLayer>(LayerParams("layer2", "test", Precision::FP32));
    CNNLayerPtr layer3 = std::make_shared<CNNLayer>(LayerParams("layer3", "test", Precision::FP32));
    layer1->userValue.v_int = 1;
    layer2->userValue.v_int = 2;
    layer3->userValue.v_int = 3;
    IE_SUPPRESS_DEPRECATED_END

    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.reserve_ptr(layer1, pFuture1, 2 * sizeof(float));
    mem.reserve_ptr(layer2, pFuture2, 2 * sizeof(float));
    mem.bind_ptr(layer3, pFuture3, pFuture2, 2 * sizeof(float), 2 * sizeof(float));

    mem.commit(isCompact);
    ASSERT_EQ(mem.getRWBytes(), 4 * sizeof(float));
    ASSERT_EQ(mem.getTotalBytes(), 4 * sizeof(float));
}


class GNAMemoryTest : public GNAPluginNS::memory::GNAMemory<GNAPluginNS::memory::PolymorphAllocator<uint8_t>> {
using GNAMemory::GNAMemory;
public:
    // GNAMemoryTest(const std::allocator<uint8_t> &a) :
    //     GNAPluginNS::memory::GNAMemory<std::allocator<uint8_t>>(a) {}
    void Test() {
        for (auto &re : _future_heap) {
            if (re._region != REGION_RW) continue;
            std::cout << "life_time: " << re._life_limits.first << ":"
                      << re._life_limits.second << ", " << std::endl;
        }
    }
    // std::vector<MemRequest> _future_heap;
    // std::vector<MemRequest> & futureHeap() override {
    //     return _future_heap;
    // }
};

class GNACompactTest : public ::testing::Test {
 protected:
    void SetUp() override {}
};

// using allocator_type = GNAPluginNS::memory::PolymorphAllocator<uint8_t>;
// using gna_memory_type = GNAMemoryTest;

class GNAPluginTested : public GNAPluginNS::GNAPlugin {
// using GNAPlugin::GNAPlugin;
// using allocator_type = std::allocator<uint8_t>;
// using gna_memory_type = GNAMemoryTest;

protected:
    // GNAGraphCompilerTest graphCompiler;
    // std::shared_ptr<GNAMemoryTest> gnamem;
    // GNAPluginTested() : GNAPluginNS::GNAPlugin() {}
    // std::shared_ptr<GNAMemoryTest> gnamem;
    // void InitGNADevice() {
    //     gnamem.reset(new GNAMemoryTest(make_polymorph<std::allocator<uint8_t>>()));
    //     graphCompiler.setGNAMemoryPtr(gnamem);
    // }

public:
    // GNAGraphCompilerTest graphCompiler;
    std::shared_ptr<GNAMemoryTest> gnamem_t;
    GNAPluginTested() : GNAPluginNS::GNAPlugin() {
        // auto mem_alloc = std::make_shared<GNAMemoryTest>(make_polymorph<std::allocator<uint8_t>>());
        // gnamem.reset(new GNAMemoryTest(std::allocator<uint8_t>()));
        // gnamem.reset(new GNAMemoryTest(make_polymorph<std::allocator<uint8_t>>()));
        gnamem_t = std::make_shared<GNAMemoryTest>(make_polymorph<std::allocator<uint8_t>>());
        gnamem = gnamem_t;
        // gnamem.reset(new gna_memory_type(memory::make_polymorph<std::allocator<uint8_t>>()));
    }
    void Test() {
        for (auto &dnn_comp : graphCompiler.dnnComponents.components) {
            std::cout << dnn_comp.name << std::endl;
            ASSERT_EQ(dnn_comp.dnnComponent.ptr_inputs, dnn_comp.dnnComponent.ptr_inputs);
            std::cout << dnn_comp.dnnComponent.ptr_inputs << std::endl;
            std::cout << dnn_comp.dnnComponent.ptr_outputs << std::endl;
            std::cout << dnn_comp.dnnComponent.op.affine.ptr_biases << std::endl;
            std::cout << dnn_comp.dnnComponent.op.affine.ptr_weights << std::endl;
            std::cout << dnn_comp.dnnComponent.op.conv1D.ptr_biases << std::endl;
            std::cout << dnn_comp.dnnComponent.op.conv1D.ptr_filters << std::endl;
            std::cout << dnn_comp.dnnComponent.op.conv2D.ptr_biases << std::endl;
            std::cout << dnn_comp.dnnComponent.op.conv2D.ptr_filters << std::endl;
            std::cout << dnn_comp.dnnComponent.op.pwl.ptr_segments << std::endl;
        }
        gnamem_t->Test();
        // graphCompiler.Test();
        // for (auto &re : graphCompiler) {
        //     if (re._region != REGION_RW) continue;
        //     std::cout << "life_time: " << re._life_limits.first << ":"
        //               << re._life_limits.second << ", " << std::endl;
        // }
    }
};

TEST_F(GNACompactTest, orderingFusedLayers) {
    // ov::SupportedOpsMap plugin_cfg({{"GNA_DEVICE_MODE", "GNA_SW_FP32"}});
    auto plugin = GNAPluginTested();
    // std::shared_ptr<GNAMemoryTest> gnamem = std::make_shared<GNAMemoryTest>(std::allocator<uint8_t>());
    // plugin.graphCompiler.setGNAMemoryPtr(gnamem);

    ov::Shape input_shape =  { 1, 8, 20, 16 };
    ov::Strides strides = { 1, 1 };
    ov::Strides dilations = { 1, 1 };
    ov::CoordinateDiff pad_begin(0, 0), pad_end(0, 0);
    auto weights = ngraph::builder::makeConstant<float>(ov::element::f32, { 8, 8, 1, 1 }, { 1.f });

    auto input = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, input_shape);
    auto conv = std::make_shared<ngraph::opset8::Convolution>(input, weights, strides, pad_begin, pad_end, dilations);
    auto activation = ngraph::builder::makeActivation(conv, ov::element::f32, ngraph::helpers::ActivationTypes::Sigmoid);
    auto maxpool = ngraph::builder::makePooling(activation, {1, 1}, {0, 0}, {0, 0}, {1, 1}, ngraph::op::RoundingType::FLOOR,
                                                    ngraph::op::PadType::VALID, false, ngraph::helpers::PoolingTypes::MAX);

    auto result = std::make_shared<ngraph::opset8::Result>(maxpool);
    auto function = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({input}), "convolution");
    //
    InferenceEngine::CNNNetwork cnn_network(function);
    plugin.LoadNetwork(cnn_network);
    plugin.Test();

    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(cnn_network);
    std::cout << cnn_network.layerCount();
    for (auto layer : layers) {
        std::cout << layer->name << " : " << layer->userValue.v_int << std::endl;
    }
}