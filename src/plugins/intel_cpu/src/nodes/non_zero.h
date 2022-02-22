// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <extension_utils.h>

namespace ov {
namespace intel_cpu {

class MKLDNNNonZeroNode : public MKLDNNNode {
public:
  MKLDNNNonZeroNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool needShapeInfer() const override {return false;};
    bool needPrepareParams() const override {return false;};
    void executeDynamicImpl(mkldnn::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override { return true; }

private:
    std::string errorPrefix;
    template <typename inputType>
    void executeSpecified();
    template<typename T>
    struct NonZeroExecute;
    template <typename T>
    size_t getNonZeroElementsCount(const T* arg, const Shape& arg_shape);
};

}   // namespace intel_cpu
}   // namespace ov
