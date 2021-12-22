// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_layer.hpp"

#include <legacy/ie_layers.h>
#include <legacy/layer_transform.hpp>
#include "frontend/quantized_layer_params.hpp"
#include "gna_layer_info.hpp"
#include "backend/am_intel_dnn.hpp"
#include "gna_fake_quantize_layer.hpp"
#include "gna_graph_tools.hpp"

namespace GNAPluginNS {

bool Is32BitState(InferenceEngine::CNNLayer* layer) {
    if (!LayerInfo(layer).isMemory()) return false;

    auto quant_params = InferenceEngine::getInjectedData<QuantizedLayerParams>(*layer);
    size_t levels;
    bool isAssign = CNNNetHasPrevLayer(layer);
    if (quant_params == nullptr) {
        auto fq = isAssign ? CNNNetPrevLayer(layer) : getInputTo(layer->outData[0]).begin()->second;
        if (!LayerInfo(fq).isFakeQuantize()) return false;
        GNAFakeQuantizeLayer fqLayer(fq);
        levels = fqLayer.getLevels();
    } else {
        auto quant = isAssign ? quant_params->_src_quant : quant_params->_dst_quant;
        if (!quant.IsStatsSet()) return false;
        levels = quant.GetLevels();
    }

    return levels == static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
}

}  // namespace GNAPluginNS