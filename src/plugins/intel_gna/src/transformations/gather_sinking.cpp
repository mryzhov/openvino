// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking.hpp"

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/gather_sinking_unary.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace ov::intel_gna::pass;

GatherSinkingGeneralForward::GatherSinkingGeneralForward() {
    MATCHER_SCOPE(GatherSinkingGeneralForward);
    add_matcher<GatherSinkingUnaryForward>();
}

GatherSinkingGeneralBackward::GatherSinkingGeneralBackward() {
    MATCHER_SCOPE(GatherSinkingGeneralBackward);
    add_matcher<GatherSinkingUnaryBackward>();
}

bool GatherSinkingGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(GatherSinkingGeneral);
    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<GatherSinkingGeneralForward>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<GatherSinkingGeneralBackward>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    return false;
}
