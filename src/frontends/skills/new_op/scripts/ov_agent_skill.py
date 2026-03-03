"""Skill runner that enables a new operation in OpenVINO frontend."""

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from generate_cpp_test import create_cpp_test
from generate_pt_model import create_pt_model

BUILD_DIR = Path(
    os.environ.get("OPENVINO_BUILD_DIR", "/home/rmikhail/src/openvino/build")
)
OPENVINO_ROOT = Path(
    os.environ.get("OPENVINO_ROOT", "/home/rmikhail/src/openvino")
)


def _canonical_op_name(op_name: str) -> str:
    name = op_name.strip()
    if name.startswith("torch."):
        name = name.split(".", maxsplit=1)[1]
    if name.startswith("aten::"):
        name = name.split("::", maxsplit=1)[1]
    if name.endswith("_"):
        name = name[:-1]
    return name


def _insert_after_once(text: str, anchor: str, insertion: str) -> str:
    if insertion in text:
        return text
    if anchor not in text:
        raise ValueError(f"Anchor not found: {anchor}")
    return text.replace(anchor, f"{anchor}\n{insertion}", 1)


def _create_pytorch_translator(op_name: str) -> str:
    translator_file = (
        OPENVINO_ROOT / f"src/frontends/pytorch/src/op/{op_name}.cpp"
    )
    translator_file.parent.mkdir(parents=True, exist_ok=True)
    if translator_file.exists():
        return f"ℹ️ translator exists: {translator_file}"

    translator_file.write_text(
        f"""// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {{
namespace frontend {{
namespace pytorch {{
namespace op {{

OutputVector translate_{op_name}(const NodeContext& context) {{
    return make_framework_node(
        context,
        "translate_{op_name} is not implemented"
    );
}}

}}  // namespace op
}}  // namespace pytorch
}}  // namespace frontend
}}  // namespace ov
"""
    )
    return f"✅ translator created: {translator_file}"


def _register_pytorch_op(op_name: str) -> str:
    op_table = OPENVINO_ROOT / "src/frontends/pytorch/src/op_table.cpp"
    text = op_table.read_text()

    text = _insert_after_once(
        text,
        "OP_CONVERTER(translate_erfc);",
        f"OP_CONVERTER(translate_{op_name});",
    )
    text = _insert_after_once(
        text,
        '        {"aten::erfc", op::translate_erfc},',
        f'        {{"aten::{op_name}", op::translate_{op_name}}},',
    )
    text = _insert_after_once(
        text,
        '        {"aten.erfc.default", op::translate_erfc},',
        f'        {{"aten.{op_name}.default", op::translate_{op_name}}},',
    )

    op_table.write_text(text)
    return f"✅ registration updated: {op_table}"


def _create_pytorch_layer_test(op_name: str) -> str:
    test_file = (
        OPENVINO_ROOT
        / f"tests/layer_tests/pytorch_tests/test_{op_name}_op.py"
    )
    if test_file.exists():
        return f"ℹ️ layer test exists: {test_file}"

    test_file.write_text(
        f"""# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class {op_name.capitalize()}Op(torch.nn.Module):
    def forward(self, x):
        fn = getattr(torch, "{op_name}", None)
        if fn is None:
            raise RuntimeError("torch.{op_name} is not available")
        return fn(x)


class Test{op_name.capitalize()}Op(PytorchLayerTest):
    def _prepare_input(self):
        x = self.random.torch_rand(2, 10)
        return (x.to(torch.float32).numpy(),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_{op_name}(self, ie_device, precision, ir_version):
        if not hasattr(torch, "{op_name}"):
            pytest.skip(
                "torch.{op_name} is not available in this torch version"
            )
        self._test(
            {op_name.capitalize()}Op(),
            "aten::{op_name}",
            ie_device,
            precision,
            ir_version,
        )
"""
    )
    return f"✅ layer test created: {test_file}"


def _build_frontends() -> str:
    for target in ["frontends", "ov_frontends"]:
        try:
            subprocess.check_call(
                ["cmake", "--build", str(BUILD_DIR), "--target", target]
            )
            return f"✅ build successful ({target})"
        except subprocess.CalledProcessError:
            continue
    return "❌ build failed"


def _check_pytorch_conversion(op_name: str) -> str:
    model_file = BUILD_DIR / f"test_model_{op_name}.pt"
    create_pt_model(op_name, str(model_file))

    try:
        import torch
        from openvino.frontend import FrontEndManager
        from openvino.frontend.pytorch.ts_decoder import (
            TorchScriptPythonDecoder,
        )

        fem = FrontEndManager()
        fe = fem.load_by_framework("pytorch")
        ts_model = torch.jit.load(str(model_file))
        decoder = TorchScriptPythonDecoder(ts_model)
        input_model = fe.load(decoder)
        converted_model = fe.convert(input_model)
        return (
            "✅ conversion successful, "
            f"ops: {len(converted_model.get_ops())}"
        )
    except ModuleNotFoundError as error:
        return f"⚠️ conversion skipped (missing module): {error}"
    except (RuntimeError, ImportError, FileNotFoundError) as error:
        return f"❌ conversion failed: {error}"


def add_frontend_operation(
    op_name: str,
    framework: str = "pytorch",
    build: bool = True,
    validate: bool = True,
) -> Dict[str, List[str]]:
    """Enable a new operation in a frontend.

    For now, this skill supports only PyTorch FE because it requires
    framework-specific op table registration logic.
    """
    framework = framework.lower()
    canonical_op_name = _canonical_op_name(op_name)
    report: Dict[str, List[str]] = {
        "op": [canonical_op_name],
        "framework": [framework],
        "actions": [],
    }

    if not re.fullmatch(r"[a-zA-Z0-9_]+", canonical_op_name):
        report["actions"].append("❌ invalid op name")
        return report

    if framework != "pytorch":
        report["actions"].append(
            "❌ only pytorch framework is supported "
            "by this skill implementation"
        )
        return report

    report["actions"].append(_create_pytorch_translator(canonical_op_name))
    report["actions"].append(_register_pytorch_op(canonical_op_name))
    report["actions"].append(
        create_cpp_test(
            canonical_op_name,
            OPENVINO_ROOT / "tests/frontend",
        )
    )
    report["actions"].append(_create_pytorch_layer_test(canonical_op_name))

    if build:
        report["actions"].append(_build_frontends())

    if validate:
        report["actions"].append(_check_pytorch_conversion(canonical_op_name))

    return report


def _format_report(data: Dict[str, List[str]]) -> str:
    lines = [f"op={data['op'][0]}", f"framework={data['framework'][0]}"]
    lines.extend(data["actions"])
    return "\n".join(lines)


def add_new_fe_op(op_name: str, framework: str = "pytorch") -> str:
    """Backward compatible wrapper for previous skill API."""
    return _format_report(add_frontend_operation(op_name, framework=framework))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enable new OpenVINO frontend operation"
    )
    parser.add_argument(
        "--op-name",
        required=True,
        help="Operation name, e.g. torch.erfinv",
    )
    parser.add_argument(
        "--framework",
        default="pytorch",
        help="Frontend framework",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip frontend build",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model conversion validation",
    )
    args = parser.parse_args()

    result = add_frontend_operation(
        op_name=args.op_name,
        framework=args.framework,
        build=not args.no_build,
        validate=not args.no_validate,
    )
    print(_format_report(result))
