# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.tools.mo.back.ie_ir_ver_2.emitter import append_ir_info
from openvino.tools.mo.back.preprocessing import apply_preprocessing
from openvino.tools.mo.utils.cli_parser import get_meta_info, parse_transform

from openvino.runtime import Model         # pylint: disable=no-name-in-module,import-error


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=ngraph_function, argv=argv)

    # Apply transformations
    from openvino.tools.mo.back.offline_transformations import apply_user_transformations, apply_moc_transformations
    apply_user_transformations(ngraph_function, parse_transform(argv.transform))
    apply_moc_transformations(ngraph_function)

    if argv.compress_fp16:
        from openvino.tools.mo.back.offline_transformations import compress_model
        compress_model(ngraph_function)

    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))

    from openvino.offline_transformations_pybind import serialize # pylint: disable=import-error,no-name-in-module
    serialize(ngraph_function, (orig_model_name + ".xml").encode('utf-8'), (orig_model_name + ".bin").encode('utf-8'))

    del argv.feManager

    # add meta information to IR
    append_ir_info(file=orig_model_name,
                   meta_info=get_meta_info(argv),
                   mean_data=None,
                   input_names=None)

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}.xml'.format(orig_model_name))
    print('[ SUCCESS ] BIN file: {}.bin'.format(orig_model_name))
    return 0
