# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_sample(NAME sdpa_to_paged_attention_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ie_samples_utils)