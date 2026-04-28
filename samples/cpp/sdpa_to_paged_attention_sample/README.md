# SDPA To PagedAttention C++ Sample

This sample demonstrates how to read a model, apply the `SDPAToPagedAttention` transformation, and compile the transformed model for one or more devices.

The sample is intended for stateful decoder models that contain `ScaledDotProductAttention` and are compatible with the `SDPAToPagedAttention` pass.

## Requirements

| Options | Values |
| --- | --- |
| Supported model formats | OpenVINO IR, ONNX |
| Supported devices | CPU, GPU |
| Other language realization | None |

The following C++ API is used in the application:

| Feature | API | Description |
| --- | --- | --- |
| Model reading | `ov::Core::read_model` | Read an input model |
| Pass manager | `ov::pass::Manager` | Run graph transformations |
| SDPA conversion | `ov::pass::SDPAToPagedAttention` | Convert SDPA-based stateful model to paged-attention form |
| Compilation | `ov::Core::compile_model` | Compile the transformed model for a target device |

## Running

```sh
sdpa_to_paged_attention_sample <path_to_model> [CPU|GPU|CPU,GPU]
```

Examples:

```sh
sdpa_to_paged_attention_sample model.xml CPU
sdpa_to_paged_attention_sample model.xml GPU
sdpa_to_paged_attention_sample model.xml CPU,GPU
```

If the device argument is omitted, the sample uses `CPU`.

## Notes

- The input model must be stateful.
- The input model must contain `ScaledDotProductAttention`.
- For text generation models, export a `text-generation-with-past` style model rather than a stateless `text-generation` model.
- **Important**: The transformed model contains custom extension operations (`PagedCausalConv1D`, `PagedGatedDeltaNet`) that require specialized plugin support.
  - Compilation may fail if the target device does not support these extension operations.
  - The transformed model IR is always saved as `sdpa_to_paged_attention_sample.xml` and `sdpa_to_paged_attention_sample.bin`, allowing it to be used on compatible backends.
  - This transformation is designed for deployment on hardware accelerators or specialized inference runtimes that support these optimized operations.