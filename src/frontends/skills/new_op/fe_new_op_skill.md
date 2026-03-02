# Agent Skill: Add New Operation to OpenVINO Frontend

## 🎯 Goal
Provide an agent capability to implement and validate a new operation
in the OpenVINO Frontend (FE) pipeline, including:

- Operation translation
- Model conversion validation
- Unit and functional tests

---

## 🧩 Scope

The skill automates or assists with:

1. Adding FE translator for a new op
2. Registering op in conversion pipeline
3. Building project
4. Running FE and conversion tests

---

## 🏗️ Architecture

Agent → Skill → FE Translator → OpenVINO Core Graph

---

## 📂 Expected Repository Layout
openvino/
├─ src/frontends/<framework>/
│ ├─ src/
│ │ ├─ op/
│ │ │ └─ new_op.cpp
│ │ ├─ translator.cpp
│ │ └─ CMakeLists.txt
│ └─ tests/
│   └─ test_new_op.cpp
├─ tests/
│ └─ layer_tests/
│   └─ <framework>/
│     └─ test_new_op_conversion.py

---

## ⚙️ Skill Interface (Python)

```python
def add_fe_op(
    op_name: str,
    framework: str,
    reference_impl: str,
    test_model_path: str
) -> str:
    """
    Adds and validates a new FE operation.

    Parameters
    ----------
    op_name : Operation name
    framework : FE name (onnx, tensorflow, tflite, paddle, pytorch, jax)
    reference_impl : Path or snippet with reference logic
    test_model_path : Model containing the op

    Returns
    -------
    Status report
    """
