# Agent Skill: Add New Operation to OpenVINO Frontend

## 🎯 Goal
Provide an agent capability to implement and validate a new operation
in the OpenVINO Frontend (FE) pipeline, including:

- Operation translation
- Model conversion validation
- Unit and functional tests

---

## 🧩 Scope

The skill now **enables** a new operation in OpenVINO PyTorch FE by:

1. Creating translator source in `src/frontends/pytorch/src/op/<op>.cpp`
2. Registering translator in `src/frontends/pytorch/src/op_table.cpp`
3. Generating smoke tests (`tests/frontend/test_<op>.cpp`)
4. Generating layer test (`tests/layer_tests/pytorch_tests/test_<op>_op.py`)
5. Building FE target and validating conversion with a generated TorchScript model

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
def add_frontend_operation(
    op_name: str,
    framework: str = "pytorch",
    build: bool = True,
    validate: bool = True,
) -> dict:
    """
    Automates full lifecycle of adding a frontend operation.

    Parameters
    ----------
    op_name : Operation name
    framework : FE name (currently implemented for "pytorch")
    build : Build FE target after modifications
    validate : Generate model and run FE conversion check

    Returns
    -------
    dict with build, test and validation results
    """

Usage:

```python
from scripts.ov_agent_skill import add_frontend_operation

result = add_frontend_operation(op_name="torch.erfinv", framework="pytorch")
print("\n".join(result["actions"]))
```

CLI usage:

```bash
python src/frontends/skills/new_op/scripts/ov_agent_skill.py \
    --op-name torch.erfinv \
    --framework pytorch
```

## ⚠️ Notes

- The generated translator is a safe placeholder (`make_framework_node`) and must be replaced with real translation logic for full support.
- Registration is idempotent: re-running the skill does not duplicate entries.
- `OPENVINO_ROOT` and `OPENVINO_BUILD_DIR` environment variables can override default paths.
