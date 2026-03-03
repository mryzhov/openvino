import torch
import torch.nn as nn


def _canonical_op_name(op_name: str) -> str:
    name = op_name.strip()
    if name.startswith("torch."):
        name = name.split(".", maxsplit=1)[1]
    if name.startswith("aten::"):
        name = name.split("::", maxsplit=1)[1]
    if name.endswith("_"):
        name = name[:-1]
    return name


class CustomOpModule(nn.Module):
    def __init__(self, op_name: str):
        super().__init__()
        self.op_name = _canonical_op_name(op_name)

    def forward(self, x):
        fn = getattr(torch, self.op_name, None)
        if fn is None:
            return x
        return fn(x)


def create_pt_model(op_name: str, filename: str):
    model = CustomOpModule(op_name)
    canonical_name = _canonical_op_name(op_name)
    if canonical_name in {"erfinv", "erf", "erfc"}:
        example = torch.rand(1, 3) * 1.8 - 0.9
    else:
        example = torch.randn(1, 3)

    traced = torch.jit.trace(model, example)
    traced.save(filename)
    print(f"TorchScript model created: {filename}")
