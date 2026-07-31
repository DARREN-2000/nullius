"""ONNX: write a real .onnx file, parse it back, and execute it.

The honest version of "ONNX Runtime inference". Three layers:

  1. `export_mlp` serialises a genuine ModelProto - protobuf wire format written
     by hand against the ONNX spec (ir_version, opset_import, graph, nodes,
     initializers, typed value_info). The output is a byte-for-byte valid
     .onnx file, not a pickle with a misleading extension.
  2. `load_graph` parses that file back from bytes with a generic protobuf
     decoder. Inference runs from the *parsed* graph, so if the serialisation
     were wrong, every prediction would be wrong and the tests would fail.
     That is the strongest guarantee available without the real runtime.
  3. `load_session` prefers `onnxruntime` when it is importable and falls back
     to the pure-Python interpreter otherwise. The sandbox has no network, so
     CI exercises the fallback; the ORT path is selected automatically wherever
     the wheel is installed, and `backend` on the result says which ran.

What is deliberately not claimed: this interpreter implements four operators
(MatMul, Add, Relu, Sigmoid), which is exactly what the exported graph uses. It
is not a general ONNX runtime and does not pretend to be one.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ONNX_IR_VERSION = 8
DEFAULT_OPSET = 13
TENSOR_FLOAT = 1
SUPPORTED_OPS = ("MatMul", "Add", "Relu", "Sigmoid")


class OnnxError(ValueError):
    pass


# ----------------------------------------------------------- protobuf writing
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _varint_field(field_number: int, value: int) -> bytes:
    return _key(field_number, 0) + _varint(value)


def _bytes_field(field_number: int, payload: bytes) -> bytes:
    return _key(field_number, 2) + _varint(len(payload)) + payload


def _string_field(field_number: int, value: str) -> bytes:
    return _bytes_field(field_number, value.encode("utf-8"))


def _tensor_proto(name: str, dims: Sequence[int], values: Sequence[float]) -> bytes:
    if len(values) != _product(dims):
        raise OnnxError(f"tensor {name}: {len(values)} values do not fill dims {list(dims)}")
    payload = b"".join(_varint_field(1, int(d)) for d in dims)
    payload += _varint_field(2, TENSOR_FLOAT)
    payload += _string_field(8, name)
    payload += _bytes_field(9, struct.pack(f"<{len(values)}f", *values))
    return payload


def _value_info(name: str, dims: Sequence[int]) -> bytes:
    shape = b"".join(_bytes_field(1, _varint_field(1, int(d))) for d in dims)
    tensor_type = _varint_field(1, TENSOR_FLOAT) + _bytes_field(2, shape)
    type_proto = _bytes_field(1, tensor_type)
    return _string_field(1, name) + _bytes_field(2, type_proto)


def _node_proto(op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str) -> bytes:
    payload = b"".join(_string_field(1, i) for i in inputs)
    payload += b"".join(_string_field(2, o) for o in outputs)
    payload += _string_field(3, name)
    payload += _string_field(4, op_type)
    return payload


def _product(dims: Iterable[int]) -> int:
    total = 1
    for d in dims:
        total *= int(d)
    return total


def export_mlp(
    path: str | Path,
    w1: Sequence[Sequence[float]],
    b1: Sequence[float],
    w2: Sequence[Sequence[float]],
    b2: Sequence[float],
    input_name: str = "features",
    output_name: str = "probability",
    graph_name: str = "nullius_lesion_mlp",
) -> Path:
    """Export a 2-layer MLP as features -> MatMul/Add/Relu/MatMul/Add/Sigmoid."""
    n_in, n_hidden = len(w1), len(w1[0])
    n_out = len(w2[0])
    if len(b1) != n_hidden or len(w2) != n_hidden or len(b2) != n_out:
        raise OnnxError("weight shapes are inconsistent")

    initializers = [
        _tensor_proto("W1", (n_in, n_hidden), [v for row in w1 for v in row]),
        _tensor_proto("B1", (n_hidden,), list(b1)),
        _tensor_proto("W2", (n_hidden, n_out), [v for row in w2 for v in row]),
        _tensor_proto("B2", (n_out,), list(b2)),
    ]
    nodes = [
        _node_proto("MatMul", [input_name, "W1"], ["hidden_raw"], "layer1_matmul"),
        _node_proto("Add", ["hidden_raw", "B1"], ["hidden_biased"], "layer1_bias"),
        _node_proto("Relu", ["hidden_biased"], ["hidden"], "layer1_relu"),
        _node_proto("MatMul", ["hidden", "W2"], ["logit_raw"], "layer2_matmul"),
        _node_proto("Add", ["logit_raw", "B2"], ["logit"], "layer2_bias"),
        _node_proto("Sigmoid", ["logit"], [output_name], "output_sigmoid"),
    ]
    graph = b"".join(_bytes_field(1, n) for n in nodes)
    graph += _string_field(2, graph_name)
    graph += b"".join(_bytes_field(5, t) for t in initializers)
    graph += _bytes_field(11, _value_info(input_name, (1, n_in)))
    graph += _bytes_field(12, _value_info(output_name, (1, n_out)))

    opset = _string_field(1, "") + _varint_field(2, DEFAULT_OPSET)
    model = _varint_field(1, ONNX_IR_VERSION)
    model += _string_field(2, "nullius")
    model += _bytes_field(7, graph)
    model += _bytes_field(8, opset)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(model)
    return target


# ----------------------------------------------------------- protobuf reading
def _read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if offset >= len(buf):
            raise OnnxError("truncated varint")
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def parse_message(buf: bytes) -> dict[int, list[Any]]:
    """Generic protobuf decoder: field number -> list of values."""
    fields: dict[int, list[Any]] = {}
    offset = 0
    while offset < len(buf):
        key, offset = _read_varint(buf, offset)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = _read_varint(buf, offset)
        elif wire_type == 2:
            length, offset = _read_varint(buf, offset)
            value = buf[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = buf[offset : offset + 4]
            offset += 4
        elif wire_type == 1:
            value = buf[offset : offset + 8]
            offset += 8
        else:
            raise OnnxError(f"unsupported wire type {wire_type}")
        fields.setdefault(field_number, []).append(value)
    return fields


@dataclass
class Tensor:
    dims: list[int]
    values: list[float]

    def rows(self) -> int:
        return self.dims[0] if len(self.dims) > 1 else 1

    def cols(self) -> int:
        return self.dims[-1] if self.dims else 1


@dataclass
class Node:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    name: str


@dataclass
class Graph:
    name: str
    nodes: list[Node] = field(default_factory=list)
    initializers: dict[str, Tensor] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    ir_version: int = 0
    opset: int = 0
    producer: str = ""


def _parse_tensor(blob: bytes) -> tuple[str, Tensor]:
    fields = parse_message(blob)
    dims = [int(d) for d in fields.get(1, [])]
    data_type = int(fields.get(2, [TENSOR_FLOAT])[0])
    if data_type != TENSOR_FLOAT:
        raise OnnxError(f"only float32 tensors are supported, got data_type={data_type}")
    name = fields.get(8, [b""])[0].decode("utf-8")
    raw = fields.get(9, [b""])[0]
    values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    return name, Tensor(dims=dims, values=values)


def _parse_value_info(blob: bytes) -> str:
    return parse_message(blob).get(1, [b""])[0].decode("utf-8")


def load_graph(path: str | Path) -> Graph:
    """Parse a .onnx file back into an executable graph."""
    model_fields = parse_message(Path(path).read_bytes())
    if 7 not in model_fields:
        raise OnnxError("model has no graph")
    graph_fields = parse_message(model_fields[7][0])

    graph = Graph(name=graph_fields.get(2, [b""])[0].decode("utf-8"))
    graph.ir_version = int(model_fields.get(1, [0])[0])
    graph.producer = model_fields.get(2, [b""])[0].decode("utf-8")
    if 8 in model_fields:
        graph.opset = int(parse_message(model_fields[8][0]).get(2, [0])[0])

    for blob in graph_fields.get(1, []):
        node_fields = parse_message(blob)
        graph.nodes.append(
            Node(
                op_type=node_fields.get(4, [b""])[0].decode("utf-8"),
                inputs=[b.decode("utf-8") for b in node_fields.get(1, [])],
                outputs=[b.decode("utf-8") for b in node_fields.get(2, [])],
                name=node_fields.get(3, [b""])[0].decode("utf-8"),
            )
        )
    for blob in graph_fields.get(5, []):
        name, tensor = _parse_tensor(blob)
        graph.initializers[name] = tensor
    graph.inputs = [_parse_value_info(b) for b in graph_fields.get(11, [])]
    graph.outputs = [_parse_value_info(b) for b in graph_fields.get(12, [])]

    unsupported = sorted({n.op_type for n in graph.nodes} - set(SUPPORTED_OPS))
    if unsupported:
        raise OnnxError(f"graph uses operators this interpreter does not implement: {unsupported}")
    return graph


# --------------------------------------------------------------- interpreters
def _matmul(a: Tensor, b: Tensor) -> Tensor:
    m, k = (a.dims + [1])[0], a.dims[-1]
    if len(a.dims) == 1:
        m, k = 1, a.dims[0]
    k2, n = (b.dims + [1])[0], b.dims[-1]
    if k != k2:
        raise OnnxError(f"MatMul shape mismatch: {a.dims} x {b.dims}")
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a.values[i * k + p] * b.values[p * n + j]
            out[i * n + j] = total
    return Tensor(dims=[m, n], values=out)


def _add(a: Tensor, b: Tensor) -> Tensor:
    if a.dims == b.dims:
        return Tensor(dims=list(a.dims), values=[x + y for x, y in zip(a.values, b.values)])
    width = b.dims[-1] if b.dims else 1
    if len(b.values) != width or a.dims[-1] != width:
        raise OnnxError(f"Add cannot broadcast {b.dims} onto {a.dims}")
    out = [a.values[i] + b.values[i % width] for i in range(len(a.values))]
    return Tensor(dims=list(a.dims), values=out)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class PythonSession:
    """Executes the parsed graph. No dependencies, deterministic, ~microseconds."""

    backend = "pure-python"

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.graph = load_graph(path)

    def run(self, features: Sequence[float]) -> list[float]:
        if not self.graph.inputs:
            raise OnnxError("graph declares no input")
        env: dict[str, Tensor] = dict(self.graph.initializers)
        env[self.graph.inputs[0]] = Tensor(dims=[1, len(features)], values=[float(v) for v in features])
        for node in self.graph.nodes:
            args = [env[name] for name in node.inputs]
            if node.op_type == "MatMul":
                result = _matmul(args[0], args[1])
            elif node.op_type == "Add":
                result = _add(args[0], args[1])
            elif node.op_type == "Relu":
                result = Tensor(dims=list(args[0].dims), values=[max(0.0, v) for v in args[0].values])
            elif node.op_type == "Sigmoid":
                result = Tensor(dims=list(args[0].dims), values=[_sigmoid(v) for v in args[0].values])
            else:  # pragma: no cover - load_graph rejects these first
                raise OnnxError(f"unimplemented operator {node.op_type}")
            env[node.outputs[0]] = result
        return list(env[self.graph.outputs[0]].values)


class OnnxRuntimeSession:  # pragma: no cover - exercised only where the wheel exists
    """The production path. Same file, same interface, vendor kernels."""

    backend = "onnxruntime"

    def __init__(self, path: str | Path) -> None:
        import numpy as np  # noqa: F401
        import onnxruntime as ort

        self.path = str(path)
        self.graph = load_graph(path)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(self.path, options, providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name

    def run(self, features: Sequence[float]) -> list[float]:
        import numpy as np

        batch = np.asarray([list(features)], dtype=np.float32)
        outputs = self.session.run(None, {self._input: batch})
        return [float(v) for v in np.asarray(outputs[0]).reshape(-1)]


def onnxruntime_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def load_session(path: str | Path, prefer_onnxruntime: bool = True) -> PythonSession | OnnxRuntimeSession:
    """Return the best available session for this .onnx file."""
    if prefer_onnxruntime and onnxruntime_available():
        try:  # pragma: no cover - environment dependent
            return OnnxRuntimeSession(path)
        except Exception:
            pass
    return PythonSession(path)
