"""
Cast Int64 tensors to Int32 throughout an ONNX graph so TensorRT 10's
parser (which forbids mixed-dtype Concat inputs) can build the engine.

Usage:
    python fix_int64_concat.py input.onnx output.onnx
"""
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs


def cast_int64_to_int32(path_in: str, path_out: str) -> None:
    # Run ONNX shape inference so intermediate tensor dtypes are populated.
    model = onnx.load(path_in)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:
        print(f"shape_inference warning: {e}")

    graph = gs.import_onnx(model)

    # TensorRT 10's parser rejects mixed int64/int32 on arithmetic ops
    # (Concat, Mul, Add, ...). Many of those ops produce values that later
    # become shape inputs for Reshape/Expand/Slice/Tile, which *require* int64.
    # So unify *upward* to int64 (promote int32 -> int64) rather than down.
    arithmetic_ops = {"Concat", "Add", "Sub", "Mul", "Div", "Mod", "Pow",
                      "Min", "Max", "Equal", "Less", "LessOrEqual",
                      "Greater", "GreaterOrEqual", "Where"}
    # Promote int32 initializer constants used by arithmetic ops to int64
    # before walking the graph, so they line up with int64 siblings.
    int32_initializers_used_by_arith: set = set()
    for node in graph.nodes:
        if node.op in arithmetic_ops:
            for inp in node.inputs:
                if isinstance(inp, gs.Constant) and inp.values.dtype == np.int32:
                    int32_initializers_used_by_arith.add(inp.name)
    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Constant) \
           and tensor.values.dtype == np.int32 \
           and tensor.name in int32_initializers_used_by_arith:
            tensor.values = tensor.values.astype(np.int64)

    # Insert Cast(int64) wherever an int32 dynamic tensor still flows into an
    # arithmetic op (Shape inference fills in dtypes for value_info entries).
    cast_count = 0
    for node in list(graph.nodes):
        if node.op not in arithmetic_ops:
            continue
        for idx, inp in enumerate(list(node.inputs)):
            if getattr(inp, "dtype", None) == np.int32:
                cast_out = gs.Variable(name=f"{inp.name}_to_i64_{cast_count}",
                                       dtype=np.int64)
                cast_node = gs.Node(
                    op="Cast",
                    name=f"Cast_to_i64_{cast_count}",
                    attrs={"to": int(onnx.TensorProto.INT64)},
                    inputs=[inp],
                    outputs=[cast_out],
                )
                graph.nodes.append(cast_node)
                node.inputs[idx] = cast_out
                cast_count += 1

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), path_out)
    print(f"Saved {path_out} (added {cast_count} Cast nodes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: fix_int64_concat.py <in.onnx> <out.onnx>")
    cast_int64_to_int32(sys.argv[1], sys.argv[2])
