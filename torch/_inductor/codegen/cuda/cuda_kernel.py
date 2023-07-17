from typing import List, Optional

import sympy

from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP
from ...ir import IRNode, TemplateBuffer, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product


cexpr = CppPrinter().doprint

def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class CUDAKernel(Kernel):
    overrides = OpOverrides
    pass


class CUDATemplateKernel(CUDAKernel):
    def __init__(
        self,
        kernel_name,
    ):
        super().__init__()
        self.kernel_name = kernel_name
        self.named_nodes = {}


    def arg_name(self, node: IRNode) -> Optional[str]:
        if node is None:
            return None
        return {
            **self.args.input_buffers,
            **self.args.output_buffers
        }.get(node.get_name(), None)


    def check_not_null(self, node: IRNode) -> str:
        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            """
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null!");
                }}
              }}
            }}
            """.format(name_str=name_str, size_str=size_str))
        return res.getvalue()


    def def_kernel(self, inputs: List[IRNode], outputs: List[IRNode], names_str: str = "") -> str:
        """
        Hook called from template code to generate function def and
        needed args.
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) > len(names):
            raise RuntimeError(f"{len(inputs) + len(outputs)=} > {len(names)=}, {inputs=}, {outputs=}, {names=}")

        for name, node in zip(names[:len(inputs)], inputs):
            if node is not None:
                print(f"before: {self.args.input_buffers=}")
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name
                print(f"after: {self.args.input_buffers=}")

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                print(f"before: {self.args.output_buffers=}")
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name
                print(f"after: {self.args.output_buffers=}")

        arg_defs, *_ = self.args.cpp_argdefs()
        arg_defs = arg_defs + names[len(inputs) + len(outputs):]
        return f"void {self.kernel_name}({', '.join(arg_defs)})"


    def dtype(self, node: IRNode) -> str:
        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)


    def offset(self, node: IRNode) -> str:
        if node is None:
            return "0"
        return str(node.get_layout().offset)


    def ptr(self, node: IRNode) -> str:
        if node is None:
            return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        offset = self.offset(node)
        return arg_name if offset == "0" else f"{arg_name} + {offset}"


    def size(self, node: IRNode, start_index: int, end_index: Optional[int] = None, default_value: int = 0) -> str:
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))

        sizes = node.get_size()[start_index : end_index + 1]
        if len(sizes) == 0:
            return str(default_value)

        val = sympy_product(sizes)
        return cexpr(self.rename_indexing(val))


    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)

        stride = node.get_stride()[index]
        return cexpr(self.rename_indexing(stride))


    def row_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None or len(node.get_stride()) < 2:
            return str(default_value)

        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        else:
            raise RuntimeError(f"At least 1 stride should be 1. Strides: {node.get_stride()=}")


class CUDATemplateCaller(ChoiceCaller):
    def __init__(
        self, name, input_nodes, layout, make_kernel_render, bmreq
    ):
        super().__init__(name, input_nodes, layout)
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f"CUDATemplateCaller({self.bmreq.module_path})"

    def call_name(self):
        return f"template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        return TensorBox.create(
            TemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
            )
        )
