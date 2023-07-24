# Owner(s): ["module: dynamo"]
import functools
import unittest
from importlib import import_module

import torch

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.wrap import handle_activation_checkpoint
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils.checkpoint import checkpoint


requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


def count_ops(gm, args, freq, op):
    actual_count = [node.target for node in gm.graph.nodes].count(op)
    assert (
        actual_count == freq
    ), f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
    return gm


def count_ops_list(gm, args, ops, freqs):
    for op, freq in zip(ops, freqs):
        actual_count = [node.target for node in gm.graph.nodes].count(op)
        assert (
            actual_count == freq
        ), f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
    return gm


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


from torch.utils._python_dispatch import TorchDispatchMode


class CachingTorchDispatchMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self.policy_fn(func, *args, **kwargs):
            out = func(*args, **kwargs)
            self.storage.append(out)
            return out
        return func(*args, **kwargs)


class CachedTorchDispatchMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self.policy_fn(func, *args, **kwargs):
            out = self.storage.pop(0)
            return out
        return func(*args, **kwargs)


def _get_default_policy(no_recompute_list=None):
    # By default, save output of these ops
    _default_no_recompute_list = [
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
    ]
    if no_recompute_list is None:
        no_recompute_list = _default_no_recompute_list

    def _default_policy(func, *args, **kwargs):
        return func in no_recompute_list

    return _default_policy


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(
                result,
                expected,
                msg="Output mismatch between torch.compile and eager versions"
            )
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(
                    arg.grad,
                    cloned_arg.grad,
                    msg="Gradient mismatch between torch.compile and eager versions"
                )

    @requires_cuda()
    def test_tags_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_via_global_checkpoint(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # This goes through VariableBuilder
            return checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_multiple_checkpoints(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    @requires_cuda()
    def test_tags_decomps(self):
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        def gn(x, y):
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            # x = torch.utils.checkpoint.checkpoint(gn, x, y)
            return x

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x)

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)

    @requires_cuda()
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            a = torch.sigmoid(torch.matmul(x, y))
            torch._dynamo.graph_break()
            return torch.cos(a)

        def fn(x, y):
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(cnt.graphs), 2)

    @requires_cuda()
    def test_kwargs(self):
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4, 4, requires_grad=True)
        args = (x, y, z)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)

        wrap_node = find_first_node(cnt.graphs[0], handle_activation_checkpoint)
        # one for checkpoint, and 3 for x, y, z
        self.assertEqual(len(wrap_node.args), 4)

        body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    @requires_cuda()
    def test_symints_location(self):
        def gn(x, y):
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, x, y)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(cnt.graphs), 2)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 3)

    @requires_cuda()
    def test_tags_selective_checkpoint_gemm_only(self):
        def selective_checkpointing_context():
            no_recompute_list = [torch.ops.aten.mm.default]
            storage = []
            caching_mode = CachingTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            cached_mode = CachedTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            return caching_mode, cached_mode

        def gn(x, y):
            return torch.sigmoid(torch.sigmoid(torch.matmul(torch.matmul(x, y), y)))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context,
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)


    @requires_cuda()
    def test_tags_selective_checkpoint_custom_rule(self):
        class CachingTorchDispatchModeCustomRule(TorchDispatchMode):
            def __init__(self, storage, no_recompute_list):
                self.storage = storage
                self.no_recompute_list = no_recompute_list
                self.sigmoid_count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                if func == torch.ops.aten.sigmoid.default:
                    self.sigmoid_count += 1
                if func in self.no_recompute_list and not (
                    func == torch.ops.aten.sigmoid.default and self.sigmoid_count == 2
                ):
                    # Saves output of all compute ops, except second sigmoid
                    # (i.e. only recomputes second sigmoid in backward pass)
                    out = func(*args, **kwargs)
                    self.storage.append(out)
                    return out
                return func(*args, **kwargs)

        class CachedTorchDispatchModeCustomRule(TorchDispatchMode):
            def __init__(self, storage, no_recompute_list):
                self.storage = storage
                self.no_recompute_list = no_recompute_list
                self.sigmoid_count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                if func == torch.ops.aten.sigmoid.default:
                    self.sigmoid_count += 1
                if func in self.no_recompute_list and not (
                    func == torch.ops.aten.sigmoid.default and self.sigmoid_count == 2
                ):
                    out = self.storage.pop(0)
                    return out
                return func(*args, **kwargs)

        def selective_checkpointing_context():
            storage = []
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            caching_mode = CachingTorchDispatchModeCustomRule(storage, no_recompute_list)
            cached_mode = CachedTorchDispatchModeCustomRule(storage, no_recompute_list)
            return caching_mode, cached_mode

        def gn(x, y):
            return torch.sigmoid(
                torch.sigmoid(torch.sigmoid(torch.matmul(torch.matmul(x, y), y)))
            )

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context,
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops_list,
            freqs=[2, 3],
            ops=[
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ],
        )
        bw_compiler = functools.partial(
            count_ops_list,
            freqs=[
                4,
                1,  # sigmoid recompute count is 1, because only second sigmoid is recomputed
            ],
            ops=[
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ],
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)


    @requires_cuda()
    def test_tags_selective_checkpoint_outplace_op(self):
        def selective_checkpointing_context():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            storage = []
            caching_mode = CachingTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            cached_mode = CachedTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            return caching_mode, cached_mode

        def gn(x, y):
            return torch.sigmoid(torch.selu(torch.matmul(torch.matmul(x, y), y)))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context,
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops_list,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops_list,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)


    @requires_cuda()
    def test_tags_selective_checkpoint_inplace_op(self):
        def selective_checkpointing_context():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            storage = []
            caching_mode = CachingTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            cached_mode = CachedTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            return caching_mode, cached_mode

        def gn(x, y):
            return torch.sigmoid(torch.selu_(torch.matmul(torch.matmul(x, y), y)))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context,
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops_list,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops_list,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        with self.assertRaisesRegex(
            AssertionError,
            "In-place ops are not supported in selective checkpointing region under torch.compile"
        ):
            self._validate(fn, backend, x, y)


    @requires_cuda()
    def test_tags_selective_checkpoint_random_op(self):
        def selective_checkpointing_context():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            storage = []
            caching_mode = CachingTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            cached_mode = CachedTorchDispatchMode(
                _get_default_policy(no_recompute_list=no_recompute_list), storage
            )
            return caching_mode, cached_mode

        def gn(x, y):
            return torch.sigmoid(
                torch.nn.functional.rrelu(
                    torch.dropout(torch.matmul(torch.matmul(x, y), y), p=0.5, train=True)
                )
            ).cauchy_()

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context,
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops_list,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops_list,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        with self.assertRaisesRegex(
            AssertionError,
            "Random ops are not supported in selective checkpointing region under torch.compile"
        ):
            self._validate(fn, backend, x, y)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
