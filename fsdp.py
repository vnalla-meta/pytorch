"""
torchrun --standalone --nproc_per_node=2 test_basic.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch._dynamo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
def init():
    torch.manual_seed(0)
    fsdp_kwargs = {
        "use_orig_params": True,
        "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
    }
    model = nn.Linear(3, 3, device="cuda")
    model = FSDP(
        model,
        **fsdp_kwargs,
    )
    # TODO: Add `model = torch.compile(model)` here if desired
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optim

def printing_eager(gm, inputs):
    gm.graph.print_tabular()
    return gm.forward

def run(model, optim):
    torch.manual_seed(42)
    losses = []
    inp = torch.randn((2, 3), device="cuda")
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        # inp = torch.randn((2, 3), device="cuda")
        out = model(inp)
        loss = out.sum()
        losses.append(loss)
        loss.backward()
        optim.step()
    return losses

def main(compiled):
    model, optim = init()
    if compiled:
        model = torch._dynamo.optimize("eager", nopython=True)(model)
    return run(model, optim)

if __name__ == "__main__":
    import time
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    eager = main(compiled=False)
    print("EAGER:", eager)
    time.sleep(2)
    compiled = main(compiled=True)
    print("COMPILED:", compiled)
    # for i in range(0, len(eager)):
        # distance = torch.norm(eager[i] - compiled[i], p=2)  # p=2 specifies the Euclidean norm
        # output_file = f"output_{os.getpid()}.txt"
        # with open(output_file, 'a') as f:
            # f.write(f"DIST AT {i} : {str(distance.item())}")
        # print("DIST AT", i, distance.item())
    # breakpoint()
