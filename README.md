# Mixture-of-Kittens (MoK)

Mixture-of-Kittens (MoK) is a fully deterministic mixture-of-experts (MoE) training megakernel optimized for B200 and B300 DGX nodes. It supports intra-host expert-parallel sizes 1, 2, 4, and 8, with compile-time-specialized peer handling for each size. MoK fuses all intra-host MoE computation and communication into a single kernel, overlapping compute and inter-GPU networking at configurable granularity, and fully eliminates CPU-GPU synchronization. It supports BF16 and MXFP8 and covers both forward and backward passes.

For multi-host training, use one EP group of size 1, 2, 4, or 8 within each host and scale across hosts with a separate data-parallel or FSDP group. The megakernel intentionally does not issue its fine-grained peer loads, stores, or multicast barriers across the inter-node fabric.

For a deep dive, read our [blog post](https://cursor.com/blog/mixture-of-kittens).

## Performance

We evaluated MoK at two levels: standalone MoE layers, with all benchmark code available in [`benchmarks`](./benchmarks), and end-to-end training on our internal production stack across multiple NVL72 racks. The following figure summarizes the standalone MoE layer benchmarks:

![MoE benchmark results](./figures/moe_benchmarks.png)

Compared with the fastest baseline, MoK is up to 2.37x faster for the MXFP8 forward, 1.78x for the MXFP8 backward, 1.92x for the BF16 forward, and 1.58x for the BF16 backward.

See our [blog post](https://cursor.com/blog/mixture-of-kittens) for the methodology and full results.

## Setup

### Requirements

- Linux host with one, two, four, or eight NVIDIA Blackwell SM100/SM103 GPUs in one DGX NVSwitch domain
- Python 3.12 or later, including the Python development headers (`Python.h`)
- GNU Make and a CUDA-compatible host C++ compiler
- CUDA 13.x NVCC. Its major and minor version must exactly match the CUDA version reported by PyTorch.
- PyTorch 2.13 with CUDA support

MoK is compiled locally during installation. It does not support CPU-only PyTorch,
non-Linux hosts, or GPUs outside the SM100/SM103 Blackwell families.

### Get the source

MoK uses the ThunderKittens Git submodule. Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/Godofnothing/mixture-of-kittens.git
cd mixture-of-kittens
```

If you already cloned the repository without submodules, initialize it before
installing:

```bash
git submodule update --init --recursive
```

### PyTorch and CUDA

The installed PyTorch build must target CUDA 13.x, and its CUDA version must
match NVCC exactly at the major and minor level. MoK is built and tested against
CUDA 13.0. The project currently pins PyTorch to 2.13.

Create and activate a Python environment, then install the pinned PyTorch
version before installing MoK. Installing PyTorch first is required because the
MoK extension imports it while building:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.13"
```

Use the PyTorch package appropriate for your CUDA 13.x installation. Before
continuing, confirm that it is CUDA-enabled and that its reported CUDA version
matches `nvcc --version`:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
```

### Installation

From the repository root, install MoK without build isolation. This installs
MoK's remaining Python and CUDA package dependencies and builds its CUDA
extension. By default, it builds for SM103:

```bash
python -m pip install . --no-build-isolation
```

To build for SM100 instead, set the `MOK_ARCH` environment variable:

```bash
MOK_ARCH=SM100 python -m pip install . --no-build-isolation
```

To verify the installation:

```bash
python -c "import mok; print(mok.__version__)"
```

### Development

For development, install MoK in editable mode once, then use `make` for fast rebuilds:

```bash
python -m pip install -e . --no-build-isolation
make
```

#### Unit tests

Launch multi-GPU unit tests through `torchrun` and `pytest`:

```bash
python -m pip install -e ".[test]" --no-build-isolation
torchrun --standalone --nproc-per-node=<ep-size> -m pytest -s <test-path>
```

## Getting Started

MoK provides two layers of abstraction:

- **Ops layer** (`mok/ops.py`): the low-level API that lets you call our CUDA kernel implementations directly with minimal overhead. With this, you fully manage your own data (e.g., symmetric tensors) and coordinate the kernel calls yourself to implement the MoE layer's computation and communication.
- **Functional layer** (`mok/functional.py`): the higher-level API that handles more for you, such as maintaining scratch memory and coordinating kernel launches. It consists of workspace creation functions, a schedule function, and forward/backward functions.

The functional layer is our choice for production training, so we recommend it unless you have specific needs. We explain only the functional layer from here.

With the functional API, MoK is simple to use: call `schedule(...)` once to build the dispatch/combine schedule, then pass that same schedule to `forward(...)` and `backward(...)`. Before doing so, however, you need to define and manage a **config** and a **workspace**.

### Config

MoK exposes 5 hyperparameters that can affect the performance of MoE execution. Because optimal values depend heavily on the workload and EP size, you should tune and sweep them before using MoK in production. The defaults target a full eight-GPU DGX node and are correctness-safe for smaller EP groups.

- `fwd_num_comm_sms`: the number of communication SMs during forward. The DGX default is 16 (two per rank); sweep nearby even values for the actual model shape.
- `bwd_num_comm_sms`: the number of communication SMs during backward. The DGX default is 16; sweep it independently from forward.
- `minibatch_size`: the granularity of computation–communication overlap. This is an important parameter in MoK, and you must tune it properly to get optimal performance. We recommend values between 2048 and 16384.
- `macrobatch_size`: the token ring buffer size. Setting this to a large value (e.g., 131072) means the ring buffer is used only once. You should maximize this value to fill the available GPU memory.
- `schedule_capacity_multiplier`: defaults to 0.5. This should be set to the worst-case fraction of tokens routed to a single rank. Setting it to 1 assumes the absolute worst case (all tokens routed to one rank) but adds kernel scheduling overhead (due to expert padding, the actual worst case is slightly above 1.0). Ideally, use a higher value during the first steps of training when expert imbalance is bad, then reduce it to around 0.5 in later steps. Note that decreasing this value does not save GPU memory meaningfully, as the schedule table is at most a few megabytes.

You can set these values when creating the `MoKConfig` dataclass, which you pass to all functional-layer functions.

### Multi-host process groups

Do not pass the multi-host world group to MoK. Choose `ep_size` from 1, 2, 4, or 8. All ranks must create groups in the same order; then each rank passes its own node-local group to `get_workspace`:

```python
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()
ep_size = 4  # 1, 2, 4, or 8; groups must not cross a host boundary
if world_size % ep_size:
    raise ValueError("world size must be divisible by ep_size")

ep_group = None
for first_rank in range(0, world_size, ep_size):
    ranks = list(range(first_rank, first_rank + ep_size))
    group = dist.new_group(ranks=ranks)
    if rank in ranks:
        ep_group = group

assert ep_group is not None
```

This example assumes consecutive global ranks are node-local and that group boundaries do not cross hosts, as with the usual `torchrun` rank layout. Use `ep_group` for MoK and construct the orthogonal data/FSDP groups in the training framework.

### Workspace

MoK relies on PyTorch symmetric memory to allocate and manage inter-GPU symmetric buffers, or identically sized memory allocations across many GPUs (i.e., all GPUs in an EP group). These buffers serve as the source/destination of token dispatch/combine, along with a few other purposes. We call the entire collection of symmetric memory, along with the other metadata and scratchpads MoK needs, the **workspace**. We provide a data structure and functions for creating and destroying workspaces.

Because symmetric buffers are expensive to allocate, we recommend creating a single workspace per model and reusing it across layers. For this, we provide the `get_workspace(...)` function, which automatically caches workspaces with identical properties (EP group, device, model shapes, etc.). If you prefer to manage a workspace's lifetime yourself, the `create_workspace(...)` function does not cache.

### MXFP8

To run MoK in MXFP8 mode, pass the activations as-is in BF16 while prequantizing the weights to MXFP8. We *could* quantize the weights inside our kernels, but prequantizing leaves better opportunities for things like FSDP, so we keep it separate and provide the `mxfp8_quantize(...)` function at the ops layer so you can prequantize the weights yourself.

### Example (MXFP8 forward and backward using the functional layer)

The following is a canonical example of implementing MoE forward and backward with MoK in MXFP8 mode:

```python
import torch.distributed as dist

from mok import functional, ops

# Inputs:
#   num_local_tokens:    int
#   hidden_size:         int
#   intermediate_size:   int
#   topk:                int
#   num_local_experts:   int
#   x:                   torch.bfloat16 [num_local_tokens, hidden_size]
#   topk_experts:        torch.int64    [num_local_tokens, topk]
#   router_weights:      torch.float32  [num_local_tokens, topk]
#   w_shared_gate:       torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_up:         torch.bfloat16 [intermediate_size, hidden_size]
#   w_shared_down:       torch.bfloat16 [hidden_size, intermediate_size]
#   w_routed_gate:       torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_up:         torch.bfloat16 [num_local_experts, intermediate_size, hidden_size]
#   w_routed_down:       torch.bfloat16 [num_local_experts, hidden_size, intermediate_size]
#   d_output:            torch.bfloat16 [num_local_tokens, hidden_size]

config = functional.MoKConfig() # tune for your workload
workspace = functional.get_workspace(
    config,
    ep_group, # a node-local group of size 1, 2, 4, or 8
    device=x.device,
    num_local_tokens=num_local_tokens,
    hidden_size=hidden_size,
    topk=topk,
)


########################
# Weight quantization
########################

(
    w_routed_gate_fp8,
    w_routed_gate_sc,
    w_routed_gate_t_fp8,
    w_routed_gate_t_sc,
) = ops.mxfp8_quantize(w_routed_gate, True, True)
(
    w_routed_up_fp8,
    w_routed_up_sc,
    w_routed_up_t_fp8,
    w_routed_up_t_sc,
) = ops.mxfp8_quantize(w_routed_up, True, True)
(
    w_routed_down_fp8,
    w_routed_down_sc,
    w_routed_down_t_fp8,
    w_routed_down_t_sc,
) = ops.mxfp8_quantize(w_routed_down, True, True)


########################
# Forward
########################

schedule = functional.build_schedule(
    workspace,
    config,
    topk_experts,
    num_local_experts=num_local_experts,
)
output, forward_context = functional.forward(
    config,
    workspace,
    schedule,
    x,
    router_weights,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    (w_routed_gate_fp8, w_routed_gate_sc),
    (w_routed_up_fp8, w_routed_up_sc),
    (w_routed_down_fp8, w_routed_down_sc),
)

# Save `schedule` and `forward_context` for the backward


########################
# Backward
########################

(
    d_x,
    d_router_weights,
    d_w_routed_gate,
    d_w_routed_up,
    d_w_routed_down,
    d_w_shared_gate,
    d_w_shared_up,
    d_w_shared_down,
) = functional.backward(
    config,
    workspace,
    schedule,
    forward_context,
    d_output,
    x,
    router_weights,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    (w_routed_gate_fp8, w_routed_gate_sc, w_routed_gate_t_fp8, w_routed_gate_t_sc),
    (w_routed_up_fp8, w_routed_up_sc, w_routed_up_t_fp8, w_routed_up_t_sc),
    (w_routed_down_t_fp8, w_routed_down_t_sc),
)
```

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

Mixture-of-Kittens is released under the Apache 2.0 License. See [`LICENSE`](LICENSE).

## Citation

If you use this work, please cite:

```
Stuart H. Sul, Nash Brown, Henry Wildermuth, William Lin, and Federico Cassano. "Mixture-of-Kittens: MoE Megakernel for NVL72s." Cursor Research, Aug 2026. https://github.com/cursor/mixture-of-kittens
```

Or in BibTeX:

```bibtex
@misc{sul2026mok,
  title        = {Mixture-of-Kittens: {MoE} Megakernel for {NVL72s}},
  author       = {Stuart H. Sul and Nash Brown and Henry Wildermuth and William Lin and Federico Cassano},
  organization = {Cursor Research},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/cursor/mixture-of-kittens}},
}
```
