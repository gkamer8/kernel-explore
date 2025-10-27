import torch
import triton.language as tl
import triton
import os

PLOT_PATH = os.path.join("plots")  # relative path fyi
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to dram
    y_ptr,  # Pointer to dram
    output_ptr,  # Pointer to dram
    n_elements,  # Integer
    BLOCK_SIZE: tl.constexpr, # tl.constexprs are for shape values!
):
    # One dimensional launch grid
    pid = tl.program_id(0)  # Argument is "axis"
    # So, this function is being launched many times
    # We get the pid so that we know "which" one we're in
    # We calculate the portion of the vector we're adding using that pid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements  # Memory safety I guess

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # Preallocate output
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()


    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # x axis of plot
        x_vals=[2**x for x in range(10, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={}
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [.5, .2, .8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    
    vector_add_path = os.path.join(PLOT_PATH, "vector-add")
    os.makedirs(vector_add_path, exist_ok=True)
    benchmark.run(print_data=True, save_path=vector_add_path)
    print(f"See plot in {vector_add_path}")

if __name__ == "__main__":
    main()
