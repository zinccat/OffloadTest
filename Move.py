# Move.py
# Using layer.to(device) to move layer between CPU and GPU and overlap layer loading with computation.

import torch
import torch.nn as nn
import time

torch.manual_seed(0)

class LargeMLP(nn.Module):
    """
    A somewhat larger MLP with multiple Linear + ReLU pairs.
    By default:
      - 8 linear layers, each output dimension = 512
      - ReLUs in between
      - Then 1 final linear to 10 classes
    """
    def __init__(self, input_dim=1024, hidden_dim=1024, num_layers=32, num_classes=10):
        super(LargeMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        # Final classification layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def chunked_layers(model, chunk_size=4):
    """
    Split the model's Sequential into chunks of size `chunk_size`.
    Each chunk is a small list of sub-layers (e.g. [Linear, ReLU]).
    """
    all_layers = list(model.net)
    chunks = []
    for i in range(0, len(all_layers), chunk_size):
        chunks.append(all_layers[i : i + chunk_size])
    return chunks


def move_layers_to_device(layers, device, non_blocking=True):
    """
    Move a list of layers in-place to a given device (CPU or CUDA).
    """
    for layer in layers:
        layer.to(device, non_blocking=non_blocking)


def run_chunk(chunk, x):
    """
    Forward pass of 'chunk' (a list of layers) on tensor x.
    """
    with torch.no_grad():
        for layer in chunk:
            x = layer(x)
    return x

def layer_by_layer_inference(model, x_cpu):
    """
    Demonstration of a simple, serial approach:
      - For each layer in the model:
         1) Move the layer to GPU
         2) Move x to GPU (if not already)
         3) Forward pass
         4) Move layer back to CPU
      - Finally bring output back to CPU.
    
    This does NOT overlap layer loading with computation; 
    everything is done in sequence.
    """
    # If no CUDA is available, fallback to a normal CPU forward
    if not torch.cuda.is_available():
        with torch.no_grad():
            return model(x_cpu)

    device = torch.device("cuda")

    # 1) Move input to GPU 
    x = x_cpu.to(device)

    # We don't need extra streams for a purely serial approach
    # Just do everything on the default stream
    with torch.no_grad():
        for layer in model.net:
            # Move layer to GPU
            layer.to(device)
            # Forward pass
            x = layer(x)
            # Move layer back to CPU
            layer.to("cpu")

    # Now move the final result x back to CPU
    x = x.to("cpu")

    return x

def pipelined_inference(model, x_cpu, chunk_size=4):
    """
    A fine-grained overlapping pipeline:
    - Preload the first chunk.
    - For each chunk i in [0..N-1]:
        1) Wait for chunk i to finish loading -> compute it (comp_stream).
        2) In parallel, load chunk (i+1) on load_stream (if it exists).
        3) When chunk i is done computing, unload chunk i in offload_stream
           (but only after compute is finished).
    - Finally, move the result back to CPU.

    We use CUDA events to coordinate “compute finished on chunk i”
    so the load_stream can unload chunk i safely.
    """
    # Fallback to CPU forward if CUDA is not available
    if not torch.cuda.is_available():
        with torch.no_grad():
            return model(x_cpu)

    device = torch.device("cuda")
    net_chunks = chunked_layers(model, chunk_size)
    num_chunks = len(net_chunks)

    # Create three streams: one for loading, one for computation, one for offloading.
    load_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    offload_stream = torch.cuda.Stream()

    # Create events to signal when compute is done for each chunk.
    compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_chunks)]
    # Create events to signal when loading is done for each chunk.
    load_done_events = [None] * num_chunks

    # 1) Move input to GPU on the compute stream.
    with torch.cuda.stream(comp_stream):
        y = x_cpu.to(device, non_blocking=True)

    # 2) Pre-load chunk 0 on the load stream.
    with torch.cuda.stream(load_stream):
        move_layers_to_device(net_chunks[0], device, non_blocking=True)
        # Record an event to mark when chunk 0 is loaded.
        load_done_events[0] = torch.cuda.Event(enable_timing=False)
        load_done_events[0].record(load_stream)

    # 3) Process each chunk in a pipelined fashion.
    for i in range(num_chunks):
        # a) Wait for chunk i to finish loading (only the current chunk).
        comp_stream.wait_event(load_done_events[i])

        # b) Compute chunk i on the compute stream.
        with torch.cuda.stream(comp_stream):
            y = run_chunk(net_chunks[i], y)
            compute_done_events[i].record(comp_stream)

        # c) On the load stream, in parallel:
        #    - If there's a next chunk, start loading chunk (i+1).
        if i + 1 < num_chunks:
            with torch.cuda.stream(load_stream):
                move_layers_to_device(net_chunks[i + 1], device, non_blocking=True)
                load_done_events[i + 1] = torch.cuda.Event(enable_timing=False)
                load_done_events[i + 1].record(load_stream)

        # d) Offload (unload) chunk i on the offload stream, but only after compute is done.
        with torch.cuda.stream(offload_stream):
            offload_stream.wait_event(compute_done_events[i])
            move_layers_to_device(net_chunks[i], "cpu", non_blocking=True)
        # Ensure that any further load_stream operations wait until offloading is done, as we have limited GPU memory.
        load_stream.wait_stream(offload_stream)

    # 4) Once all chunks are computed, move the final result back to CPU.
    comp_stream.wait_event(compute_done_events[-1])
    with torch.cuda.stream(comp_stream):
        y = y.to("cpu", non_blocking=True)

    # Synchronize all streams to ensure completion.
    comp_stream.synchronize()
    load_stream.synchronize()
    offload_stream.synchronize()

    return y

def pin_model(model):
    for param in model.parameters():
        param.data = param.data.pin_memory()
    for buf in model.buffers():
        buf.data = buf.data.pin_memory()

if __name__ == "__main__":
    # Instantiate model & input
    torch.manual_seed(0)
    model = LargeMLP().eval().cpu()
    # pin_model(model)

    x_cpu = torch.randn(32, 1024).pin_memory()  # batch=32, input_dim=1024

    # Warm-up
    for _ in range(10):
        _ = layer_by_layer_inference(model, x_cpu)
        _ = pipelined_inference(model, x_cpu, chunk_size=4)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = layer_by_layer_inference(model, x_cpu)
    torch.cuda.synchronize()
    end = time.time()
    # print("Output shape:", out.shape)
    # print("Sample output:", out[0, :5])  # just to verify
    print(f"Average time (layer-by-layer): {(end - start)/100:.6f} seconds")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = pipelined_inference(model, x_cpu, chunk_size=4)
    torch.cuda.synchronize()
    end = time.time()
    # print("Output shape:", out.shape)
    # print("Sample output:", out[0, :5])  # just to verify
    print(f"Average time (pipelined, chunk_size=4): {(end - start)/100:.6f} seconds")