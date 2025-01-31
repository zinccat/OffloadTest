# Copy.py
# Copying layers to GPU and overlapping layer loading with computation, then discarding the GPU copies.

import torch
import torch.nn as nn
import copy
import time

torch.manual_seed(0)

class LargeMLP(nn.Module):
    """
    A somewhat larger MLP with multiple Linear + ReLU pairs.
    By default:
      - 32 linear layers (each followed by a ReLU)
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
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def chunked_layers(model, chunk_size=4):
    """
    Split the model's Sequential into chunks of size `chunk_size`.
    Each chunk is a list of sub-layers.
    """
    all_layers = list(model.net)
    chunks = []
    for i in range(0, len(all_layers), chunk_size):
        chunks.append(all_layers[i : i + chunk_size])
    return chunks

def copy_layers_to_device(layers, device, non_blocking=True):
    """
    Create copies (via deepcopy) of a list of layers and move them to the specified device.
    The original layers on CPU remain unchanged.
    """
    new_layers = []
    for layer in layers:
        layer_copy = copy.deepcopy(layer).to(device, non_blocking=non_blocking)  # Create a separate copy.
        new_layers.append(layer_copy)
    return new_layers

def run_chunk(chunk, x):
    """
    Run a forward pass through a list of layers (a chunk) on input tensor x.
    """
    with torch.no_grad():
        for layer in chunk:
            x = layer(x)
    return x

def full_forward(model, x_cpu):
    """
    Run a full forward pass on the model, moving everything to GPU.
    """
    if not torch.cuda.is_available():
        with torch.no_grad():
            return model(x_cpu)

    device = torch.device("cuda")
    with torch.no_grad():
        x = x_cpu.to(device)
        y = model(x)
        y = y.to("cpu")
    return y


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
            layer_copy = copy.deepcopy(layer)
            layer_copy.to(device)
            # Forward pass
            x = layer_copy(x)
            layer_copy = None
            torch.cuda.empty_cache()
    # Now move the final result x back to CPU
    x = x.to("cpu")
    return x

def pipelined_inference(model, x_cpu, chunk_size=4):
    """
    Pipelined inference that:
      - Splits the model into chunks.
      - For each chunk, creates a **copy** of the chunk on GPU.
      - Runs inference chunk-by-chunk.
      
    After a chunk is used for computation, its GPU copy is released by setting its reference to None.
    Then, explicit garbage collection is triggered and the CUDA cache is cleared.
    The original model on CPU remains intact.
    """
    # Fallback to CPU inference if CUDA is not available.
    if not torch.cuda.is_available():
        with torch.no_grad():
            return model(x_cpu)

    device = torch.device("cuda")
    net_chunks = chunked_layers(model, chunk_size)
    num_chunks = len(net_chunks)

    # Create two streams for overlapping loading and computation.
    load_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()

    # Create events to signal when compute is done on a chunk.
    compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_chunks)]
    # Create events to signal when loading of a chunk is finished.
    load_done_events = [None] * num_chunks

    # This list will hold GPU copies of chunks.
    gpu_chunks = [None] * num_chunks

    # 1) Move input to GPU on the compute stream.
    with torch.cuda.stream(comp_stream):
        y = x_cpu.to(device, non_blocking=True)

    # 2) Pre-load (i.e. create a GPU copy of) the first chunk on the load stream.
    with torch.cuda.stream(load_stream):
        gpu_chunks[0] = copy_layers_to_device(net_chunks[0], device, non_blocking=True)
        # Record an event once chunk 0 is loaded.
        load_done_events[0] = torch.cuda.Event(enable_timing=False)
        load_done_events[0].record(load_stream)

    # 3) Process each chunk in a pipelined fashion.
    for i in range(num_chunks):
        # a) Wait for the current chunk's loading to finish.
        comp_stream.wait_event(load_done_events[i])

        # b) Run the current GPU chunk on the compute stream.
        with torch.cuda.stream(comp_stream):
            y = run_chunk(gpu_chunks[i], y)
            compute_done_events[i].record(comp_stream)
            gpu_chunks[i] = None  # Clear the reference immediately after use.

        # c) Preload the next chunk (if any) on the load stream.
        if i + 1 < num_chunks:
            with torch.cuda.stream(load_stream):
                gpu_chunks[i + 1] = copy_layers_to_device(net_chunks[i + 1], device, non_blocking=True)
                # Record an event for when the next chunk is loaded.
                load_done_events[i + 1] = torch.cuda.Event(enable_timing=False)
                load_done_events[i + 1].record(load_stream)
        # Optionally call torch.cuda.empty_cache() if needed.

    # 4) Wait for the last chunk's computation and bring the result back to CPU.
    comp_stream.wait_event(compute_done_events[-1])
    with torch.cuda.stream(comp_stream):
        y = y.to("cpu", non_blocking=True)

    # Synchronize the streams.
    comp_stream.synchronize()
    load_stream.synchronize()

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

    x_cpu = torch.randn(32, 1024).pin_memory()  # batch=32, input_dim=1024

    # Warm-up
    for _ in range(10):
        model = model.cuda()
        _ = full_forward(model, x_cpu)
        model = model.cpu()
        _ = layer_by_layer_inference(model, x_cpu)
        _ = pipelined_inference(model, x_cpu, chunk_size=4)

    # Benchmark
    model = model.cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = full_forward(model, x_cpu)
    torch.cuda.synchronize()
    end = time.time()
    model = model.cpu()
    # print("Output shape:", out.shape)
    # print("Sample output:", out[0, :5])  # just to verify
    print(f"Average time (full): {(end - start)/100:.6f} seconds")

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