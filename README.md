# Model Offloading
This repo benchmarks two techniques for model offloading during inference in PyTorch, with examples showing how to dynamically move or copy model layers between CPU and GPU, and how to overlap layer loading with computation using CUDA streams.

## Overview
The repo contains two scripts:

- Move.py: Implements model offloading by moving layers in-place between CPU and GPU. It includes:
    - Layer-by-layer inference:
    Sequentially moves each layer to the GPU, processes the input, and then moves the layer back to CPU.
    - Pipelined inference:
    Splits the model into chunks and overlaps GPU transfers with computation by using separate CUDA streams and events.
- Copy.py: Implements model offloading by creating copies of layers (using deepcopy) on the GPU. The original model remains on CPU. So we won't need to move the model from GPU to CPU in this case.
    - Full model copy inference:
    Copies the entire model to the GPU for inference.
    - Layer-by-layer inference:
    Creates a GPU copy of each layer, processes the input, and then discards the copy.
    - Pipelined inference:
    Splits the model into chunks and asynchronously creates GPU copies for each chunk, overlapping loading with computation.

Both scripts use a large multi-layer perceptron (MLP) with 32 hidden layers (each with a Linear layer and ReLU activation) followed by a final classification layer. The following result is run on a RTX 3090 with PCIE 4.0:

| Inference Approach       | Move.py (s) | Copy.py (s) |
| ------------------------ | ----------- | ----------- |
| Full                     | N/A         | 0.032591    |
| Layer-by-Layer           | 0.034961    | 0.021093    |
| Pipelined (chunk_size=4) | 0.018758    | 0.019857    |

The (relevent) speed is highly dependent on PCIE bandwidth, as Move requires copying from GPU to CPU. Though the overhead is small when we have a high bandwidth.