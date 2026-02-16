# BitNet WebGPU PoC

An experimental proof-of-concept aimed at running 1-bit AI models (like Microsoft's BitNet b1.58) natively in JavaScript using WebGPU.

## The Goal
Standard AI relies on heavy floating-point (`f32`) matrix multiplication, which creates a massive memory bottleneck. This PoC explores rewriting the core mathematical kernels in WGSL (WebGPU Shading Language) to use ternary weights (-1, 0, 1). 

By replacing complex multiplication with simple addition and subtraction directly on the GPU, we aim to drastically reduce memory bandwidth and make running massive Large Language Models in the browser a reality.

## Current Status
- [x] Initial repository setup
- [x] WebGPU environment configuration
- [x] WGSL compute shader for ternary math
- [x] Browser-based execution via local web server
- [x] Bit-packed weights (16 ternary values per u32)
- [x] Branchless ternary arithmetic (no if/else, no select)
- [x] 2D tiled mat-vec kernel using `var<workgroup>`
- [x] Automated CPU vs GPU validation (PASS/FAIL)
- [x] Stress-tested at 4096×4096 (~16.7M parameters)
- [x] Isolated GPU setup vs compute timing
- [x] Real AI weights from Hugging Face (`microsoft/bitnet-b1.58-2B-4T`)
- [ ] Tokenizer integration for text-in → text-out inference

## How It Works

The WGSL compute shader receives an input vector and a ternary weight vector (`{-1, 0, +1}`). Instead of multiplying, it branches on each weight:

| Weight | Operation | Cost |
|--------|-----------|------|
| `+1`   | Copy the input value | One add |
| `-1`   | Negate the input value | One subtract |
| `0`    | Output zero (skip) | Nothing |

This completely eliminates floating-point multiplication, which is the core insight behind BitNet b1.58.

### Bit-packing (2 bits per weight)

Weights are packed into `u32` buffers to reduce memory bandwidth:

| 2-bit code | Weight | Meaning |
|------------|--------|---------|
| `00`       | 0      | skip    |
| `01`       | +1     | add     |
| `10`       | -1     | subtract |

The WGSL kernel unpacks 16 weights per `u32` and applies a branchless
bitmask to include or exclude the input value.

### Branchless ternary math

Instead of `if/else`, the kernel uses full-width bitmasks to select
`+input`, `-input`, or `0` without warp divergence.

### 2D tiled mat-vec kernel

For matrix-vector multiplication, input tiles are cached in
`var<workgroup>` shared memory. Each workgroup computes one output row,
and a reduction across the workgroup produces the final dot product.

### Real AI weight integration

The `extract_weights.py` script downloads pre-trained weights from
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
on Hugging Face. The model stores weights as row-packed `uint8` tensors
(4 ternary values per byte). The script unpacks these, then re-packs into
our kernel's column-packed `uint32` format (16 weights per `u32`) and saves
a binary `.bin` file that the browser can `fetch()` directly into GPU buffers.

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/qwatts-dev/bitnet-webgpu-poc.git
   cd bitnet-webgpu-poc
   ```

2. **Start a local web server**
   ```bash
   npx serve . -l 8080
   ```

3. **Open in a WebGPU-capable browser** (Chrome 113+, Edge 113+, Safari 18+)
   
   Navigate to `http://localhost:8080` — the page will automatically run the ternary-weight compute shader on your GPU and display the results. Tests 1 and 2 run CPU-vs-GPU validation with synthetic data. Test 3 runs a real AI weight matrix through the WebGPU kernel.

## Extracting Real AI Weights (Optional)

To reproduce the real-weight integration (Test 3), you need Python 3.10+ and a Hugging Face account:

1. **Install Python dependencies**
   ```bash
   pip install torch safetensors huggingface-hub numpy
   ```

2. **Run the extraction script**
   ```bash
   python extract_weights.py
   ```
   This will:
   - Download `microsoft/bitnet-b1.58-2B-4T` (~1.1 GB safetensors file)
   - Extract `model.layers.0.mlp.down_proj.weight` (2560 × 6912 ternary matrix)
   - Unpack the HF row-packed `uint8` format (4 weights/byte)
   - Re-pack into our JS kernel's column-packed `uint32` format (16 weights/u32)
   - Save `bitnet_layer_0_down_proj.bin` (4.2 MB)

3. **Serve and test** — the `.bin` file must be in the same directory as `index.html`:
   ```bash
   npx serve . -l 8080
   ```

## Latest Benchmark Results

### Test 2: Synthetic 4096×4096 mat-vec (CPU vs GPU validation)

| Metric | iPhone 14 Pro Max | iPad Air M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| CPU mat-vec | 78 ms | 71 ms | 93 ms |
| GPU setup | 84 ms | 73 ms | 98 ms |
| GPU compute | **50 ms** | **24 ms** | **3.1 ms** |
| Speedup | **1.6×** | **3.0×** | **30.1×** |
| Max error | 2.08e-3 | 2.08e-3 | 2.08e-3 |

### Test 3: Real AI weights — `microsoft/bitnet-b1.58-2B-4T`

Layer: `model.layers.0.mlp.down_proj` (2560 × 6912 = 17.7M ternary params)

| Metric | iPhone 14 Pro Max | iPad Air M3 | MacBook M2 Max |
|--------|-------------------|-------------|----------------|
| GPU setup | 2 ms | 1 ms | 1.4 ms |
| GPU compute | **13 ms** | **7 ms** | **4.8 ms** |
| Non-zero outputs | 2560/2560 (100%) | 2560/2560 (100%) | 2560/2560 (100%) |
| Result | ✅ PASS | ✅ PASS | ✅ PASS |

**Key takeaways:**

- **Real AI weights work end-to-end.** Pre-trained ternary weights from Hugging Face are extracted, bit-packed, fetched by the browser, and processed by the WebGPU kernel — producing non-trivial output on all three devices.
- **M2 Max dominates on compute** at 3.1 ms (Test 2) and 4.8 ms (Test 3), benefiting from its 30-core GPU and 400 GB/s memory bandwidth.
- **Even an iPhone processes a real 17.7M-parameter layer in 13 ms** — well within interactive latency requirements.
- **Setup cost is a one-time expense** — the pipeline and buffers would be reused across tokens in a real inference loop, so the compute time is what matters for throughput.
- **Numerical precision is identical** across all three Apple GPU generations — max error of 2.08e-3 at the same row, confirming deterministic f32 accumulation.

## Project Structure

| File | Description |
|------|-------------|
| `index.html` | Minimal page that loads the kernel as an ES module |
| `bitnet-kernel.js` | WebGPU setup, WGSL shaders, buffer management, and result display |
| `extract_weights.py` | Python script to extract and bit-pack weights from Hugging Face |
| `bitnet_layer_0_down_proj.bin` | Pre-packed weight binary for Test 3 (generated by `extract_weights.py`) |
| `package.json` | Project metadata |

## License
This project is licensed under the MIT License - free and open for anyone to contribute, fork, and hack on!
