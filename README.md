# Parallix Accelerator — Embedded SIMD Compute Accelerator

## Project Overview

The **Parallix Accelerator** is an open-source hardware project aimed at extending the capabilities of the Microwatt OpenPOWER core with a dedicated parallel compute unit. The accelerator is controlled through memory-mapped registers, enabling seamless integration with the CPU without requiring ISA extensions or compiler changes. This project is an entry for the [Microwatt Momentum Hackathon 2025](https://chipfoundry.io/challenges/microwatt).

Unlike traditional GPUs that are primarily focused on graphics pipelines, Parallix is designed for **general-purpose parallel computing (GPGPU-style workloads)** such as AI inference, matrix operations, and data-parallel tasks, while still being capable of supporting lightweight graphics calculations. 

The goal is to design and implement a scalable embedded SIMD accelerator that significantly boosts performance for key workloads like AI inference and graphics processing while maintaining full integration with the open-source Microwatt ecosystem.

> **Scope:** This project targets an **embedded, on-SoC compute accelerator** (not an external/Thunderbolt eGPU).

---

## Architecture & Design Philosophy

Parallix is designed with a **modular architecture** and a **warp-based execution model** for scheduling and grouping, while keeping the overall design simple and lightweight for embedded contexts.

- **Warp-based scheduling:**  
  - Threads are grouped into *warps* equal to the SIMD width.  
  - Warps serve as the scheduling granularity, simplifying hardware and avoiding idle lanes.  
  - All threads in a warp follow the same execution path (no SIMT divergence handling).  

- **SIMD Execution:** Each warp maps to one SIMD engine (8 lanes). A kernel can schedule up to 4 warps in parallel across the 4 SIMD engines, keeping hardware fully utilized while keeping scheduling simple.
>*For the hackathon implementation, we adopt a simplicity-first scheduling strategy: instead of launching many small kernels (which would add overhead and complexity), Parallix maps up to 4 warps per kernel across its SIMD engines. This makes the runtime and hardware controller leaner. More advanced features such as multi-kernel concurrency and dynamic load balancing are left as potential extensions beyond the initial design.*


- **Memory-Mapped Control:** The CPU programs the accelerator by writing to its Memory-Mapped Registers (MMRs). This avoids modifications to the Microwatt CPU pipeline or compiler toolchain.  

While initially targeting the Microwatt OpenPOWER ecosystem, the design is bus-agnostic and can be adapted to other architectures (RISC-V, ARM, etc.) via interface wrappers.

---

## System Integration

**Key Communication Flows:**
1. **Control Path (CPU → Parallix):** Microwatt CPU writes to the Parallix control registers via the Wishbone slave interface.  
2. **Data Path (Parallix → Memory):** Parallix’s DMA controller acts as a Wishbone master for efficient burst transfers to/from main memory.  
3. **Synchronization (Parallix → CPU):** Parallix notifies the host of task completion or error conditions via an interrupt signal and/or status registers; the host may poll status or handle interrupts.

---

## Hardware Execution Model

- **Threads & Warps:**  
  - Each ALU lane executes one *thread*.  
  - Threads are grouped into **warps**, matching SIMD width.  
  - Example: **warp size = 8 threads**.  
  - Scheduler issues instructions at the *warp* level (warps are a scheduling/grouping abstraction aligned to SIMD width), not per-thread.
  

- **SIMD Engines:**  
  - Target configuration: **4 SIMD engines**, each with **8 ALUs**.  
  - This provides **32 concurrent threads per cycle** at full occupancy.  
  - Warp size and engine count are **parameterized** and can be tuned depending on area constraints (~15 mm² OpenFrame SoC).  

---

## Processing Element Capabilities

Each ALU (Processing Element) supports a range of integer and floating-point operations, with packed SIMD execution for smaller data types:

| Data Type | Ops per PE per Cycle | Primary Use Case |
| :--- | :--- | :--- |
| **FP32** | 1 | Vector math, General Compute |
| **INT32** | 1 | Control Logic, Addressing |
| **INT16/FP16** | 2 | AI, Graphics Math |
| **INT8** | 4 | **High-Efficiency AI Inference** |

> *Note: While ALUs support operations commonly used in graphics (e.g., FP32 vector math), the architecture does not currently include fixed-function graphics pipeline hardware such as rasterizers or texture units.*

---

### Core ALU Operations

The Parallix ALUs support the following baseline operations:

1. **Logic & Bitwise**
   - AND, OR, XOR, NOT (32/16/8-bit)
   - Bit shifts

2. **Integer Arithmetic**
   - ADD, SUB, MUL (32/16/8-bit)
   - Fused Multiply-Add (IMAC, 16/8-bit)
   - Min/Max, Compare

3. **Floating-Point Arithmetic**
   - ADD, SUB, MUL (FP32)
   - FMA (FP32)
   - Min/Max, Compare

These operations cover the most common needs for AI inference (quantized math, MACs, ReLU), graphics-like transforms (matrix multiply, dot products), and general-purpose SIMD acceleration.

---

### ALU: Future Extensions (Optional)

While not part of the initial hardware, the following could be added in later revisions if needed:

- **Division (integer / FP32)**  
- **Reciprocal / Reciprocal sqrt**  
- **Special functions (exp, log, sigmoid, tanh)**  
- **Population count / leading-zero count**  
- **Shuffle / permute primitives**  

This approach keeps the current design lightweight and feasible while leaving room for **scalability and specialization** in future versions.

---
## Memory System

- **MMR Control Interface:** Command and configuration registers memory-mapped into Wishbone space.  
- **DMA Engine:** Supports burst transfers between main memory and local SRAM.  
- **Local Buffers:** On-chip SRAM for staging data close to compute units.  
- **Configurable Bus Wrappers:** Support for Wishbone, AXI, or AHB.  

---

## Example Register Map (MMRs)

Microwatt is a **64-bit core**, but the Parallix datapath is **32-bit**. All addresses are 64-bit aligned, while data fields within registers are 32-bit wide.  

| Address Offset | Register     | Description |
|----------------|-------------|-------------|
| `0x00`         | **CTRL**    | Control register (bits:  START,  RESET, precision mode, etc.) |
| `0x08`         | **STATUS**  | Status flags (bits:  BUSY, DONE, ERROR) |
| `0x10`         | **DMA_SRC** | Source physical address (64-bit) |
| `0x18`         | **DMA_DST** | Destination physical address (64-bit) |
| `0x20`         | **DMA_LEN** | Transfer length in bytes (32-bit) |
| `0x28`         | **IRQ_EN**  | Interrupt enable/mask (32-bit) |
| `0x30`         | **IRQ_STAT**| Interrupt status/acknowledge (32-bit) |

This minimal register map allows the CPU to control Parallix entirely through **MMIO stores/loads**, preserving simplicity and compatibility.

---

## Software Library API

Parallix is controlled entirely through memory-mapped registers, but to simplify development we provide a lightweight **C API**. This library abstracts the register interface into easy-to-use functions for AI, graphics-like transformations, and general-purpose compute.

> **Note on CPU–Parallix Interaction:**  
> All API calls ultimately translate into simple `store` operations from the Microwatt CPU into the Parallix control registers (MMRs).  
> This keeps the programming model lightweight and avoids any changes to the Microwatt compiler or toolchain.

### High-Level vs. Low-Level Access

- **High-Level Functions** (e.g., `parallix_qmatmul`, `parallix_transform_vertices`) are convenience wrappers.  
  Internally, they expand into a series of primitive operations (ADD, MUL, FMA, etc.) expressed through the control registers.  
  These cover the most common AI and graphics use cases.

- **Low-Level Function** (`parallix_array_op`) exposes direct control.  
  Programmers can select an operation (logic, arithmetic, floating-point) and apply it element-wise across arrays.  
  This ensures that *any* operation supported by the ALUs can be accelerated, even if it is not pre-wrapped in a high-level API call.

This dual-level approach ensures both **ease of use** and **maximum flexibility**.

### Kernel Execution Model

Parallix executes **one kernel at a time**.  
- A kernel launch (via API call) occupies Parallix until completion.  
- The CPU can poll the status register or block until the kernel finishes.  
- Parallelism comes from multi-warp SIMD execution inside the kernel, not from running multiple kernels concurrently.  



```c
// =============================
// AI / Machine Learning Primitives
// =============================

// Quantized Matrix Multiply (INT8 → INT32)
// Internally: uses INT8 multiply + accumulate ops.
// Example: Neural network fully-connected layers.
parallix_qmatmul(int8_t *matrix_a, int8_t *matrix_b, int32_t *result, int dim);


// =============================
// Graphics-like Functions
// =============================

// Apply a transformation matrix to a vertex array.
// Internally: sequences of FP32 multiply-add ops.
// Example: 3D-to-2D coordinate transforms, embedded visualization.
parallix_transform_vertices(float *vertex_array,
                            float *transform_matrix,
                            float *output_array,
                            int num_vertices);


// =============================
// General Compute (Low-Level Access)
// =============================

// Element-wise operations on arrays.
// Supported ops map directly to ALU primitives:
// add, sub, mul, min, max, compare, logic.
parallix_array_op(float *array_a,
                  float *array_b,
                  float *result,
                  int len,
                  enum operation_t op);


// =============================
// Configuration and Control
// =============================

parallix_init(void *base_address);          // Initialize with MMR base
parallix_set_mode(uint32_t precision_mode); // INT8 / FP16 / FP32
parallix_get_status();                      // BUSY / DONE / ERROR
```
### Design Notes

- **Consistency:** All functions follow a *“arrays in, arrays out”* model, making the accelerator easy to integrate with existing C code.  
- **Flexibility:** Both AI and graphics functions share the same underlying SIMD execution resources.  
- **Future-proofing:** The API is extensible — new operations (e.g., convolution, activation functions, rasterization steps) can be added as additional high-level wrappers, while the low-level `parallix_array_op` ensures backward compatibility.  

---

## Target Performance Goals

Parallix is designed to provide **substantial acceleration** for parallel workloads that would otherwise execute sequentially on the scalar Microwatt core.  

- **Parallel throughput:** Leverage multiple SIMD engines to process many data elements per cycle versus scalar execution on Microwatt.  
- **AI acceleration:** Efficient support for low-precision (INT8/FP16) arithmetic makes Parallix well-suited for inference tasks.  
- **Unified compute resources:** The same SIMD/ALU units handle both AI workloads (e.g., matrix multiplies) and graphics-style tasks (e.g., vertex transformations), since both reduce to parallel arithmetic operations.  
- **Scalable design:** SIMD engine count and warp size are parameterized, allowing tuning to area/power constraints (e.g., OpenFrame SoC’s ~15 mm² budget).  

---

## Future Extensions

Potential longer-term enhancements that extend beyond the hackathon scope:

- **Tensor Cores / Dot-Product Units:** Specialized INT8/FP16 matrix-multiply hardware for deep learning acceleration.  
- **Texture & Raster Units:** Hardware-accelerated graphics pipeline extensions (sampling, shading, rasterization).  
- **Shared Memory per SIMD Engine:** On-chip scratchpad for fast intra-warp communication and reduction operations.  
- **Dynamic Warp Scheduling:** Hardware-based scheduling to maximize ALU utilization under divergent workloads.  
- **Multi-core Scaling:** Replicating Parallix blocks for heterogeneous SoC clusters.  

---
## Note to Judges

**Short version:** This proposal describes the goals, intended architecture, and target deliverables for **Parallix**. All numerical sizing, microarchitecture choices, and feature lists in the document are **design intents**, not fixed commitments. Final implementation details will be chosen to match real constraints (available silicon area, timing/complexity, toolchain effort, and the hackathon timeframe).

**What this means in practice**
- The register map, SIMD engine count, warp size, and instruction/feature set are *parameterized*. They may be adjusted (scaled up or down) to fit the OpenFrame SoC area budget (~15 mm²) and the implementation schedule.  
- We will prioritize producing a **working, well-tested baseline** (RTL + Wishbone integration, DMA, a small runtime, and at least one demo) over adding every optional feature. Additional hardware features are listed as *future extensions* and may be deferred to later revisions.  
- Software components (runtime, packing/quantization tools, and demos) are part of the deliverable plan — but the exact set of demo models or optimization levels may change based on progress.

**Planned minimum deliverables for the hackathon**
1. RTL for the Parallix datapath with a Wishbone interface and a minimal register set.  
2. DMA + local SRAM staging and a simple kernel launch flow (MMR-based).  
3. A functional microkernel or a close prototype that demonstrates correct acceleration behavior.    
4. Documentation.

**Why we call this out**
- Early estimates (area, timing, verification effort) can change during RTL development and synthesis on target processes. We want judges to evaluate the idea, architecture, and feasibility rather than expect exact, immutable silicon counts or feature lists on day one.

Thank you for reviewing Parallix — we made the proposal to be realistic, implementable, and extensible, and we welcome feedback.

---

*This repository was initialized from the [OpenFrame template](https://github.com/chipfoundry/openframe) to maintain compatibility with the OpenFrame padframe and integration flow.*
