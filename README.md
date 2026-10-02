# Benchmarking llama.cpp Inference on a Raspberry Pi 4

Measuring LLM inference throughput on a Raspberry Pi 4 Model B (4GB) across
model size, quantization level, and thread count, to identify the dominant
hardware constraint and characterize the speed/quality tradeoff.

**Status: in progress.** The results below are preliminary pilot measurements
taken without thermal control. The full controlled sweep is being built. See
*Limitations* before citing any number here.

## Hardware and software

| Item | Detail |
|---|---|
| Board | Raspberry Pi 4 Model B, 4GB LPDDR4 |
| CPU | Broadcom BCM2711, quad-core Cortex-A72 @ 1.5 GHz (ARMv8.0) |
| OS | Raspberry Pi OS Lite 64-bit (Debian 13), kernel 6.18.50 |
| Swap | Disabled, so memory exhaustion fails cleanly instead of thrashing |
| Inference | llama.cpp, commit 930e2fa59, build b10991, CPU-only |
| Build | `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON` |

The Cortex-A72 predates the ARM extensions that accelerate quantized integer
math. The build detects FMA but reports **no** dotprod, i8mm, SVE, SME, or FP16
vector arithmetic. This is a reportable constraint: results here should not be
extrapolated to newer ARM silicon.

## Models

Three families at four quantization levels each, 12 files total.

| Family | Params | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|---|
| TinyLlama 1.1B Chat | 1.1B | 638M | 747M | 863M | 1.1G |
| Qwen2.5 1.5B Instruct | 1.5B | 1.1G | 1.2G | 1.4G | 1.8G |
| Llama 3.2 3B Instruct | 3.2B | 1.9G | 2.2G | 2.5G | 3.2G |

Quantization reduces the numeric precision of model weights. Q4 stores roughly
4 bits per weight, Q8 roughly 8. Lower precision means a smaller file, less
memory traffic per token, and some loss of output quality.

## The central idea: arithmetic intensity

Inference has two phases with different bottlenecks.

**Prompt processing (pp)** reads the input. All 512 prompt tokens are processed
at once, so the math is matrix times matrix and each weight loaded from memory
is reused across many tokens. High arithmetic intensity: the CPU is doing real
work.

**Token generation (tg)** writes the output, one token at a time, each one
depending on the last. The math is matrix times vector. Generating a single
token requires reading essentially the entire model from RAM, and each weight is
used once. Low arithmetic intensity: the CPU spends most of its time waiting on
memory.

Every hypothesis below follows from this distinction.

An early observation supports it independent of throughput numbers: CPU
temperature ran in the 60s C during prompt processing and *dropped* into the 50s
during generation. A CPU doing arithmetic heats up; a CPU stalled on memory
cools. The bottleneck is visible thermally.

## Hypotheses

Stated before measurement.

1. **tg is memory-bandwidth bound.** Thread scaling should be strong for pp and
   weak for tg, since additional cores cannot make RAM faster.
2. **Heavier quantization saturates bandwidth at fewer threads.** More bytes per
   weight means the memory ceiling is reached with fewer cores — an interaction
   effect between quantization and thread count, visible only in the full grid.
3. **tg throughput scales inversely with file size**, since each token requires
   reading the whole model.
4. **OpenBLAS improves pp but not tg.** BLAS kernels optimize matrix-matrix
   arithmetic (GEMM), which is pp's bottleneck. Generation is matrix-vector
   (GEMV) and limited by memory transfer, which no arithmetic optimization
   addresses. Prior estimate: 20-50% on pp, roughly flat on tg.
5. ~~**Llama 3.2 3B at Q8_0 will exhaust memory on a 4GB board.**~~
   **Falsified — see Results.**

## Experimental design

**Full grid:** 3 models x 4 quantization levels x 4 thread counts = 48
configurations per build variant. The full grid was chosen over a reduced design
specifically to expose the interaction effect in hypothesis 2.

**Per configuration:** `llama-bench -p 512 -n 128 -r 3`. Three repetitions were
chosen after the pilot showed 0.015% relative standard deviation — far below any
effect size of interest.

**Instrumentation:** llama-bench emits structured JSON (`-o json`) including
per-repetition samples, so mean, median, and standard deviation are all
recoverable from the raw output. Raw JSON is archived per configuration; no
summary statistic is computed destructively.

**Thermal control (full sweep, not yet applied):** wait for CPU below 50 C
before each configuration, capped at 180 s. If the cap is reached, the
configuration proceeds and the row is flagged rather than silently violating the
control. Start and end temperature are recorded per row, along with the
board's throttle status.

## Results

### Llama 3.2 3B Q8_0 fits in 4GB (hypothesis 5 falsified)

Predicted to fail: 3.18 GiB of weights against roughly 3.5 GiB available, before
the KV cache. It ran.

| Test | t/s |
|---|---|
| pp512 | 4.23 |
| tg128 | 1.10 |

The prediction assumed llama.cpp would allocate the weights on the heap. It
memory-maps the model file instead. Mapped pages are file-backed, so the kernel
can evict and re-read them from disk at any time. Disabling swap prevents
eviction of *anonymous* memory only; it places no constraint on file-backed
pages. The weights therefore never required a 3.18 GiB allocation, and only the
KV cache and compute buffers — roughly 300 MB — needed real memory.

It was not paging from the SD card. Implied memory throughput is
3.18 GiB x 1.10 t/s = 3.50 GiB/s, consistent with LPDDR4 bandwidth. Sustained
reads from the microSD card would be roughly two orders of magnitude slower.

Headroom is thin. This result holds at a 512-token prompt and 128-token
generation; a longer context would grow the KV cache and may not fit.

### Evidence for the bandwidth ceiling

Derived memory throughput during generation, across a 5x range in model size:

| Model | Size | tg t/s | Implied GiB/s |
|---|---|---|---|
| TinyLlama 1.1B Q4_K_M | 0.636 GiB | 5.69 | 3.62 |
| Llama 3.2 3B Q8_0 | 3.18 GiB | 1.10 | 3.50 |

Throughput fell almost exactly in proportion to size, and the implied bandwidth
agrees within 3.4%. This is consistent with hypothesis 3 and sets an approximate
ceiling of 3.5 GiB/s for the platform.

### Measurement stability

The same configuration measured on two separate days, across a reboot:

| Session | pp512 t/s | tg128 t/s |
|---|---|---|
| First pilot | 12.51 | 5.66 |
| Second pilot | 12.55 | 5.69 |

0.3% and 0.6% apart.

### Thread scaling and quantization sweep

Data collected; analysis pending.

## Limitations

**These are preliminary measurements.** They were collected in a single session
to produce honest numbers for an application deadline, ahead of the controlled
sweep.

- **No thermal cooldown between configurations.** Later configurations in a run
  started from a hotter board than earlier ones, confounding load with thermal
  state. The designed control (50 C threshold, 180 s cap, flagged timeouts) was
  deliberately skipped for speed and is not reflected in any number above.
- **Partial grid.** Thread scaling was measured on one model; the quantization
  sweep was measured at one thread count. The 48-configuration grid has not been
  run.
- **Single build variant.** CPU-only. No OpenBLAS comparison exists yet, so
  hypothesis 4 is untested.
- **Quality not yet evaluated.** A 10-prompt quality set with binary scoring
  criteria is designed but not run. No claim is made here about output quality
  at any quantization level.
- **Power supply.** The board runs on a 5.0V/3A supply rather than the official
  5.1V. Throttle status reads 0x0 after sustained load, and is logged per
  configuration.

The controlled sweep supersedes everything in Results.

## Repository layout

configs.py generates the 48-configuration matrix
data/raw/ raw llama-bench JSON output
tests/fixtures/ saved output used by unit tests


## Reproducing

Requires a Raspberry Pi 4 (4GB), llama.cpp built per the table above, and the
GGUF model files from their respective Hugging Face repositories. Model weights
are not redistributed here.
