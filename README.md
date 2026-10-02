# Benchmarking llama.cpp Inference on a Raspberry Pi 4

Measuring LLM inference throughput on a Raspberry Pi 4 Model B (4GB) across
model size, quantization level, and thread count, to identify the dominant
hardware constraint and characterize the speed/quality tradeoff.

**Status: in progress.** The results below are preliminary measurements taken
without thermal control between configurations. The full controlled sweep is
being built. See *Limitations* before citing any number here.

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
memory traffic per token, and some loss of output quality. The K-quants
(Q4_K_M, Q5_K_M, Q6_K) use a superblock structure with multiple levels of
scaling; Q8_0 uses a simpler block format. That difference turns out to matter
— see *Quantization format affects prompt processing*.

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
   **Confirmed — see Results.**
2. **Heavier quantization saturates bandwidth at fewer threads.** More bytes per
   weight means the memory ceiling is reached with fewer cores — an interaction
   effect between quantization and thread count, visible only in the full grid.
   *Untested: requires the full grid.*
3. **tg throughput scales inversely with file size**, since each token requires
   reading the whole model. **Confirmed — see Results.**
4. **OpenBLAS improves pp but not tg.** BLAS kernels optimize matrix-matrix
   arithmetic (GEMM), which is pp's bottleneck. Generation is matrix-vector
   (GEMV) and limited by memory transfer, which no arithmetic optimization
   addresses. Prior estimate: 20-50% on pp, roughly flat on tg.
   *Untested: OpenBLAS build not yet made.*
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
recoverable from the raw output. Raw JSON is archived; no summary statistic is
computed destructively. llama-bench does not label the test type, so it is
derived: `n_gen == 0` indicates prompt processing, otherwise generation.

**Thermal control (full sweep, not yet applied):** wait for CPU below 50 C
before each configuration, capped at 180 s. If the cap is reached, the
configuration proceeds and the row is flagged rather than silently violating the
control. Start and end temperature are recorded per row, along with the
board's throttle status.

## Results

All figures are medians of three repetitions.

### Thread scaling: pp parallelizes, tg does not (hypothesis 1)

TinyLlama 1.1B Q4_K_M, 1 to 4 threads:

| Threads | pp512 t/s | tg128 t/s | Implied GiB/s during tg |
|---|---|---|---|
| 1 | 3.24 | 2.79 | 1.74 |
| 2 | 6.44 | 5.47 | 3.40 |
| 3 | 9.51 | 5.82 | 3.62 |
| 4 | 12.49 | 5.55 | 3.45 |

**Prompt processing scales 3.86x across four cores** — 1.99x, 2.93x, 3.86x
relative to single-threaded. Near-linear, as expected for compute-bound work.

**Token generation scales 1.99x and then stops.** The jump from one to two
threads is a clean doubling. The third thread adds 6%. The fourth makes it
*slower than three* — 5.55 against 5.82, a 4.6% regression from contention over
a bus that is already saturated.

Two cores are sufficient to saturate memory bandwidth on this platform. The
implied throughput column shows it directly: 1.74 GiB/s at one thread, then flat
between 3.40 and 3.62 for every thread count above one.

### tg throughput is predicted by file size, not by model (hypothesis 3)

Eleven models at 4 threads, sorted by file size:

| Model | Size (GiB) | tg128 t/s | Implied GiB/s |
|---|---|---|---|
| TinyLlama 1.1B Q4_K_M | 0.62 | 5.54 | 3.44 |
| TinyLlama 1.1B Q5_K_M | 0.73 | 4.95 | 3.61 |
| TinyLlama 1.1B Q6_K | 0.84 | 4.41 | 3.71 |
| Qwen2.5 1.5B Q4_K_M | 1.04 | 3.76 | 3.90 |
| TinyLlama 1.1B Q8_0 | 1.09 | 3.27 | 3.56 |
| Qwen2.5 1.5B Q5_K_M | 1.19 | 2.98 | 3.55 |
| Qwen2.5 1.5B Q6_K | 1.36 | 2.94 | 4.00 |
| Qwen2.5 1.5B Q8_0 | 1.76 | 2.23 | 3.93 |
| Llama 3.2 3B Q4_K_M | 1.88 | 1.84 | 3.45 |
| Llama 3.2 3B Q5_K_M | 2.15 | 1.51 | 3.25 |
| Llama 3.2 3B Q6_K | 2.46 | 1.43 | 3.52 |

Throughput varies 3.9x across this set. Implied memory bandwidth varies 1.23x,
with a mean of 3.63 GiB/s and a coefficient of variation of 6.3%.

This yields a predictive rule for the platform:

tg throughput (t/s) ~= 3.6 / model size (GiB)


Tested against a model outside this sweep: Llama 3.2 3B Q8_0 at 3.18 GiB
predicts 1.13 t/s. Measured 1.10 t/s — 2.7% error.

### Quantization format affects prompt processing, but not generation

An unpredicted result. Within every family, Q8_0 outperforms both Q5_K_M and
Q6_K at prompt processing, despite being the largest file:

| Family | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|
| TinyLlama 1.1B | 12.41 | 10.01 | 9.43 | **11.59** |
| Qwen2.5 1.5B | 9.05 | 7.21 | 6.80 | **8.60** |
| Llama 3.2 3B | 4.27 | 3.43 | 3.15 | **4.23** |

pp512 t/s at 4 threads. Llama 3.2 3B Q8_0 measured separately.

The K-quants pack weights into superblocks with nested scaling factors, so
dequantizing them before arithmetic costs more CPU work than Q8_0's simpler
block format. Prompt processing is compute-bound, so that cost appears directly
in throughput.

The effect disappears for generation. The Q8_0 bandwidth figures above (3.56,
3.93) sit within the normal spread. When the CPU is stalled waiting on memory,
dequantization cost is absorbed into the stall.

Same models, same hardware, opposite behavior in the two phases — because one
is limited by arithmetic and the other is not.

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
3.18 GiB x 1.10 t/s = 3.50 GiB/s, consistent with the ceiling measured above.
Sustained reads from the microSD card would be roughly two orders of magnitude
slower.

Headroom is thin. This result holds at a 512-token prompt and 128-token
generation; a longer context would grow the KV cache and may not fit.

### Measurement stability

TinyLlama 1.1B Q4_K_M at 4 threads, measured four times across three sessions
and a reboot:

| Measurement | pp512 t/s | tg128 t/s |
|---|---|---|
| Pilot, session 1 | 12.51 | 5.66 |
| Pilot, session 2 | 12.55 | 5.69 |
| Thread sweep | 12.49 | 5.55 |
| Quantization sweep | 12.41 | 5.54 |

Spread: 1.1% for pp, 2.7% for tg. The two lowest tg values come from the two
long unattended runs, which is consistent with thermal accumulation. This gives
an approximate magnitude for the no-cooldown confound: on the order of 2-3% for
generation.

Throttle status read `0x0` after the full two-hour sweep, so CPU frequency
capping is ruled out as a confound. Thermal accumulation and thermal throttling
are distinct; only the former is present here.

## Limitations

**These are preliminary measurements**, collected in a single session to produce
honest numbers ahead of the controlled sweep.

- **No thermal cooldown between configurations.** Later configurations in a run
  started from a hotter board than earlier ones, confounding load with thermal
  state. The designed control (50 C threshold, 180 s cap, flagged timeouts) was
  deliberately skipped for speed. Estimated magnitude from the stability table:
  roughly 2-3% for generation.
- **Partial grid.** Thread scaling was measured on one model; the quantization
  sweep was measured at one thread count. The 48-configuration grid has not been
  run, so hypothesis 2 is untested.
- **Single build variant.** CPU-only. No OpenBLAS comparison exists yet, so
  hypothesis 4 is untested.
- **Quality not yet evaluated.** A 10-prompt quality set with binary scoring
  criteria is designed but not run. No claim is made here about output quality
  at any quantization level.
- **Power supply.** The board runs on a 5.0V/3A supply rather than the official
  5.1V. Throttle status reads `0x0` after sustained load, and will be logged per
  configuration in the full sweep.

The controlled sweep supersedes everything in Results.

## Repository layout

configs.py generates the 48-configuration matrix
show.py summarizes raw JSON into readable tables
data/raw/ raw llama-bench JSON output
tests/fixtures/ saved output used by unit tests


## Reproducing

Requires a Raspberry Pi 4 (4GB), llama.cpp built per the table above, and the
GGUF model files from their respective Hugging Face repositories. Model weights
are not redistributed here.

python3 show.py
