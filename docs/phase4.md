# 📘 Phase 4: 4-Bit QLoRA Fine-Tuning Pipeline with Unsloth & PEFT

---

## 🎯 1. Overview & Objective

Base foundational Small Language Models (SLMs) like `Llama-3.2-1B` or `Qwen-2.5-1.5B` have strong general conversational ability, but they often struggle with **domain-specific compliance rules, proprietary JSON schemas, and nuanced risk classification policies**.

Full parameter fine-tuning of an SLM requires storing model weights, gradients, and optimizer states in 32-bit/16-bit precision, consuming over $16\text{GB}–32\text{GB}$ of VRAM and making it impossible to train on a standard developer machine.

**Phase 4 Goal:** Implement a complete **4-bit Quantized Low-Rank Adaptation (QLoRA)** fine-tuning pipeline using **Unsloth & Hugging Face PEFT** to:
- Freeze the base model in **4-bit NormalFloat (NF4)** precision (~1.2GB VRAM).
- Inject trainable low-rank adapter matrices ($r=16, \alpha=32$) representing $<1.0\%$ of total parameters.
- Fine-tune domain instruction datasets in **under 2 hours on commodity hardware**.
- Export trained adapters into **GGUF** (for Ollama / Apple Silicon Metal) and **AWQ** (for vLLM continuous batching).

---

## 📐 2. Mathematical Foundation: LoRA & QLoRA

### A. Low-Rank Matrix Decomposition (LoRA)
During standard fine-tuning, weight matrix updates are represented as $\Delta W \in \mathbb{R}^{d \times k}$. 

LoRA hypothesizes that weight updates have a **low "intrinsic rank" $r \ll \min(d, k)$**. It decomposes $\Delta W$ into two low-rank matrices $A$ and $B$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$ is the **frozen** pretrained base weight matrix.
- $B \in \mathbb{R}^{d \times r}$ initialized to **zeros** ($0$).
- $A \in \mathbb{R}^{r \times k}$ initialized from a **Gaussian distribution** $\mathcal{N}(0, \sigma^2)$.
- $\alpha$ is a constant scaling hyperparameter (typically $\alpha = 2r = 32$).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LoRA FORWARD PASS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Input Activation: x                                 │
│                                │                                            │
│               ┌────────────────┴────────────────┐                           │
│               ▼                                 ▼                           │
│     [ Frozen Base Weight: W0 ]        [ Down-Projection: A (d -> r) ]       │
│               │                                 │                           │
│               │                       [ Up-Projection: B (r -> k) ]         │
│               │                                 │                           │
│               │                       [ Scaling Factor: alpha / r ]         │
│               │                                 │                           │
│               └────────────────┬────────────────┘                           │
│                                ▼                                            │
│                     Output: h = x*W0 + x*(B*A)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. 4-Bit NormalFloat (NF4) & Double Quantization
QLoRA introduces three innovations to compress base weights:
1. **NF4 Quantization:** Information-theoretically optimal quantile quantization for normally distributed weights.
2. **Double Quantization (DQ):** Quantizes the quantization constants themselves, saving an extra $0.37\text{ bits per parameter}$.
3. **Paged Optimizers:** Uses memory-mapped OS paging to prevent GPU out-of-memory spikes during long gradient checkpointing steps.

### C. Parameter Efficiency on Commodity & Edge Hardware:
- **Base Model (Llama-3.2-1B):** $1,235,814,400$ total parameters (frozen in 4-bit $\approx 1.2\text{ GB}$).
- **Trainable LoRA Adapters ($r=16$):** $11,534,336$ parameters (**only $0.93\%$ of the model!**).
- **Peak Training Memory:** $\approx 1.4\text{ GB VRAM}$, easily training on edge & commodity servers.

---

## 🛠️ 3. Step-by-Step Code Walkthrough

### Step 1: Instruction Dataset Formatter (`src/fine_tuning/dataset.py`)
- Formats domain instruction-input-output triplets into the standard **Llama 3.2 Chat template** (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).

### Step 2: 4-Bit QLoRA Trainer (`src/fine_tuning/train_qlora.py`)
- Configures target attention and MLP projection modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Sets rank hyperparameters ($r=16, \alpha=32$, `dropout=0.0`, `lr=2e-4`).
- Provides a standalone training simulation computing loss curves for cross-platform verification.

### Step 3: Deployment Exporter (`src/fine_tuning/export.py`)
- **GGUF (`q4_k_m`):** Quantized format for running inside Ollama / llama.cpp on Apple Silicon Metal.
- **AWQ (4-bit):** Activation-aware weight quantization for high-throughput continuous batching in vLLM.
- Generates automated **Ollama Modelfile** definitions linking base models with trained LoRA adapters.

---

## 🧪 4. How to Run & Verify Phase 4

### Command:
```bash
python3 tests/test_fine_tuning.py
```

### Expected Output:
```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK
```

### What the Tests Verify:
1. `test_dataset_formatting`: Asserts ChatML and Llama 3.2 template tags match tokenization standards.
2. `test_trainer_config_and_parameters`: Proves trainable parameters are $<1.0\%$ of total base parameters.
3. `test_simulated_training_loop`: Validates loss convergence over training epochs.
4. `test_export_modelfile_generation`: Verifies Ollama Modelfile syntax.

---

## 💡 5. Technical Questions & Architectural Explanations

### Q: Why apply LoRA to all linear layers instead of just query and value projections (q_proj, v_proj)?
> **Answer:** Early LoRA papers only adapted attention query and value matrices (`q_proj`, `v_proj`). Recent empirical benchmarks demonstrate that adapting **all linear layers** (including attention output `o_proj`, key `k_proj`, and MLP gates `gate_proj`, `up_proj`, `down_proj`) with a smaller rank (e.g. $r=16$) yields significantly higher downstream reasoning accuracy and domain recall than using a high rank ($r=64$) on attention layers alone, with negligible extra VRAM overhead.
