# 🩺 Medical QLoRA Fine-Tuning — Qwen3-8B

A parameter-efficient fine-tuning project that adapts **Qwen3-8B** into a focused, on-domain **medical assistant** using **QLoRA** (4-bit quantization + LoRA), accelerated end-to-end with **Unsloth**. The result is a lightweight, portable LoRA adapter that transforms a general-purpose 8B model into a model that answers medical questions with a measurably sharper, more consistent, textbook-style voice — trained and evaluated entirely on free-tier GPU hardware.

---

## ✨ Highlights

- **Base model:** [`unsloth/Qwen3-8B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit) — Unsloth's pre-quantized 4-bit NF4 checkpoint of Qwen3-8B
- **Method:** QLoRA (4-bit frozen base + trainable LoRA adapters on attention projections)
- **Trainable footprint:** only **15.3M / 8.2B parameters (0.187%)** were trained
- **Dataset:** [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) — 33,275 medical instruction/answer pairs
- **93.1% reduction in perplexity** versus the untuned base model on held-out medical data
- Trained and validated entirely on a single **Tesla T4 (Kaggle free tier)**
- Ships as a **~60 MB adapter**, not a multi-gigabyte full model — cheap to store, share, and deploy
- Full **training**, **evaluation**, and **inference** pipelines included as ready-to-run notebooks

---

## 📁 Repository Structure

```
.
├── training-notebook/
│   └── medical-fine-tuning.ipynb      # End-to-end QLoRA training + evaluation pipeline
├── Inference-notebook/
│   └── model-inference-notebook.ipynb # Standalone inference notebook (loads adapter only)
├── medical_adapter/                   # The trained LoRA adapter, ready to deploy
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
│   ├── special_tokens_map.json / added_tokens.json / chat_template.jinja
│   └── README.md                      # Hugging Face-style model card
└── graphs/
    ├── Loss Graph.jpeg                # Training / eval loss curve
    ├── Perpelexity comparison.jpeg    # Base vs fine-tuned perplexity
    └── Evaluation comparison.jpeg     # Base vs fine-tuned token-level metrics
```

---

## 🧠 Why Qwen3-8B?

**Qwen3-8B** was chosen as the base model because it strikes a strong balance for this project's constraints:

- Large enough (8B parameters) to already carry broad world and medical knowledge from pretraining, so fine-tuning only needs to *specialize* tone and domain focus rather than teach facts from scratch.
- Available pre-quantized to 4-bit NF4 via Unsloth (`unsloth/Qwen3-8B-unsloth-bnb-4bit`), which removes the need to quantize a full-precision checkpoint at load time — faster startup and a lower memory footprint on a single T4 GPU.
- Fully compatible with Unsloth's fused Triton kernels (RoPE, RMSNorm, cross-entropy, attention), which materially speed up both training and inference without changing the underlying math.
- Ships with a native chat template (`apply_chat_template`), so instruction data could be formatted correctly without hand-rolling prompt formats.

---

## 🏋️ Training Notebook (`training-notebook/medical-fine-tuning.ipynb`)

An end-to-end, well-documented QLoRA pipeline that takes the project from raw dataset to a saved, evaluated adapter. It runs top to bottom on Kaggle's free T4 GPU tier.

### Pipeline

1. **Install dependencies** — Unsloth pulls in matched, tested versions of `transformers`, `trl`, `peft`, `accelerate`, and `bitsandbytes`.
2. **Load the 4-bit base model** via `FastLanguageModel.from_pretrained`, with Unsloth's Triton kernel patches applied.
3. **Apply LoRA adapters** via `FastLanguageModel.get_peft_model`:
   - Rank `r = 16`, `alpha = 32`, `dropout = 0.05`, `bias = "none"`
   - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Gradient checkpointing via Unsloth's own memory-efficient implementation
4. **Load and split the dataset** — 33,275 train / 680 validation examples from the Medical Meadow Flashcards dataset.
5. **Format with the Qwen3 chat template** — each `instruction`/`input`/`output` row is converted into a proper system/user/assistant chat turn.
6. **Tokenize** with truncation at 256 tokens (mean example length: 125.5 tokens).
7. **Configure training** with `SFTConfig`:
   - 2 epochs, cosine LR schedule, `paged_adamw_8bit` optimizer
   - Effective batch size 16 (`per_device=2` × `grad_accum=8`)
   - Sequence packing enabled for >2x training throughput
8. **Train** with `SFTTrainer` — automatically resumes from the latest checkpoint if interrupted.
9. **Save the adapter** — only the LoRA weights + tokenizer are persisted (a few tens of MB, not the 8B base model).
10. **Optional Kaggle Hub upload** for pulling the adapter into other notebooks.
11. **Reload and sanity-check** the adapter from disk, exactly as a fresh inference session would.
12. **Quantitative + qualitative evaluation** of the fine-tuned model against the untouched base model (see below).
13. **Optional 16-bit merge** — bakes the adapter into a standalone, deployable full model.

### Training Configuration

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 15,335,424 / 8,206,070,784 (**0.187%**) |
| Epochs | 2 |
| Effective batch size | 16 |
| Learning rate | 2e-4 (cosine schedule, 50-step warmup) |
| Max sequence length | 256 tokens |
| Optimizer | `paged_adamw_8bit` |
| Hardware | 1× Tesla T4 (14.6 GB VRAM) |
| Training time | ~7h 55m (2,158 steps) |

---

## 📊 Results

Fine-tuning delivered a large, consistent improvement over the base model across every metric measured on the held-out medical validation set.

### Training Loss

Loss dropped smoothly and consistently across both epochs, from **1.686 → 0.588** (train) and to **0.613** (eval) — clean convergence with no sign of divergence or instability.

![Training Loss](graphs/Loss%20Graph.jpeg)

### Perplexity — Base vs. Fine-Tuned

| Model | Perplexity |
|---|---:|
| Base (Qwen3-8B, no adapter) | 26.57 |
| **Fine-tuned (with medical adapter)** | **1.83** |

**→ A 93.1% reduction in perplexity.** The fine-tuned model is dramatically more confident and accurate at predicting the next token in genuine medical answers — the clearest, most direct evidence that QLoRA training successfully specialized the model to the medical domain.

![Perplexity Comparison](graphs/Perpelexity%20comparison.jpeg)

### Token-Level Classification Metrics

To complement perplexity, next-token prediction was bucketed into the top-50 most frequent answer tokens plus an `OTHER` class, turning it into a tractable (K+1)-way classification problem:

| Metric | Base model | Fine-tuned | Δ |
|---|---:|---:|---:|
| Token F1 (macro) | 0.549 | **0.875** | +0.326 |
| Token Recall (macro) | 0.555 | **0.871** | +0.316 |

Both metrics improved by roughly 60%, reinforcing that the fine-tuned model isn't just less "surprised" by medical text (lower perplexity) — it's also substantially more accurate at picking the correct next token among the domain's most common vocabulary.

![Token Metrics Comparison](graphs/Evaluation%20comparison.jpeg)

### Qualitative Behavior Change

Beyond the numbers, fine-tuning visibly changed *how* the model answers. The base model tends to reason at length in a visible `<think>` block before answering (verbose, exploratory, sometimes second-guessing itself mid-thought). The fine-tuned model skips extended reasoning and responds directly, in a **concise, textbook-flashcard style** that closely mirrors the training data — e.g., for "What precautions should a patient take after knee replacement surgery?" the fine-tuned model responds with a single, direct, clinically-grounded sentence rather than a long exploratory answer. This is exactly the kind of stylistic and tonal specialization QLoRA fine-tuning is meant to produce.

---

## 🤖 Inference Notebook (`Inference-notebook/model-inference-notebook.ipynb`)

A standalone, **training-free** notebook for using the adapter on its own — designed to run on Kaggle with the adapter attached as a dataset. It is intentionally decoupled from the training notebook so the adapter can be reused, shared, or swapped for any other LoRA adapter (medical, legal, finance, etc.) by changing a single configuration cell.

### What it does

1. Installs the same pinned Unsloth/`transformers`/`trl` stack used in training.
2. Auto-detects the adapter directory anywhere under `/kaggle/input` (or accepts an explicit path).
3. Loads the 4-bit base model + adapter in a single `FastLanguageModel.from_pretrained` call.
4. Switches into Unsloth's inference-optimized mode (`FastLanguageModel.for_inference`) for up to 2x faster generation.
5. Provides:
   - A reusable `generate_response()` chat function supporting multi-turn conversation history
   - A single-prompt example
   - A multi-turn conversation example
   - An optional interactive chat loop
   - Batch inference over a prompt list, saved to JSON
6. Includes a troubleshooting section covering the most common failure modes (missing adapter, base-model mismatch, OOM, mismatched system prompt).

### Sample output

> **Q:** How to recover from diabetes type 2?
>
> **A:** Diabetes type 2 is a chronic condition that can be managed and even reversed in some cases through lifestyle changes. One of the most effective ways to manage diabetes type 2 is through weight loss, which can help to improve insulin sensitivity and reduce blood sugar levels. In addition to weight loss, other lifestyle changes that may be helpful include regular exercise, a healthy diet low in refined sugars and processed foods, and stress management techniques such as meditation or yoga. It is important to work closely with a healthcare provider to develop a personalized treatment plan...

---

## 🚀 How to Use This Adapter

The `medical_adapter/` directory is a complete, self-contained PEFT/LoRA adapter — everything needed to attach it to the base model is included (adapter weights, tokenizer, chat template).

### Option 1 — Quick start with the provided inference notebook (recommended)

1. Zip the `medical_adapter/` folder.
2. Upload it as a **Kaggle Dataset** (Datasets → New Dataset).
3. Open `Inference-notebook/model-inference-notebook.ipynb` in Kaggle, attach your dataset via **+ Add Input**, set a GPU accelerator, and run all cells.

### Option 2 — Load it yourself with Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/medical_adapter",   # local path or HF/Kaggle dataset path
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant. User will ask medical related questions and you will be answering them in a helpful manner"},
    {"role": "user", "content": "What are the common symptoms of type 2 diabetes?"},
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

### Option 3 — Standard PEFT / Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "path/to/medical_adapter")
tokenizer = AutoTokenizer.from_pretrained("path/to/medical_adapter")
```

> **Important:** always pair this adapter with its matching base model, `unsloth/Qwen3-8B-unsloth-bnb-4bit`, and keep the system prompt close to the one used in training/inference above — a mismatched system prompt is the most common cause of the adapter appearing "not to work."

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Base model | Qwen3-8B (4-bit NF4, Unsloth checkpoint) |
| Fine-tuning method | QLoRA (4-bit + LoRA) |
| Training acceleration | [Unsloth](https://github.com/unslothai/unsloth) (fused Triton kernels) |
| Training loop | 🤗 `trl` `SFTTrainer` / `SFTConfig` |
| PEFT backend | 🤗 `peft` (LoRA) |
| Quantization | `bitsandbytes` (4-bit NF4) |
| Dataset handling | 🤗 `datasets` |
| Evaluation | `scikit-learn` (F1 / recall / ROC-AUC), custom perplexity computation |
| Visualization | `matplotlib` |
| Compute | Kaggle free-tier Tesla T4 GPU |

---

## 📌 Notes & Limitations

- This adapter is a **research/educational project** demonstrating parameter-efficient fine-tuning for a domain-specialized assistant — it is **not a certified medical device** and its outputs should not be used as a substitute for professional medical advice.
- Evaluation was run on a held-out sample from the same dataset distribution; broader generalization to unseen medical question styles has not been separately validated.
- The `medical_adapter/README.md` model card is the standard Hugging Face template scaffold and can be filled in further if the adapter is published to the Hub.
