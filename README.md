# Domain Adapted Fine-tuning of an LLM 

A complete, end-to-end pipeline for converting a LaTeX physics PhD thesis into a QLoRA fine-tuned language model. Built as a practical demonstration of domain-adapted LLM training on constrained hardware.

**Base model**: `Qwen2.5 3B Instruct` [(find it here)](https://huggingface.co/anantshri1/qwen2.5-3b-mirror-symmetry)  |  **Hardware**: Google Colab T4 (free tier)  |  Dataset: 2,224 instruction pairs  |  Domain: 3d mirror symmetry, N=2 supersymmetric Chern-Simons-Matter theories

---
## Motivation
My PhD thesis (*The ABCDs of Mirror Symmetry*, SISSA, expected Sept. 2026) is dense with custom LaTeX macros, specialised notation, and physics reasoning that general-purpose LLMs handle poorly. This project builds a pipeline to convert that thesis into structured training data and fine-tune a small language model on it — entirely on free hardware.

### Pipeline

```
Raw .tex files
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1 · LaTeX Parser             │
│                                     │
│  macro_expander.py                  │
│  block_extractor.py                 │
│  latex_parser.py                    │
│                                     │
│  Recursive descent parser —         │
│  not regex. Handles nested          │
│  environments, multi-file           │
│  \input resolution, custom          │
│  macro expansion.                   │
│                                     │
│  Output: block txt per chapter     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 2 · Instruction Generator    │
│                                     │
│  template_generator.py              │
│  gemini_generator.py                │
│                                     │
│  Hybrid strategy:                   │
│  · equations/tables → deterministic │
│    templates + Gemini Flash Lite    │
│  · prose/headers → Gemini Flash     │
│    Lite with context window         │
│                                     │
│  Output: (instruction, output) txt  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 3 · JSONL Assembler          │
│                                     │
│  jsonl_assembler.py                 │
│                                     │
│  Dedup · length filter · shuffle    │
│  90/10 train/val split              │
│                                     │
│  Output: train.jsonl / val.jsonl    │
│          2,224 ChatML examples      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 4 · QLoRA Fine-Tuning        │
│                                     │
│  finetune_colab.ipynb               │
│                                     │
│  Unsloth + HuggingFace TRL          │
│  4-bit quantisation (bitsandbytes)  │
│  LoRA rank 16, all linear layers    │
│  3 epochs · ~35 min on T4           │
│                                     │
│  Output: LoRA adapters + merged     │
│          model on HuggingFace Hub   │
└─────────────────────────────────────┘
```

### Block Schema
The unit of data flowing between stages is a block — a typed, context-annotated segment of the thesis:
```
{
  "block_id": "ch2_s3_2_block_007",
  "chapter": 2,
  "section": "3.2",
  "section_title": "The Mirror Symmetry Map",
  "block_type": "text | equation | table | caption | section_header",
  "content": "...",
  "raw_latex": "...",
  "preceding_context": "...",
  "following_context": "..."
}
```

`content` is macro-expanded and human-readable. `raw_latex` preserves the original verbatim. Context fields are passed to Gemini to ground question generation.

### Training details

| Parameter | Value |
|---|---|
| Base model | `Qwen2.5-3B-Instruct` |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down |
| Batch size (effective) | 8 (2 × 4 grad accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 3 |
| Max sequence length | 2048 |
| Hardware | Google Colab T4 (16GB) |
| Peak VRAM | ~8GB |
| Training time | ~35 minutes |
| Final val loss | 1.033 |

Open `notebooks/finetune_colab.ipynb` in Google Colab and run all cells.

---

## Dataset

The training dataset is not included in this repo as it is derived from an unpublished PhD thesis. The pipeline code is fully general and can be applied to any LaTeX document corpus.

---

## Limitations

- **Model size**: A quantized 3B parameter model has limited capacity for deep physics reasoning in a specialised domain. The fine-tuned model handles thesis-specific terminology and phrasing better than the base model, but should not be used as a physics reference.
- **Dataset size**: 2,224 examples is small for domain adaptation. Accuracy improves significantly with a larger or more diverse instruction set.
- **Hardware**: Training was constrained to free Colab T4. A larger base model (7B+) would likely yield meaningfully better results.
- **Equation hallucination**: Gemini Flash Lite tends to override provided context for common/generic equations (e.g. free scalar kinetic terms, standard gauge kinetic terms), generating physically plausible but thesis-inaccurate outputs. Thesis-specific equations are grounded correctly.
- **Gemini API rate limits**: The free tier (15 req/min) is insufficient for a full thesis run. A paid API key (~$0.05 total for this corpus) **is** strongly recommended.
- Gemini often leaves the template pairs for equations/tables empty.
- The length filter is character-based (not tokenizer-based) pending final model selection
---

## Future work

- **v2 — GraphRAG**: Integrating the fine-tuned LLM over legacy [GraphRAG project](https://github.com/anantshri1/GraphRAG_for_3dQFT/tree/main) as the LLM backend for graph-based retrieval-augmented generation. More faithful to the source material than pure fine-tuning.
- Larger base model experiment (Qwen2.5 7B on Colab Pro)
- Automatic evaluation with BERTScore against gold set

---

## Stack

`Python` · `Unsloth` · `HuggingFace TRL / PEFT / Transformers` · `bitsandbytes` · `Google Gemini Flash Lite` · `Google Colab`

