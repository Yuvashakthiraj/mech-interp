# Multi-Task Circuit Sharing in Modular Arithmetic Transformers

**Research question:** When a small transformer is trained on several modular arithmetic operations at once
(addition, subtraction, multiplication, and eventually a non-abelian group operation), does it learn
**one shared internal circuit**, separate circuits per operation, or something in between?

This is a final-year undergraduate mechanistic interpretability project extending:
- Power et al. (2022) — first documented "grokking" on algorithmic tasks.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR) — fully reverse-engineers a 1-layer transformer on modular addition; the Fourier-circuit result this project replicates in Phase 0.
- Chughtai et al. (2023), *A Toy Model of Universality* (ICML) — extends circuit analysis to finite group operations.
- Stander et al. (2023), *Grokking Group Multiplication with Cosets* — contradicting follow-up to Chughtai et al.; a reminder that this subfield is still actively contested.

---

## Results so far

### Phase 0 — Single-task baseline: Modular Addition ✅

A 1-layer, 4-head transformer (d_model=128) trained on modular addition mod 113.
Confirmed grokking and the sparse Fourier circuit from prior literature.

| | |
|---|---|
| ![Loss curves](results/loss_curves.png) | ![Fourier spectrum](results/fourier_spectrum.png) |

**Key findings:**
- Train accuracy hit ~100% within the first few hundred epochs; test accuracy stayed near 0% until epoch ~2000–4000, then jumped sharply to ~100% — the grokking phase transition.
- Fourier analysis of the number-token embedding matrix shows a small number of clearly dominant frequency spikes against a near-zero background, matching the "sparse Fourier circuit" signature from Nanda et al. 2023.

---

### Phase 1 — Multi-task: Addition + Subtraction ✅

Same architecture retrained on **both** addition and subtraction simultaneously, disambiguated by an operation token in the input sequence `[a, op_token, b, =]`.

| | |
|---|---|
| ![Grokking curves](results/phase1_grokking_curves.png) | ![Fourier spectrum](results/phase1_fourier_spectrum.png) |
| ![Head ablation](results/phase1_head_ablation.png) | ![Activation patching](results/phase1_activation_patching.png) |

**Key findings:**

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **Grokking** | Both add and sub reach 100% test accuracy at the **same epoch (~6200)** | Strong evidence of a single shared learning event, not two separate ones |
| **Fourier spectrum** | Top frequencies: {9, 18, 28, 47, 49} — 5 non-DC peaks vs ~4 in Phase 0 | Slight expansion but not doubled → circuit is largely shared across tasks |
| **Head ablation** | All 4 heads: 0.000 accuracy drop for both tasks | Attention heads are not individually critical; MLP appears to do the heavy lifting |
| **Activation patching** | Patching **token embeddings** drops both task accuracies to 0.0; patching **pos embeddings** leaves both at 1.0 | Op-token embedding carries task-specific information; positional embeddings are fully shared |

**Preliminary conclusion (Phase 1):** Addition and subtraction appear to share the majority of their internal circuit. Both tasks grok simultaneously, use overlapping Fourier frequencies, and are sensitive to the same components (token embeddings, MLP). This is consistent with the hypothesis that algebraically related operations (add/sub differ only by negation mod p) reuse the same underlying computation.

> ⚠️ **Caveat:** These are preliminary results from a single training run. Replication across seeds and more rigorous causal analysis (e.g., path patching) needed before making strong claims.

---

### Phase 2 — Multi-task: Addition + Subtraction + Multiplication 🔄 *In progress*

Multiplication mod p is algebraically unrelated to add/sub — it does not decompose as simple clock-face rotation. The question is whether adding it breaks the shared circuit or forces the model to learn a second, separate one.

---

### Upcoming

- **Phase 3:** Add a non-abelian operation (permutation composition in S₅ or S₆) — the sharpest structural break.
- **Phase 4 (stretch):** Scale to more operations; study capacity trade-offs.

---

## Repo structure

```
mech-interp/
├── README.md               ← you are here; contains result summary
├── requirements.txt        ← pip installs for Colab (transformer_lens<3)
├── src/
│   ├── data.py             ← synthetic dataset generation (multi-op, op-token)
│   ├── model.py            ← tiny HookedTransformer via TransformerLens
│   ├── train.py            ← training loop, AdamW + high weight decay
│   └── analysis.py         ← Fourier analysis + activation patching utilities
├── notebooks/
│   ├── phase0_modular_addition.ipynb       ← Phase 0: single-task baseline
│   └── phase1_add_sub_circuit_sharing.ipynb ← Phase 1: add+sub, circuit sharing
├── results/
│   ├── loss_curves.png                     ← Phase 0 grokking
│   ├── fourier_spectrum.png                ← Phase 0 Fourier circuit
│   ├── phase1_grokking_curves.png          ← Phase 1 per-task grokking
│   ├── phase1_fourier_spectrum.png         ← Phase 1 Fourier spectrum
│   ├── phase1_head_ablation.png            ← Phase 1 head ablation
│   └── phase1_activation_patching.png     ← Phase 1 activation patching
└── docs/
    └── proposal.md         ← living research proposal; updated as project evolves
```

## How to run (Colab)

1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**
2. Upload the notebook for the phase you want to run (start with `phase0_...`, then `phase1_...`)
3. **Runtime → Change runtime type → T4 GPU**
4. Run **Cell 1** (installs dependencies) → **Runtime → Restart session** → run **Cell 2 onward**
5. The final Summary cell prints all findings and confirms which result PNGs were saved

> **Important:** After running, download the result PNGs from the Colab file browser (left panel → Files) and commit them to `results/` so they appear in this README.

## Proof-of-work workflow

- Commit after every meaningful step — small, frequent commits with clear messages are the evidence of ongoing work.
- Download result PNGs from Colab and commit them to `results/` after each experiment.
- Update `docs/proposal.md` as findings evolve — this becomes the seed of the final report.
