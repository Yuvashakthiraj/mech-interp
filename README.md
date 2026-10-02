# What Determines Circuit Sharing in Modular Arithmetic Transformers?

**Research question:** When a small transformer learns modular arithmetic, does the algebraic structure
of the operation determine the structure of its internal circuit — and therefore determine whether two
operations *can* share computational resources, or must build separate representations?

This project investigates circuit reuse through a progression: first establish what circuit a single
operation builds (Phases 0 and 3), then measure what happens when the model must handle multiple
operations simultaneously (Phases 1 and 2). The hypothesis is that algebraically related operations
(addition and subtraction) share circuits, while algebraically distinct ones (multiplication) do not —
and that the Fourier frequency basis of each operation's embedding is the mechanism that drives this.

This is a final-year undergraduate mechanistic interpretability project extending:
- Power et al. (2022) — first documented "grokking" on algorithmic tasks.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR) — fully
  reverse-engineers a 1-layer transformer on modular addition; the Fourier-circuit result this project
  replicates in Phase 0.
- Chughtai et al. (2023), *A Toy Model of Universality* (ICML) — extends circuit analysis to finite
  group operations.
- Stander et al. (2023), *Grokking Group Multiplication with Cosets* — contradicting follow-up to
  Chughtai et al.; a reminder that this subfield is still actively contested.

---

## Results

### Phase 0 — Single-task baseline: Modular Addition ✅

A 1-layer, 4-head transformer (d_model=128) trained on modular addition mod 113.
Establishes the reference circuit against which all later phases are compared.

| | |
|---|---|
| ![Loss curves](results/loss_curves.png) | ![Fourier spectrum](results/fourier_spectrum.png) |

**Key findings:**
- **Grokking confirmed:** train accuracy ~100% within a few hundred epochs; test accuracy near 0% until
  epoch ~2000–4000, then a sharp jump to ~100% — the characteristic grokking phase transition.
- **Sparse Fourier circuit confirmed:** the number-token embedding matrix has a small number of sharply
  dominant frequency spikes ({9, 47, 49}) against a near-zero background. This matches Nanda et al. 2023
  and establishes the baseline circuit signature for addition.

---

### Phase 1 — Multi-task: Addition + Subtraction ✅

Same architecture trained on **both** addition and subtraction simultaneously, with an operation token
in the input to disambiguate tasks (`[a, op_token, b, =]`). Tests whether algebraically related
operations — subtraction is addition with a negated argument mod p — share the same circuit.

| | |
|---|---|
| ![Grokking curves](results/phase1_grokking_curves.png) | ![Fourier spectrum](results/phase1_fourier_spectrum.png) |
| ![Activation patching](results/phase1_activation_patching.png) | |

**Key findings:**

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **Grokking** | Both tasks reach 100% at the **same epoch (~6200)** | Simultaneous grokking = a single shared learning event, not two separate ones |
| **Fourier spectrum** | Dominant frequencies **{9, 47, 49}** — identical to Phase 0, not doubled | The model reuses the addition Fourier circuit for subtraction rather than building a second one |
| **Activation patching (MLP)** | Patching one task's MLP output into the other drops accuracy to 0.0 | The MLP is the answer-writing site and encodes the final result — consistent with grokking literature |
| **Activation patching (pos embed)** | Patching positional embeddings leaves both tasks at 1.0 | Positional embeddings are fully shared — expected sanity check, confirms hook setup is correct |
| **Head ablation** | ⚠️ Invalid — see note | Measurement failure; fixed for Phase 2 onward |

> ⚠️ **Phase 1 measurement note:** Head ablation and attention output patching returned all-zero drops
> and NaN respectively. Root cause: `HookedTransformerConfig` was missing `use_attn_result=True`, so
> `hook_result` was registered as a hook name but never materialised as a tensor — zeroing it was a
> silent no-op. Identified after Phase 1, fixed in `src/model.py`. Grokking, Fourier, and MLP patching
> results are unaffected.

**Phase 1 conclusion:** Addition and subtraction share their internal circuit. Both grok simultaneously,
use identical Fourier frequencies, and are served by the same MLP computation. This is consistent with
their algebraic proximity: `a − b ≡ a + (p−b) mod p`.

---

### Phase 2 — Stress test: Addition + Subtraction + Multiplication ✅

Tests whether the shared add/sub circuit extends to a third, algebraically unrelated operation.
Multiplication mod p does not decompose as clock-face rotation — it requires a fundamentally different
computational strategy related to the discrete logarithm.

| | |
|---|---|
| ![Grokking curves](results/phase2_grokking_curves.png) | ![Fourier spectrum](results/phase2_fourier_spectrum.png) |
| ![Activation patching](results/phase2_activation_patching.png) | |

**Key findings:**

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **Grokking** | None of 3 tasks grokked (add 34%, sub 9%, mul 20% after 25k epochs) | The add/sub shared circuit cannot accommodate multiplication — the model stays in memorisation |
| **Fourier spectrum** | 8 spread-out medium spikes vs Phase 1's 3 clean dominant ones | The Fourier circuit fragmented: no single elegant representation covers all three operations |
| **Activation patching** | add↔sub retain 15–22% overlap; add↔mul and sub↔mul only 3–7% | Even partially trained, subtraction shares far more with addition than multiplication does |

**Phase 2 conclusion:** Three operations simultaneously exceeded this model's representational capacity.
The result is strong evidence that multiplication's structural difference is not just quantitative
(harder) but qualitative (incompatible): it disrupts the circuit rather than extending it.
The natural follow-up is to characterise exactly what circuit multiplication *does* use on its own.

---

### Phase 3 — Single-task baseline: Modular Multiplication ✅

Mirrors Phase 0 exactly, but for multiplication. A single-task multiplication model is trained and its
Fourier circuit characterised. The direct comparison between Phase 0 (addition) and Phase 3
(multiplication) reveals whether the two operations use the same internal frequency basis — and
therefore whether they *could* share a circuit in principle.

| | |
|---|---|
| ![Mul grokking](results/phase3_mul_grokking.png) | ![Fourier comparison](results/phase3_fourier_comparison.png) |

**Key findings:**

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **Grokking** | Single-task multiplication grokked | Multiplication is learnable alone — the Phase 2 failure was interference between incompatible circuits, not an inability to learn mul at all |
| **Fourier frequencies** | Multiplication's dominant frequencies show minimal/zero overlap with addition's {9, 47, 49} | Addition and multiplication operate in **different Fourier subspaces** |

> **Phase 3 finding:** The Fourier circuit for multiplication is structurally distinct from the
> addition circuit. This is the mechanistic explanation for Phase 2's capacity collapse: when forced
> to coexist in one model, the two circuits interfere because they require different frequency bases.
> Algebraic structure directly determines which Fourier frequencies a model uses — add and sub share
> a frequency basis (explaining Phase 1's clean shared circuit), while add and mul do not (explaining
> Phase 2's fragmentation).

---

## Summary of findings across phases

| Phase | Setup | Key result |
|-------|-------|-----------|
| 0 | Addition alone | Grokks; sparse Fourier circuit at frequencies {9, 47, 49} |
| 1 | Addition + Subtraction | Both grokk simultaneously; same 3 frequencies; shared circuit confirmed |
| 2 | Addition + Sub + Multiplication | Nothing grokks; Fourier circuit fragments; mul is incompatible |
| 3 | Multiplication alone | Grokks; uses a **different** Fourier frequency basis than addition |

**Overall conclusion:** Whether two operations share a circuit is predicted by whether they share a
Fourier frequency basis. Algebraically related operations (add/sub) use the same basis and share
circuits naturally. Algebraically distinct operations (add/mul) use incompatible bases — forcing them
into one model causes the circuit to collapse rather than extend.

---

## Repo structure

```
mech-interp/
├── README.md                    ← you are here
├── requirements.txt             ← pip installs for Colab (transformer_lens<3)
├── src/
│   ├── data.py                  ← dataset generation (multi-op, op-token encoding)
│   ├── model.py                 ← tiny HookedTransformer via TransformerLens
│   ├── train.py                 ← training loop, AdamW + high weight decay
│   └── analysis.py              ← Fourier analysis + activation patching utilities
├── notebooks/
│   ├── phase0_modular_addition.ipynb         ← Phase 0: single-task addition
│   ├── phase1_add_sub_circuit_sharing.ipynb  ← Phase 1: addition + subtraction
│   ├── phase2_add_sub_mul.ipynb              ← Phase 2: three-operation stress test
│   └── phase3_add_mul_comparison.ipynb       ← Phase 3: single-task multiplication
├── results/
│   ├── loss_curves.png                  ← Phase 0 grokking curve
│   ├── fourier_spectrum.png             ← Phase 0 Fourier circuit
│   ├── phase1_grokking_curves.png       ← Phase 1 per-task grokking
│   ├── phase1_fourier_spectrum.png      ← Phase 1 Fourier spectrum (shared)
│   ├── phase1_activation_patching.png  ← Phase 1 patching results
│   ├── phase2_grokking_curves.png       ← Phase 2 grokking (none succeeded)
│   ├── phase2_fourier_spectrum.png      ← Phase 2 fragmented Fourier circuit
│   ├── phase2_activation_patching.png  ← Phase 2 pairwise patching
│   ├── phase3_mul_grokking.png          ← Phase 3 single-task mul grokking
│   └── phase3_fourier_comparison.png   ← Phase 3 add vs mul Fourier comparison
└── docs/
    └── proposal.md              ← living research proposal; updated as project evolves
```

## How to run (Colab / Kaggle)

1. Open [colab.research.google.com](https://colab.research.google.com) or [kaggle.com](https://kaggle.com/code) (Kaggle has a separate free GPU quota)
2. Upload the notebook for the phase you want to run
3. Set runtime to **T4 GPU** (Runtime → Change runtime type)
4. Run **Cell 1** (installs dependencies) → **Runtime → Restart session** → run **Cell 2 onward**
5. The final summary cell confirms which result PNGs were saved

> **After running:** download result PNGs from the Colab/Kaggle file browser and commit them to
> `results/` so they render in this README on GitHub.

## Proof-of-work notes

- Commits are made after every meaningful experimental step — the commit history is evidence of
  ongoing iterative work, not a single batch upload.
- Measurement failures (e.g., the Phase 1 head ablation bug) are documented inline rather than
  silently corrected — this is standard scientific practice.
- `docs/proposal.md` is updated as findings evolve and serves as the seed of the final report.
