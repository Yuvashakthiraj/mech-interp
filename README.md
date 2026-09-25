# Multi-Task Circuit Sharing in Modular Arithmetic Transformers

**Research question:** When a small transformer is trained on several modular
arithmetic operations at once (addition, subtraction, multiplication, and
eventually a non-abelian group operation), does it learn one shared internal
circuit, separate circuits per operation, or something in between?

This repo is the working codebase for a final-year mechanistic interpretability
project, extending:
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic
  Interpretability* (ICLR) — single-operation modular addition.
- Chughtai et al. (2023), *A Toy Model of Universality: Reverse Engineering
  How Networks Learn Group Operations* (ICML) — group composition tasks.
- Stander et al. (2023), *Grokking Group Multiplication with Cosets* — a
  contradicting follow-up to Chughtai et al., worth reading for how open this
  subfield still is.

## Repo structure

```
mech-interp-project/
├── README.md              <- you are here
├── requirements.txt        <- pip installs for Colab
├── src/
│   ├── data.py             <- synthetic dataset generation
│   ├── model.py             <- tiny transformer (via TransformerLens)
│   ├── train.py              <- training loop, checkpointing
│   └── analysis.py            <- Fourier analysis + activation patching
├── notebooks/
│   └── phase0_modular_addition.ipynb   <- self-contained Colab notebook, start here
└── docs/
    └── proposal.md          <- living project proposal / mini-paper draft
```

## How to run (Colab)

1. Open Google Colab (colab.research.google.com), New Notebook, or upload
   `notebooks/phase0_modular_addition.ipynb` directly (File → Upload notebook).
2. Runtime → Change runtime type → GPU (T4 is fine, this project is tiny).
3. Run all cells top to bottom. The notebook installs dependencies, generates
   data, trains a 1-layer transformer on modular addition mod 113, and plots
   the loss curves + a first Fourier analysis of the embeddings.
4. Once Phase 0 works and reproduces the grokking curve, move to the
  multi-task extension described in `docs/proposal.md` Phase 1 onward.

## Proof-of-work workflow

- Commit after every meaningful step (a working data generator, a training
  run that completes, a plot that shows something) — small, frequent commits
  with real messages are your evidence of ongoing work, not one giant commit
  at the end.
- Save loss curves, checkpoints, and analysis plots into a `results/` folder
  (gitignore large checkpoint files if they get big; keep the plots).
- Update `docs/proposal.md` as your understanding evolves — this becomes the
  seed of your actual final report/paper.
