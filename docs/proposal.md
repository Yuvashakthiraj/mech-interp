# Project Proposal (Draft — update as the project evolves)

## Working title
Multi-Task Circuit Sharing in Small Transformers Trained on Modular Arithmetic

## Abstract (draft)
Mechanistic interpretability has produced detailed, verified accounts of how
small transformers solve single algorithmic tasks such as modular addition
(Nanda et al., 2023) and finite group composition (Chughtai et al., 2023;
Stander et al., 2023). Far less is known about what happens when a single
model is trained on *several* related algorithmic tasks at once: does it
build one shared circuit, separate circuits per task, or something in
between? This project trains small transformers on combinations of modular
addition, subtraction, multiplication, and (time permitting) a non-abelian
group operation, and uses activation patching, weight/Fourier analysis, and
ablation to directly test how much circuitry is shared across tasks, and how
that sharing changes as the algebraic structure of the tasks diverges.

## Research question
Does a small transformer trained on multiple modular arithmetic operations
develop shared internal circuitry across operations that share algebraic
structure (e.g. addition/subtraction), and does that sharing break down for
operations with different structure (e.g. multiplication, or a non-abelian
operation)?

## Related work
- Power et al. (2022) — first documented the "grokking" phenomenon on
  algorithmic tasks.
- Nanda, Chan, Lieberum, Smith & Steinhardt (2023), *Progress Measures for
  Grokking via Mechanistic Interpretability* (ICLR) — fully reverse-engineers
  a 1-layer transformer trained on modular addition; the Fourier-circuit
  analysis this project's Phase 0 replicates.
- Chughtai, Chan & Nanda (2023), *A Toy Model of Universality: Reverse
  Engineering How Networks Learn Group Operations* (ICML) — extends circuit
  analysis to general finite groups, including non-abelian ones.
- Stander, Yu, Fan & Biderman (2023), *Grokking Group Multiplication with
  Cosets* — replicates Chughtai et al.'s S5/S6 setup but reports a different
  underlying mechanism; a useful reminder that even single-task circuit
  claims in this area are still contested.
- [Add more as you find them — check for anything specifically on multi-task
  or multi-operation grokking/circuit-sharing before finalizing your novelty
  claim; this subfield is moving fast (new papers appearing through 2026).]

## Method
1. **Phase 0 — Replication.** Train a 1-layer, attention+MLP transformer on
   modular addition mod 113 alone; confirm grokking and the known Fourier
   circuit. (De-risks the technical pipeline before adding complexity.)
2. **Phase 1 — Two related tasks.** Add subtraction via a second op-token;
   retrain; use activation patching to test whether addition and subtraction
   share components.
3. **Phase 2 — A structurally different task.** Add multiplication mod p;
   repeat the analysis; test whether sharing holds up.
4. **Phase 3 — Non-abelian task.** Add composition in a small permutation
   group (S5 or S6, following Chughtai et al.'s setup); test whether any
   sharing survives a genuine structural break.
5. **Phase 4 (stretch) — Scaling tasks.** Add more operations and study
   capacity trade-offs / interference as the number of tasks grows.

## Planned experiments
- Grokking curves (train/test accuracy vs. epoch) for each task combination.
- Activation patching between tasks at each layer/head to quantify shared
  causal contribution.
- Fourier/weight analysis per task to compare which frequencies/components
  each operation relies on.
- Ablation: zero out components used by one task, measure degradation on
  the others.

## Novelty framing (to refine after the literature check)
Existing work reverse-engineers circuits for *single* operations or single
group structures. The multi-task setting — same model, several operations,
explicit measurement of circuit overlap — is the gap this project targets.

## Timeline (adjust as you go)
- Weeks 1–2: Phase 0 replication, working infrastructure, first GitHub commits.
- Weeks 3–4: Literature deep-dive on multi-task/interference in grokking;
  finalize novelty framing.
- Weeks 5–7: Phase 1 (addition + subtraction) + patching analysis.
- Weeks 8–10: Phase 2 (multiplication) + comparison.
- Weeks 11–13: Phase 3 (non-abelian task), the sharpest test.
- Weeks 14+: Write-up, polish plots, (stretch) Phase 4.

## Evidence log
Keep a running list here of what's been done, linking to commits/results —
this section is your fastest way to answer "what have you done so far" in
any check-in or supervisor email.

- [ ] Phase 0 environment set up on Colab
- [ ] Phase 0 grokking reproduced
- [ ] Fourier circuit confirmed
- [ ] Phase 1 started
