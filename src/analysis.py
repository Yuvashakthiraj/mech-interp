"""
Core interpretability tools for this project:
  1. Fourier analysis of the number-token embeddings (the key trick from
     Nanda et al. for reading off the circuit modular addition uses).
  2. A basic activation-patching function using TransformerLens's caching, for
     testing whether two operations share circuitry (Phase 1 onward).
"""
import numpy as np
import torch


def fourier_embedding_analysis(model, p: int):
    """
    Take the embedding matrix for the number tokens [0, p-1], apply a DFT
    across the number dimension, and return the power spectrum -- if the
    model has learned the Fourier-based circuit found in prior work, a small
    number of frequencies should dominate.
    """
    W_E = model.W_E[:p].detach().cpu().numpy()  # [p, d_model]
    fourier_basis = np.fft.rfft(W_E, axis=0)  # transform across the p numbers
    power = np.abs(fourier_basis) ** 2
    power_per_freq = power.sum(axis=1)  # collapse over d_model
    return power_per_freq  # shape [p//2 + 1]; look for a handful of large spikes


def get_cache(model, inputs):
    """Run the model and return logits + full activation cache."""
    logits, cache = model.run_with_cache(inputs)
    return logits, cache


def patch_activation(model, clean_inputs, corrupted_inputs, hook_name, patch_fn=None):
    """
    Runs the model on corrupted_inputs, but at hook_name, replaces the
    activation with the corresponding activation from a clean run on
    clean_inputs. Use this to test whether components used for one operation
    (e.g. addition) are causally involved in another (e.g. subtraction) --
    the core tool for the circuit-sharing question.

    If the two operations share the relevant circuitry, patching should
    produce structured, interpretable changes in the output (not noise).
    """
    _, clean_cache = model.run_with_cache(clean_inputs)
    clean_act = clean_cache[hook_name]

    def hook_fn(activation, hook):
        if patch_fn is not None:
            return patch_fn(activation, clean_act)
        return clean_act

    patched_logits = model.run_with_hooks(
        corrupted_inputs, fwd_hooks=[(hook_name, hook_fn)]
    )
    return patched_logits
