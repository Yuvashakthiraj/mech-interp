"""
Tiny transformer for algorithmic tasks, built with TransformerLens so that
hooks, caching, and activation patching are available out of the box for the
interpretability phase.
"""
from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(
    vocab_size: int,
    seq_len: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    n_layers: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """
    A small transformer following the standard 'grokking' setup (Nanda et al.,
    Power et al.): 1 layer, a handful of heads, no LayerNorm (keeps the math
    cleaner for later weight/Fourier analysis).
    """
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=vocab_size,
        n_ctx=seq_len,
        act_fn="relu",
        normalization_type=None,  # LayerNorm-free: makes weight analysis cleaner
        seed=seed,
        device=device,
    )
    model = HookedTransformer(cfg)
    return model
