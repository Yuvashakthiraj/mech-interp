"""
Synthetic dataset generation for modular arithmetic tasks.

Each example is a short sequence: [a, op_token, b, equals_token] -> predict result.
Multiple operations can be mixed into one dataset by giving each its own op_token,
which is how we set up the multi-task circuit-sharing experiments.
"""
import numpy as np
import torch

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
}


def make_dataset(p: int = 113, ops=("add",), train_frac: float = 0.3, seed: int = 0):
    """
    Build a dataset covering all (a, b) pairs in Z_p for each requested op.

    Token layout:
      [0, p-1]              -> numbers 0..p-1
      [p, p+len(ops)-1]     -> one token per operation, in the order given
      p+len(ops)            -> '=' token

    Returns:
      train_data, test_data: each a tuple (inputs [N,4] long tensor, labels [N] long tensor)
      vocab_size: total vocabulary size
      equals_token: the integer id of the '=' token
      op_list: the list of op names in the order their tokens were assigned
    """
    rng = np.random.RandomState(seed)
    op_list = list(ops)
    equals_token = p + len(op_list)

    all_inputs = []
    all_labels = []
    for op_idx, op_name in enumerate(op_list):
        op_token = p + op_idx
        fn = OPS[op_name]
        for a in range(p):
            for b in range(p):
                result = fn(a, b, p)
                all_inputs.append([a, op_token, b, equals_token])
                all_labels.append(result)

    inputs = torch.tensor(all_inputs, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)

    n = len(inputs)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_data = (inputs[train_idx], labels[train_idx])
    test_data = (inputs[test_idx], labels[test_idx])

    vocab_size = p + len(op_list) + 1
    return train_data, test_data, vocab_size, equals_token, op_list


if __name__ == "__main__":
    # quick smoke test
    train, test, vocab, eq, ops = make_dataset(p=113, ops=("add", "sub"), train_frac=0.3)
    print("vocab size:", vocab, "equals token:", eq, "ops:", ops)
    print("train examples:", train[0].shape[0], "test examples:", test[0].shape[0])
    print("sample input/label:", train[0][0].tolist(), train[1][0].item())
