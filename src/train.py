"""
Training loop for the grokking setup: full-batch gradient descent, AdamW with
high weight decay (grokking is very sensitive to this hyperparameter -- sweep
it if you don't see the phase transition), and periodic checkpointing so you
can later study *how* the circuit forms across training, not just the end state.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model(
    model,
    train_data,
    test_data,
    epochs: int = 20000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    checkpoint_every: int = 200,
    device: str = "cuda",
):
    train_inputs, train_labels = train_data
    test_inputs, test_labels = test_data
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )

    history = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    checkpoints = []

    for epoch in tqdm(range(epochs)):
        model.train()
        logits = model(train_inputs)[:, -1, :]  # prediction at the '=' position
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % checkpoint_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_inputs)[:, -1, :]
                test_logits = model(test_inputs)[:, -1, :]
                train_loss = F.cross_entropy(train_logits, train_labels).item()
                test_loss = F.cross_entropy(test_logits, test_labels).item()
                train_acc = (train_logits.argmax(-1) == train_labels).float().mean().item()
                test_acc = (test_logits.argmax(-1) == test_labels).float().mean().item()

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            checkpoints.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    return history, checkpoints
