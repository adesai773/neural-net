import matplotlib.pyplot as plt
import numpy as np

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import Mse
from neural_net.model import Model
from neural_net.node import Node


class MyRegressionModel(Model):
    def __init__(
        self,
        hidden_dim: int = 16,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        child_rngs = rng.spawn(2)
        self._hidden_dim = hidden_dim

        self.linear1 = Linear(1, hidden_dim, seed=child_rngs[0])
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, 1, seed=child_rngs[1])

    def forward(self, x: Node) -> Node:
        linear1_out = self.linear1(x)
        relu_out = self.relu(linear1_out)
        linear2_out = self.linear2(relu_out)

        return linear2_out


def main():
    rng = np.random.default_rng(7)
    N = 500
    X_train = rng.uniform(0, 1, size=(N, 1))
    y_true = X_train**2 + 1 + rng.normal(loc=0.0, scale=0.03, size=X_train.shape)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = X_test**2 + 1

    hidden_dims = [4, 8, 32, 256]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, h in zip(axes.flat, hidden_dims, strict=True):
        model = MyRegressionModel(hidden_dim=h, seed=42)
        print(model)
        losses = model.train(
            x_train=X_train,
            y_true=y_true,
            loss=Mse(),
            optimizer_key="sgd",
            num_epochs=3000,
            batch_size=40,
            learning_rate=0.04,
            shuffle=True,
            seed=np.random.default_rng(7),  # fresh rng per model for fair comparison
        )
        y_pred = model.predict(X_test)[0]

        ax.scatter(X_train, y_true, color="gray", alpha=0.3, s=8, label="Training data")
        ax.plot(X_test, y_test, "k--", linewidth=1.5, label="Truth")
        ax.plot(X_test, y_pred, color="crimson", linewidth=2, label="Prediction")
        ax.set_title(f"hidden_dim={h}  (final loss={losses[-1]:.5f})")
        ax.grid(True, linestyle=":", alpha=0.5)

    # Single legend, single set of axis labels — cleaner than repeating on every subplot
    axes[0, 0].legend(loc="upper left", fontsize=9)
    for ax in axes[-1, :]:  # bottom row
        ax.set_xlabel("x")
    for ax in axes[:, 0]:  # left column
        ax.set_ylabel("y")

    fig.suptitle("Regression fit vs. hidden dimension", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
