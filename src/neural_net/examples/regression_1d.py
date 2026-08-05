import matplotlib.pyplot as plt
import numpy as np

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import Mse
from neural_net.model import Model
from neural_net.node import Node
from neural_net.optimizer import Adam, Momentum, RMSprop, Sgd


class MyRegressionModel(Model):
    def __init__(
        self,
        hidden_dim: int = 16,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        child_rngs = rng.spawn(2)

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
    N = 1000
    X_train = rng.uniform(0, 1, size=(N, 1))
    y_true = (
        np.sin(2 * np.pi * X_train)
        + 0.3 * np.sin(6 * np.pi * X_train)
        + X_train**3
        + 1.5
        + rng.normal(loc=0.0, scale=0.1, size=X_train.shape)
    )

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = (
        np.sin(2 * np.pi * X_test) + 0.3 * np.sin(6 * np.pi * X_test) + X_test**3 + 1.5
    )

    hidden_dims = [8, 32, 128]
    optimizers = [
        ("SGD", lambda p: Sgd(p, learning_rate=0.015)),
        ("Momentum", lambda p: Momentum(p, learning_rate=0.03, momentum=0.9)),
        (
            "RMSprop",
            lambda p: RMSprop(p, learning_rate=0.005, decay=0.99, epsilon=1e-8),
        ),
        (
            "Adam",
            lambda p: Adam(
                p, learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8
            ),
        ),
    ]

    fig, axes = plt.subplots(
        len(hidden_dims), len(optimizers), figsize=(18, 10), sharex=True, sharey=True
    )

    for row_idx, h in enumerate(hidden_dims):
        for col_idx, (name, optimizer_factory) in enumerate(optimizers):
            model = MyRegressionModel(hidden_dim=h, seed=42)
            optimizer = optimizer_factory(model.parameters())
            print(model)
            print(optimizer)

            losses = model.train(
                x_train=X_train,
                y_true=y_true,
                loss=Mse(),
                optimizer=optimizer,
                num_epochs=500,
                batch_size=16,
                shuffle=True,
                seed=np.random.default_rng(
                    7
                ),  # fresh rng per model for fair comparison
            )
            y_pred = model.predict(X_test)[0]

            ax = axes[row_idx, col_idx]
            ax.scatter(
                X_train, y_true, color="gray", alpha=0.3, s=8, label="Training data"
            )
            ax.plot(X_test, y_test, "k--", linewidth=1.5, label="Truth")
            ax.plot(X_test, y_pred, color="crimson", linewidth=2, label="Prediction")

            # Optimizer name as column header on the top row only
            if row_idx == 0:
                ax.set_title(name, fontsize=12)

            # Per-cell final loss, tucked in the top-left corner
            ax.text(
                0.03,
                0.97,
                f"loss={losses[-1]:.4f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )

            ax.grid(True, linestyle=":", alpha=0.5)

    # Hidden dim as row labels on the leftmost column
    for row_idx, h in enumerate(hidden_dims):
        axes[row_idx, 0].set_ylabel(
            f"h={h}", rotation=0, labelpad=30, fontsize=11, fontweight="bold"
        )

    # x labels only on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("x")

    # Single shared legend below the grid — pulled from any subplot since all match
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False)

    fig.suptitle("Regression fit: hidden dim vs. optimizer", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # leave room at bottom for the legend
    plt.show()


if __name__ == "__main__":
    main()
