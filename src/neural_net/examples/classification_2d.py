import matplotlib.pyplot as plt
import numpy as np

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import BceWithLogits
from neural_net.model import Model
from neural_net.node import Node
from neural_net.optimizer import Adam


class MyClassificationModel(Model):
    def __init__(
        self,
        hidden_dim: int = 16,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        child_rngs = rng.spawn(2)

        self.linear1 = Linear(2, hidden_dim, seed=child_rngs[0])
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, 1, seed=child_rngs[1])

    def forward(self, x: Node) -> Node:
        linear1_out = self.linear1(x)
        relu_out = self.relu(linear1_out)
        linear2_out = self.linear2(relu_out)

        return linear2_out


def target_condition(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Multi-region ground truth: wavy stripe + two isolated islands."""
    return (
        (np.abs(x2 - 0.4 * np.sin(3 * np.pi * x1)) < 0.25)  # wavy horizontal stripe
        | ((x1 - 0.7) ** 2 + (x2 + 0.7) ** 2 < 0.05)  # island bottom-right
        | ((x1 + 0.7) ** 2 + (x2 - 0.7) ** 2 < 0.05)  # island top-left
    )


def make_labels(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.where(target_condition(x1, x2), 1, 0)[:, np.newaxis]


def train_and_evaluate(
    X_train: np.ndarray,
    y_true: np.ndarray,
    hidden_dim: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[MyClassificationModel, list[float]]:
    model = MyClassificationModel(hidden_dim=hidden_dim, seed=42)
    losses = model.train(
        x_train=X_train,
        y_true=y_true,
        loss=BceWithLogits(),
        optimizer=Adam(
            model.parameters(),
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        ),
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        seed=np.random.default_rng(7),
    )
    return model, losses


def main():
    rng = np.random.default_rng(7)
    N = 3000
    X_train = rng.uniform(-1, 1, size=(N, 2))
    y_true = make_labels(X_train)

    N_test = 500
    X_test = rng.uniform(-1, 1, size=(N_test, 2))
    y_test = make_labels(X_test)

    # Dense grid for both the ground-truth boundary and the model's decision heatmap
    grid_size = 400
    xx, yy = np.meshgrid(
        np.linspace(-1.05, 1.05, grid_size),
        np.linspace(-1.05, 1.05, grid_size),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_truth = target_condition(xx, yy).astype(float)

    hidden_dims = [8, 32, 128]

    fig, axes = plt.subplots(2, len(hidden_dims), figsize=(17, 11))

    for col_idx, h in enumerate(hidden_dims):
        model, losses = train_and_evaluate(
            X_train,
            y_true,
            hidden_dim=h,
            num_epochs=2000,
            batch_size=32,
            learning_rate=0.005,
        )

        y_pred_logits = model.predict(X_test)[0]
        y_pred_labels = (y_pred_logits > 0).astype(int)
        test_accuracy = float(np.mean(y_pred_labels == y_test))

        y_train_pred_logits = model.predict(X_train)[0]
        y_train_pred_labels = (y_train_pred_logits > 0).astype(int)
        train_accuracy = float(np.mean(y_train_pred_labels == y_true))

        print(
            f"h={h:3d}: "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
            f"train_acc={train_accuracy:.1%}, "
            f"test_acc={test_accuracy:.1%}"
        )

        # Model predictions over the grid — used for the decision-boundary heatmap
        grid_logits = model.predict(grid_points)[0]
        grid_probs = 1 / (1 + np.exp(-grid_logits))
        grid_probs_2d = grid_probs.reshape(grid_size, grid_size)

        # Top row: training data colored by model's predicted class
        ax_task = axes[0, col_idx]
        ax_task.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train_pred_labels.ravel(),
            cmap="RdBu_r",
            s=8,
            alpha=0.5,
            edgecolor="none",
            label="Training data",
        )
        # Ground-truth boundary via 0.5-contour of the boolean condition
        ax_task.contour(
            xx,
            yy,
            grid_truth,
            levels=[0.5],
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )
        ax_task.set_title(
            f"h={h} — Training data by predicted class "
            f"(train accuracy = {train_accuracy:.1%})"
        )
        ax_task.set_xlabel("$x_1$")
        ax_task.set_ylabel("$x_2$")
        ax_task.set_aspect("equal")
        ax_task.set_xlim(-1.05, 1.05)
        ax_task.set_ylim(-1.05, 1.05)
        ax_task.grid(True, linestyle=":", alpha=0.4)

        # Bottom row: model's learned probability surface
        ax_pred = axes[1, col_idx]
        im = ax_pred.contourf(
            xx,
            yy,
            grid_probs_2d,
            levels=np.linspace(0, 1, 41),
            cmap="RdBu_r",
            alpha=0.75,
        )
        ax_pred.contour(
            xx, yy, grid_probs_2d, levels=[0.5], colors="black", linewidths=2
        )
        ax_pred.contour(
            xx,
            yy,
            grid_truth,
            levels=[0.5],
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )
        ax_pred.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test.ravel(),
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=25,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.3,
            marker="*",
        )
        ax_pred.set_title(
            f"h={h} — Model prediction (test accuracy = {test_accuracy:.1%})"
        )
        ax_pred.set_xlabel("$x_1$")
        ax_pred.set_ylabel("$x_2$")
        ax_pred.set_aspect("equal")
        ax_pred.set_xlim(-1.05, 1.05)
        ax_pred.set_ylim(-1.05, 1.05)
        plt.colorbar(im, ax=ax_pred, label="P(class = 1)", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Classification: multi-region task across hidden dimensions", fontsize=14
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
