import numpy as np

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import Mse
from neural_net.model import Model
from neural_net.node import Node


class MyModel(Model):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        seed: int | None = None,
    ):
        hidden_dims = hidden_dims or []

        rng = np.random.default_rng(seed)
        child_rngs = rng.spawn(len(hidden_dims) + 1)
        hidden_dims = [input_dim, *hidden_dims, output_dim]

        self.linear_layers = []
        for i in range(len(hidden_dims) - 1):
            self.linear_layers.append(
                Linear(hidden_dims[i], hidden_dims[i + 1], seed=child_rngs[i])
            )
        self.relu = ReLU()

    def forward(self, x: Node) -> Node:
        for linear in self.linear_layers[:-1]:
            x = linear(x)
            x = self.relu(x)

        return self.linear_layers[-1](x)


def main():
    rng = np.random.default_rng(7)
    N = 1000
    input_dim = 10
    output_dim = 4
    hidden_dims = [100, 50, 25, 9]

    X_train = rng.uniform(0, 1, size=(N, input_dim))

    # 1. Random weight matrix (shape: input_dim x output_dim) and bias (shape: output_dim)
    W = rng.normal(loc=0.0, scale=1.0, size=(input_dim, output_dim))
    b = rng.normal(loc=0.0, scale=1.0, size=(output_dim,))

    # 2. Add Gaussian noise (adjust noise_std to control noise level)
    noise_std = 0.1
    noise = rng.normal(loc=0.0, scale=noise_std, size=(N, output_dim))

    # 3. Noisy linear combination: y = X @ W + b + noise
    y_true = X_train @ W + b + noise

    model = MyModel(input_dim, hidden_dims, output_dim, seed=42)

    losses = model.train(
        x_train=X_train,
        y_true=y_true,
        loss=Mse(),
        optimizer_key="sgd",
        num_epochs=2000,
        batch_size=64,
        learning_rate=0.01,
        shuffle=True,
        seed=rng,
    )
    print(losses[0], losses[-1])


if __name__ == "__main__":
    main()
