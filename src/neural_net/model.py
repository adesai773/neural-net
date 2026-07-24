from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from tqdm import tqdm

from .layer import Add
from .loss_function import LossFunction
from .node import Node
from .optimizer import Optimizer


class Model(ABC):
    def __call__(self, *args: Node | np.ndarray) -> list[Node]:
        input_nodes = [
            Node(arg, requires_grad=False) if not isinstance(arg, Node) else arg
            for arg in args
        ]

        output_nodes = self.forward(*input_nodes)
        return output_nodes if isinstance(output_nodes, list) else [output_nodes]

    def parameters(self) -> list[Node]:
        all_params: list[Node] = []
        for layer in vars(self).values():
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters())

        return all_params

    def train(
        self,
        x_train: np.ndarray | list[np.ndarray],
        y_true: np.ndarray | list[np.ndarray],
        loss: LossFunction | list[LossFunction],
        optimizer_key: str = "sgd",
        num_epochs: int = 5,
        batch_size: int | None = None,
        learning_rate: float = 0.01,
        shuffle: bool = False,
        seed: int | np.random.Generator | None = None,
    ) -> list[float]:
        # Basic input validation
        if isinstance(x_train, list):
            assert len(x_train) > 0
        if isinstance(y_true, list):
            assert len(y_true) > 0
        if isinstance(loss, list):
            assert len(loss) > 0
        assert optimizer_key in Optimizer.REGISTRY
        assert num_epochs >= 1
        assert batch_size is None or batch_size >= 1
        assert learning_rate > 0

        # rng for shuffling; deterministic when seed is provided
        rng = np.random.default_rng(seed)

        # Optimizer setup
        optimizer = Optimizer.REGISTRY[optimizer_key](
            parameters=self.parameters(), learning_rate=learning_rate
        )

        # Convert input/output/loss to list to standardize
        x_train = [x_train] if not isinstance(x_train, list) else x_train
        y_true = [y_true] if not isinstance(y_true, list) else y_true
        loss = [loss] if not isinstance(loss, list) else loss

        # Ensure shapes are valid (i.e. they have a batch dimension)
        num_samples = int(x_train[0].shape[0])
        for x in x_train:
            assert x.ndim >= 2, (
                f"x_train must have >= 2 dimensions (i.e. first dim is batch dim), current: {x.ndim}"
            )
            assert x.shape[0] == num_samples
        for y in y_true:
            assert y.ndim >= 2, (
                f"y_true must have >= 2 dimensions (i.e. first dim is batch dim), current: {y.ndim}"
            )
            assert y.shape[0] == num_samples

        # Sanity check forward pass before training loop
        one_sample_x = [x[:1] for x in x_train]
        y_probe = self(*one_sample_x)
        # Check if number of loss functions matches number of outputs
        assert len(y_probe) == len(loss), (
            f"Model produces {len(y_probe)} outputs but received {len(loss)} loss functions"
        )
        assert len(loss) == len(y_true), (
            f"Need 1 LossFunction per output Node, current {len(loss)} LossFunctions, but {len(y_true)} expected output nodes"
        )
        # Check if forward pass's output shape matches ground truth samples' shapes
        for i, (yp, yt) in enumerate(zip(y_probe, y_true)):
            assert yp.data.shape[1:] == yt.shape[1:], (
                f"Output {i} shape {yp.data.shape[1:]} doesn't match target shape {yt.shape[1:]}"
            )

        # If batch_size is unset, use full batch gradient descent
        if batch_size is None:
            batch_size = num_samples

        # Training loop
        epoch_avg_losses: list[float] = []
        pbar = tqdm(range(num_epochs), desc="Training")
        for _ in pbar:
            indices = (
                rng.permutation(num_samples) if shuffle else np.arange(num_samples)
            )

            epoch_avg_losses.append(0)
            batch_count: int = 0

            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]

                # Extract shuffled batches
                x_train_batch = [x[batch_idx] for x in x_train]
                y_true_batch = [y[batch_idx] for y in y_true]

                # Forward pass
                y_pred: list[Node] = self(*x_train_batch)

                # Compute per-output loss
                loss_nodes: list[Node] = []
                for prediction, ground_truth, loss_func in zip(
                    y_pred, y_true_batch, loss
                ):
                    loss_nodes.append(loss_func(prediction, ground_truth))

                # Sum to obtain total loss
                total_loss = Add()(*loss_nodes)

                # Reset gradients
                optimizer.zero_grad()

                # Backpropagation
                total_loss.backward()
                epoch_avg_losses[-1] += float(total_loss.data)
                batch_count += 1

                # Gradient Descent
                optimizer.step()

            epoch_avg_losses[-1] /= batch_count
            pbar.set_postfix({"loss": f"{epoch_avg_losses[-1]:.4f}"})  # type: ignore[reportUnknownArgumentType]

        return epoch_avg_losses

    def predict(self, *args: np.ndarray) -> list[np.ndarray]:
        out_nodes: list[Node] = self(*args)
        return [node.data for node in out_nodes]

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Node | list[Node]:
        raise NotImplementedError("Users must subclass Model and define graph.")

    def __str__(self) -> str:
        return "Model()"
