from typing import Any

import numpy as np
import pytest

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import Mse
from neural_net.model import Model
from neural_net.node import Node
from neural_net.optimizer import Adam, Momentum, Optimizer, RMSprop, Sgd


class MySimpleModel(Model):
    def forward(self, x: Node) -> Node | list[Node]:
        return x


class MyMultiInputModel(Model):
    def forward(self, x1: Node, x2: Node) -> Node | list[Node]:
        return x1 + x2


class MyMultiOutputModel(Model):
    def __init__(self):
        self.linear1 = Linear(2, 3, seed=42)
        self.linear2 = Linear(2, 3, seed=42)

    def forward(self, x: Node) -> Node | list[Node]:
        linear1_out = self.linear1(x)
        linear2_out = self.linear2(x)
        return [linear1_out, linear2_out]


class MyNonLinearModel(Model):
    def __init__(self):
        self.linear = Linear(2, 1, seed=42)
        self.relu = ReLU()

    def forward(self, x: Node) -> Node | list[Node]:
        linear1_out = self.linear(x)
        relu_out = self.relu(linear1_out)
        return relu_out


class MyMultiInOutModel(Model):
    def __init__(self):
        self.linear1 = Linear(2, 1, seed=42)
        self.linear2 = Linear(3, 1, seed=43)
        self.relu = ReLU()

    def forward(self, x1: Node, x2: Node) -> Node | list[Node]:
        out1 = self.relu(self.linear1(x1))
        out2 = self.relu(self.linear2(x2))
        return [out1, out2]


@pytest.fixture
def my_simple_model() -> MySimpleModel:
    return MySimpleModel()


@pytest.fixture
def my_multi_input_model() -> MyMultiInputModel:
    return MyMultiInputModel()


@pytest.fixture
def my_multi_output_model() -> MyMultiOutputModel:
    return MyMultiOutputModel()


@pytest.fixture
def my_non_linear_model() -> MyNonLinearModel:
    return MyNonLinearModel()


@pytest.fixture
def my_multi_in_out_model() -> MyMultiInOutModel:
    return MyMultiInOutModel()


def test_model_parameters_simple(my_simple_model: MySimpleModel):
    assert my_simple_model.parameters() == []


def test_model_parameters_multi_input(my_multi_input_model: MyMultiInputModel):
    assert my_multi_input_model.parameters() == []


def test_model_parameters_multi_output(my_multi_output_model: MyMultiOutputModel):
    expected_params = [
        my_multi_output_model.linear1.W_node,
        my_multi_output_model.linear1.b_node,
        my_multi_output_model.linear2.W_node,
        my_multi_output_model.linear2.b_node,
    ]
    assert my_multi_output_model.parameters() == expected_params


def test_model_parameters_non_linear(my_non_linear_model: MyNonLinearModel):
    expected_params = [
        my_non_linear_model.linear.W_node,
        my_non_linear_model.linear.b_node,
    ]
    assert my_non_linear_model.parameters() == expected_params


def test_model_parameters_list():
    class MyModel(Model):
        def __init__(self):
            self.layers = [Linear(2, 3, seed=42), Linear(2, 3, seed=42)]

        def forward(self, x: Node) -> Node | list[Node]:
            return x

    model = MyModel()
    expected_params = [
        model.layers[0].W_node,
        model.layers[0].b_node,
        model.layers[1].W_node,
        model.layers[1].b_node,
    ]

    assert model.parameters() == expected_params


def test_model_call_wraps_ndarray(my_simple_model: MySimpleModel):
    out_nodes = my_simple_model(np.array(0))

    assert len(out_nodes) == 1
    assert not out_nodes[0].requires_grad


def test_model_call_wraps_ndarray_non_linear(my_non_linear_model: MyNonLinearModel):
    out_nodes = my_non_linear_model(np.zeros((2, 2)))

    assert len(out_nodes) == 1
    assert out_nodes[0].requires_grad


def test_model_call_accepts_node(my_simple_model: MySimpleModel):
    in_node = Node(np.array(0))
    out_nodes = my_simple_model(in_node)

    assert len(out_nodes) == 1
    assert out_nodes[0] == in_node


def test_model_call_multi_input(my_multi_input_model: MyMultiInputModel):
    out_nodes = my_multi_input_model(Node(np.array(1)), np.array(3))

    assert len(out_nodes) == 1
    assert np.array_equal(out_nodes[0].data, np.array(4))


def test_model_call_multi_output(my_multi_output_model: MyMultiOutputModel):
    in_node = Node(np.ones((2, 2)))
    out_nodes = my_multi_output_model(in_node)

    assert len(out_nodes) == 2
    assert out_nodes[0].parents[0] == in_node
    assert out_nodes[1].parents[0] == in_node


def test_model_predict_returns_arrays(my_simple_model: MySimpleModel):
    out_array = my_simple_model.predict(np.array(0))

    assert isinstance(out_array, list)
    assert len(out_array) == 1
    assert isinstance(out_array[0], np.ndarray)


def test_model_predict_returns_arrays_multi_output(
    my_multi_output_model: MyMultiOutputModel,
):
    out_arrays = my_multi_output_model.predict(np.ones((1, 2)))

    assert isinstance(out_arrays, list)
    assert len(out_arrays) == 2
    assert isinstance(out_arrays[0], np.ndarray)
    assert isinstance(out_arrays[1], np.ndarray)


def assert_loss_reduced(epoch_losses: list[float], factor: float = 0.1) -> None:
    assert epoch_losses[-1] < factor * epoch_losses[0], (
        f"loss only went from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}"
    )


@pytest.mark.parametrize(
    ("optimizer_cls", "optimizer_kwargs", "batch_size", "shuffle"),
    [
        pytest.param(Sgd, {"learning_rate": 0.02}, None, False, id="full_batch"),
        pytest.param(Sgd, {"learning_rate": 0.02}, 500, False, id="mini_batch"),
        pytest.param(Sgd, {"learning_rate": 0.02}, 500, True, id="mini_batch_shuffled"),
        pytest.param(Sgd, {"learning_rate": 0.02}, 600, False, id="ragged_last_batch"),
        pytest.param(
            Momentum,
            {"learning_rate": 0.02, "momentum": 0.9},
            None,
            False,
            id="momentum",
        ),
        pytest.param(
            RMSprop,
            {"learning_rate": 0.01, "decay": 0.99},
            None,
            False,
            id="rmsprop",
        ),
        pytest.param(
            Adam,
            {"learning_rate": 0.01, "beta1": 0.9, "beta2": 0.999},
            None,
            False,
            id="adam",
        ),
    ],
)
def test_model_train_reduces_loss(
    my_non_linear_model: MyNonLinearModel,
    optimizer_cls: type[Optimizer],
    optimizer_kwargs: dict[str, Any],
    batch_size: int | None,
    shuffle: bool,
):
    rng = np.random.default_rng(7)

    N = 2000
    X = rng.uniform(0, 1, size=(N, 2))

    coeffs = np.array([[2.0], [1.0]])
    y = X @ coeffs + 0.5

    optimizer = optimizer_cls(my_non_linear_model.parameters(), **optimizer_kwargs)

    n_epochs = 2000
    epoch_losses = my_non_linear_model.train(
        X,
        y,
        loss=Mse(),
        optimizer=optimizer,
        num_epochs=n_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=rng,
    )

    assert len(epoch_losses) == n_epochs
    assert_loss_reduced(epoch_losses)
    np.testing.assert_allclose(
        my_non_linear_model.linear.W_node.data, coeffs, atol=0.05
    )
    np.testing.assert_allclose(my_non_linear_model.linear.b_node.data, 0.5, atol=0.05)


def test_model_train_seed():
    rng = np.random.default_rng(7)

    model_1 = MyNonLinearModel()
    model_2 = MyNonLinearModel()
    model_3 = MyNonLinearModel()

    N = 200
    X = rng.uniform(0, 1, size=(N, 2))

    coeffs = np.array([[2.0], [1.0]])
    y = X @ coeffs + 0.5

    n_epochs = 10
    epoch_losses_1 = model_1.train(
        X,
        y,
        loss=Mse(),
        optimizer=Sgd(model_1.parameters(), learning_rate=0.01),
        num_epochs=n_epochs,
        batch_size=1,
        shuffle=True,
        seed=10,
    )
    epoch_losses_2 = model_2.train(
        X,
        y,
        loss=Mse(),
        optimizer=Sgd(model_2.parameters(), learning_rate=0.01),
        num_epochs=n_epochs,
        batch_size=1,
        shuffle=True,
        seed=10,
    )
    epoch_losses_3 = model_3.train(
        X,
        y,
        loss=Mse(),
        optimizer=Sgd(model_3.parameters(), learning_rate=0.01),
        num_epochs=n_epochs,
        batch_size=1,
        shuffle=True,
        seed=11,
    )

    assert epoch_losses_1 == epoch_losses_2
    np.testing.assert_array_equal(
        model_1.linear.W_node.data, model_2.linear.W_node.data
    )
    np.testing.assert_array_equal(
        model_1.linear.b_node.data, model_2.linear.b_node.data
    )

    assert epoch_losses_1 != epoch_losses_3


@pytest.mark.parametrize(
    ("batch_size"),
    [
        pytest.param(None, id="full_batch"),
        pytest.param(600, id="ragged_last_batch"),
    ],
)
def test_model_train_with_multi_in_out(
    my_multi_in_out_model: MyMultiInOutModel, batch_size: int | None
):
    rng = np.random.default_rng(7)

    N = 2000

    X1 = rng.uniform(0, 1, size=(N, 2))
    X2 = rng.uniform(0, 1, size=(N, 3))

    coeffs1 = np.array([[2.0], [1.0]])
    coeffs2 = np.array([[1.0], [3.0], [0.5]])
    y1 = X1 @ coeffs1 + 0.5
    y2 = X2 @ coeffs2 + 0.25

    n_epochs = 2000
    epoch_losses = my_multi_in_out_model.train(
        [X1, X2],
        [y1, y2],
        loss=[Mse(), Mse()],
        optimizer=Sgd(my_multi_in_out_model.parameters(), learning_rate=0.02),
        num_epochs=n_epochs,
        batch_size=batch_size,
        shuffle=False,
        seed=rng,
    )

    assert len(epoch_losses) == n_epochs
    assert_loss_reduced(epoch_losses, factor=0.01)

    np.testing.assert_allclose(
        my_multi_in_out_model.linear1.W_node.data, coeffs1, atol=0.05
    )
    np.testing.assert_allclose(
        my_multi_in_out_model.linear2.W_node.data, coeffs2, atol=0.05
    )

    np.testing.assert_allclose(
        my_multi_in_out_model.linear1.b_node.data, 0.5, atol=0.05
    )
    np.testing.assert_allclose(
        my_multi_in_out_model.linear2.b_node.data, 0.25, atol=0.05
    )


@pytest.fixture
def model_kwargs() -> dict[str, Any]:
    return {
        "x_train": np.zeros((4, 2)),
        "y_true": np.zeros((4, 1)),
        "loss": Mse(),
        "optimizer": Sgd([]),
        "num_epochs": 1,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"num_epochs": 0}, id="zero_epochs"),
        pytest.param({"y_true": np.zeros(4)}, id="y_missing_batch_dim"),
        pytest.param({"loss": [Mse(), Mse()]}, id="too_many_losses"),
        pytest.param({"loss": []}, id="zero_losses"),
        pytest.param({"x_train": []}, id="empty_input"),
        pytest.param({"y_true": []}, id="empty_output"),
        pytest.param({"batch_size": 0}, id="batch_size_0"),
        pytest.param({"x_train": np.zeros(4)}, id="x_without_batch_dim"),
        pytest.param({"y_true": np.zeros((3, 1))}, id="sample_count_mismatch"),
        pytest.param({"y_true": np.zeros((4, 2))}, id="output_shape_mismatch"),
        pytest.param(
            {"y_true": [np.zeros((4, 1)), np.zeros((4, 1))]},
            id="too_many_outputs",
        ),
    ],
)
def test_model_train_invalid_args(
    my_non_linear_model: MyNonLinearModel,
    model_kwargs: dict[str, Any],
    overrides: dict[str, Any],
):
    model_kwargs.update(overrides)

    with pytest.raises(AssertionError):
        my_non_linear_model.train(**model_kwargs)


def test_model_train_args_should_pass(
    my_non_linear_model: MyNonLinearModel, model_kwargs: dict[str, Any]
):
    # Should not raise
    epoch_losses = my_non_linear_model.train(**model_kwargs)
    assert isinstance(epoch_losses, list)
    assert len(epoch_losses) == model_kwargs["num_epochs"]
