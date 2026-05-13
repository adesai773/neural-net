import numpy as np
import pytest

from neural_net.loss_function import MseLoss
from neural_net.model import Model
from neural_net.node import Node


class MyModel(Model):
    def forward(self, x: Node) -> list[Node]:
        return [x]


@pytest.fixture
def my_model() -> MyModel:
    return MyModel()


def test_my_model_call(my_model: MyModel):
    assert my_model() == []


def test_my_model_parameters(my_model: MyModel):
    assert my_model.parameters() == []


def test_my_model_train(my_model: MyModel):
    assert my_model.train(np.array(0), np.array(0), MseLoss()) is None


def test_my_model_predict(my_model: MyModel):
    assert my_model.predict() == []


def test_my_model_forward(my_model: MyModel):
    assert my_model.forward(Node(np.array(0)))[0].data == np.array(0)
