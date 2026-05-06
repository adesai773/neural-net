import numpy as np

from neural_net.loss_function import MseLoss
from neural_net.model import Model
from neural_net.node import Node


class MyModel(Model):
    def forward(self, x: Node) -> list[Node]:
        return [x]


def main():
    neural_net = MyModel()
    neural_net.train(
        x_train=np.array(0), y_true=np.array(0), loss=[MseLoss()], optimizer_key="sgd"
    )
    print(neural_net)


if __name__ == "__main__":
    main()
