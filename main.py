from neural_net.layer import LinearLayer, ReLU
from neural_net.loss_function import MseLoss
from neural_net.neural_network import NeuralNetwork
from neural_net.optimizer import SgdOptimizer


def main():

    neural_net = NeuralNetwork(optimizer=SgdOptimizer(), loss_fn=MseLoss())
    neural_net.add_layer(LinearLayer())
    neural_net.add_layer(ReLU())
    print(neural_net)


if __name__ == "__main__":
    main()
