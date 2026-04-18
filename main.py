from layer import LinearLayer, ReLU
from loss_function import MseLoss
from neural_network import NeuralNetwork
from optimizer import SgdOptimizer


def main():

    neural_net = NeuralNetwork(optimizer=SgdOptimizer(), loss_fn=MseLoss())
    neural_net.add_layer(LinearLayer())
    neural_net.add_layer(ReLU())
    print(neural_net)


if __name__ == "__main__":
    main()
