from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        return "Layer()"


class LinearLayer(Layer):
    def forward(self):
        pass

    def __str__(self) -> str:
        return "LinearLayer()"


class ReLU(Layer):
    def forward(self):
        pass

    def __str__(self) -> str:
        return "ReLU()"
