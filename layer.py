from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
