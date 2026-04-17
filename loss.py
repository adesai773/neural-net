from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
